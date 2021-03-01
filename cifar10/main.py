from pathlib import Path
from datetime import datetime

import fire

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

import ignite
import ignite.distributed as idist
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, DiskSaver
from ignite.utils import manual_seed, setup_logger

from ignite.contrib.engines import common
from ignite.contrib.handlers import PiecewiseLinear, ProgressBar

import utils


def training(local_rank, config):

    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)
    device = idist.device()

    output_path = logger_filepath = config["output_path"]
    if rank == 0:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")

        folder_name = f"{config['model']}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
        output_path = Path(output_path) / folder_name
        if not output_path.exists():
            output_path.mkdir(parents=True)
        config["output_path"] = output_path.as_posix()

        if "cuda" in device.type:
            config["cuda device name"] = torch.cuda.get_device_name(local_rank)

        logger_filepath = (output_path / "logs").as_posix()

    logger = setup_logger(name="CIFAR10-Training", filepath=logger_filepath, distributed_rank=local_rank)

    log_basic_info(logger, config)

    # Setup dataflow, model, optimizer, criterion
    train_loader, test_loader = get_dataflow(config)

    config["num_iters_per_epoch"] = len(train_loader)
    model, optimizer, criterion, lr_scheduler = initialize(config)

    logger.info(f"# model parameters (M): {sum([m.numel() for m in model.parameters()]) * 1e-6}")

    # Create trainer for current task
    trainer = create_trainer(model, optimizer, criterion, lr_scheduler, train_loader.sampler, config, logger)

    # Let's now setup evaluator engine to perform model's validation and compute metrics
    metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(criterion),
    }

    # We define two evaluators as they wont have exactly similar roles:
    # - `evaluator` will save the best model based on validation score
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    if config["with_pbar"] and rank == 0:
        ProgressBar(desc="Evaluation (train)", persist=False).attach(train_evaluator)
        ProgressBar(desc="Evaluation (val)", persist=False).attach(evaluator)

    def run_validation(engine):
        epoch = trainer.state.epoch
        with autocast(enabled=config["with_amp"]):
            state = train_evaluator.run(train_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Train", state.metrics)
        with autocast(enabled=config["with_amp"]):
            state = evaluator.run(test_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Test", state.metrics)

    ev = Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.COMPLETED
    trainer.add_event_handler(ev, run_validation)

    if rank == 0:
        # Setup TensorBoard logging on trainer and evaluators. Logged values are:
        #  - Training metrics, e.g. running average loss values
        #  - Learning rate
        #  - Evaluation train/test metrics
        evaluators = {"training": train_evaluator, "test": evaluator}
        tb_logger = common.setup_tb_logging(output_path, trainer, optimizer, evaluators=evaluators)

    # Store 1 best model by validation accuracy:
    common.gen_save_best_models_by_val_score(
        save_handler=DiskSaver(config["output_path"], require_empty=False),
        evaluator=evaluator,
        models={"model": model},
        metric_name="accuracy",
        n_saved=1,
        trainer=trainer,
        tag="test",
    )

    try:
        trainer.run(train_loader, max_epochs=config["num_epochs"])
    except Exception as e:
        import traceback

        print(traceback.format_exc())

    if rank == 0:
        tb_logger.close()


def run(
    seed=543,
    data_path="/tmp/cifar10",
    output_path="/tmp/output-cifar10/",
    model="vit_b3_224x244",
    optimizer="adam",
    batch_size=128,
    weight_decay=1e-4,
    num_workers=4,
    num_epochs=200,
    learning_rate=0.001,
    num_warmup_epochs=0,
    validate_every=3,
    checkpoint_every=1000,
    backend=None,
    resume_from=None,
    nproc_per_node=None,
    with_pbar=False,
    with_amp=False,
    rescaled_size=None,
    **spawn_kwargs,
):
    """Main entry to train an model on CIFAR10 dataset.

    Args:
        seed (int): random state seed to set. Default, 543.
        data_path (str): input dataset path. Default, "/tmp/cifar10".
        output_path (str): output path. Default, "/tmp/output-cifar10".
        model (str): model name (from torchvision) to setup model to train. Default, "resnet18".
        batch_size (int): total batch size. Default, 512.
        optimizer (str): optimizer name. Possible values: "sgd", "adam", "adamw". Default, "adam".
        weight_decay (float): weight decay. Default, 1e-4.
        num_workers (int): number of workers in the data loader. Default, 12.
        num_epochs (int): number of epochs to train the model. Default, 24.
        learning_rate (float): peak of piecewise linear learning rate scheduler. Default, 0.4.
        num_warmup_epochs (int): number of warm-up epochs before learning rate decay. Default, 4.
        validate_every (int): run model's validation every ``validate_every`` epochs. Default, 3.
        checkpoint_every (int): store training checkpoint every ``checkpoint_every`` iterations. Default, 200.
        backend (str, optional): backend to use for distributed configuration. Possible values: None, "nccl", "xla-tpu",
            "gloo" etc. Default, None.
        nproc_per_node (int, optional): optional argument to setup number of processes per node. It is useful,
            when main python process is spawning training as child processes.
        resume_from (str, optional): path to checkpoint to use to resume the training from. Default, None.
        with_pbar(bool): if True adds a progress bar on training iterations.
        with_amp(bool): if True uses torch native AMP
        rescaled_size (int, optional): if provided then input image will be rescaled to that value.
        **spawn_kwargs: Other kwargs to spawn run in child processes: master_addr, master_port, node_rank, nnodes

    """
    # catch all local parameters
    config = locals()
    config.update(config["spawn_kwargs"])
    del config["spawn_kwargs"]

    spawn_kwargs["nproc_per_node"] = nproc_per_node

    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:

        parallel.run(training, config)


def get_dataflow(config):
    # - Get train/test datasets
    if idist.get_rank() > 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()

    train_dataset, test_dataset = utils.get_train_test_datasets(config["data_path"])

    if idist.get_rank() == 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = idist.auto_dataloader(
        train_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=True, drop_last=True,
    )

    test_loader = idist.auto_dataloader(
        test_dataset, batch_size=2 * config["batch_size"], num_workers=config["num_workers"], shuffle=False,
    )
    return train_loader, test_loader


def initialize(config):
    model = utils.get_model(config["model"])
    # Adapt model for distributed backend if provided
    model = idist.auto_model(model)

    optimizer = utils.get_optimizer(
        config["optimizer"], model, learning_rate=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    # Adapt optimizer for distributed backend if provided
    optimizer = idist.auto_optim(optimizer)
    criterion = nn.CrossEntropyLoss().to(idist.device())

    le = config["num_iters_per_epoch"]
    milestones_values = [
        (0, 0.0),
        (le * config["num_warmup_epochs"], config["learning_rate"]),
        (le * config["num_epochs"], 0.0),
    ]
    lr_scheduler = PiecewiseLinear(optimizer, param_name="lr", milestones_values=milestones_values)

    return model, optimizer, criterion, lr_scheduler


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {int(elapsed)} - {tag} metrics:\n {metrics_output}")


def log_basic_info(logger, config):
    logger.info(f"Train {config['model']} on CIFAR10")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("Distributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


def create_trainer(model, optimizer, criterion, lr_scheduler, train_sampler, config, logger):

    device = idist.device()

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    scaler = GradScaler(enabled=config["with_amp"])

    def train_step(engine, batch):

        x, y = batch[0], batch[1]

        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        model.train()

        with autocast(enabled=config["with_amp"]):
            y_pred = model(x)
            loss = criterion(y_pred, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if idist.backend() == "horovod":
            optimizer.synchronize()
            with optimizer.skip_synchronize():
                scaler.step(optimizer)
                scaler.update()
        else:
            scaler.step(optimizer)
            scaler.update()

        return {
            "batch loss": loss.item(),
        }

    trainer = Engine(train_step)
    trainer.logger = logger

    if config["with_pbar"] and idist.get_rank() == 0:
        ProgressBar().attach(trainer)

    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
    metric_names = [
        "batch loss",
    ]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        save_handler=DiskSaver(config["output_path"], require_empty=False),
        lr_scheduler=lr_scheduler,
        output_names=metric_names,
        with_pbars=False,
        clear_cuda_cache=False,
    )

    resume_from = config["resume_from"]
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        logger.info(f"Resume from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


if __name__ == "__main__":
    fire.Fire({"run": run})
