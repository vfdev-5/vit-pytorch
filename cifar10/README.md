# Train ViT on CIFAR10 with [PyTorch-Ignite](https://github.com/pytorch/ignite)

- on 1 or more GPUs
- compute training/validation metrics
- log learning rate, metrics etc
- save the best model weights

Configurations:

- [x] single GPU
- [x] multi GPUs on a single node

## Requirements:

- [pytorch-ignite](https://github.com/pytorch/ignite): `pip install pytorch-ignite`
- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [tqdm](https://github.com/tqdm/tqdm/): `pip install tqdm`
- [tensorboardx](https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`
- [python-fire](https://github.com/google/python-fire): `pip install fire`
- (Optional) timm: `pip install timm`

## Usage:

Run the example on a single GPU:

```bash
python main.py run
```

For details on accepted arguments:

```bash
python main.py run -- --help
```

If user would like to provide already downloaded dataset, the path can be setup in parameters as

```bash
--data_path="/path/to/cifar10/"
```

### Distributed training

#### Single node, multiple GPUs

Let's start training on a single node with 2 gpus:

```bash
# using torch.distributed.launch
python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py run --backend="nccl"
```

##### Using [Horovod](https://horovod.readthedocs.io/en/latest/index.html) as distributed backend

Please, make sure to have Horovod installed before running.

Let's start training on a single node with 2 gpus:

```bash
# horovodrun
horovodrun -np=2 python -u main.py run --backend="horovod"
```

or

```bash
# using function spawn inside the code
python -u main.py run --backend="horovod" --nproc_per_node=2
```

### Online logs

On TensorBoard.dev: 
