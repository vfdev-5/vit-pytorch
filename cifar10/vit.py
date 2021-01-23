import torch
import torch.nn as nn


class VisionAttention(nn.Module):
    """Vision Multi-Head Attention layer with trainable parameters:
    - W_q, W_k, W_k : embed_dim * embed_dim * 3
    - W_o : embed_dim * embed_dim
    
    .. code-block:: text

        x, (B, N, embed_dim) -> 
                    -> Q = X * W_q, (B, N, embed_dim) -> 
                    -> K = X * W_k, (B, N, embed_dim) ->
                    -> V = X * W_v, (B, N, embed_dim) ->
                    -> A = softmax(Q @ K^t * scale), (B, N, N) -> Dropout(A) ->
                    -> H = A @ V, (B, N, embed_dim) -> 
                    -> y = H @ W_o, (B, N, embed_dim) ->
                    -> output
                    
    https://github.com/google/flax/blob/master/flax/nn/attention.py
    """
    
    def __init__(self, embed_dim, num_heads=8, qkv_bias=True, attn_drop=0.):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.scale = float(head_dim) ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.att_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        attention = (Q @ K.transpose(1, 2)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.att_drop(attention)
        heads = attention @ V
        output = self.proj(heads)
        return output


class MLP(nn.Module):
    """The positionwise feed-forward network or MLP.    
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderBlock(nn.Module):
    """ViT EncoderBlock
    
    https://github.com/google-research/vision_transformer/blob/9dbeb0269e0ed1b94701c30933222b49189aa33c/vit_jax/models.py#L94
    """
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.lnorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.lnorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = VisionAttention(
            embed_dim, num_heads=num_heads, attn_drop=attn_drop_rate
        )
        self.dropout = nn.Dropout(drop_rate)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), drop_rate=drop_rate)
        
    def forward(self, x):
        y = self.lnorm1(x)
        y = self.attention(y)
        y = self.dropout(y)
        x = x + y
        
        y = self.lnorm2(x)
        y = self.mlp(y)
        return x + y

    
class VisionTransformer(nn.Module):
    """VisionTransformer model
    
    https://github.com/google-research/vision_transformer/
    """
    
    def __init__(
        self, 
        patch_size=16,
        hidden_size=768,
        input_channels=3,
        input_size=224,
        num_classes=1000,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        drop_rate=0.1,
        attn_drop_rate=0.0,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            input_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        self.pos_dropout = nn.Dropout(p=drop_rate)
    
        # Define encoder blocks
        kwargs = {
            "embed_dim": hidden_size,
            "num_heads": num_heads,
            "mlp_ratio": mlp_dim / hidden_size,
            "drop_rate": drop_rate,
            "attn_drop_rate": attn_drop_rate,
        }
        blocks = [EncoderBlock(**kwargs) for _ in range(num_layers)]
        self.blocks = nn.Sequential(*blocks)
        self.lnorm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.mlp_head = nn.Linear(hidden_size, num_classes)

        nn.init.normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def features(self, x):
        patches = self.patch_embed(x)
        patches = patches.flatten(start_dim=2)
        patches = patches.transpose(1, 2)
        
        batch_size = patches.shape[0]

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, patches], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        x = self.blocks(x)
        x = self.lnorm(x)
        
        # Return the first token
        return x[:, 0, ...]
    
    def forward(self, x):
        f = self.features(x)
        y = self.mlp_head(f)
        return y
    

def vit_b16(num_classes=1000, input_channels=3, input_size=224):
    return VisionTransformer(
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=input_size,
        patch_size=16,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        drop_rate=0.1,
        attn_drop_rate=0.0,
    )
