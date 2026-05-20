import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

class STH_MAE(nn.Module):
    """
    Spatial-Temporal Heatmap Masked Autoencoder (STH-MAE).
    Processes dense 3D Heatmap Volumes.
    """
    def __init__(self, target_shape=(32, 32, 32), patch_size=(4, 4, 4), in_channels=1, embed_dim=768, depth=12, decoder_depth=4):
        super(STH_MAE, self).__init__()
        
        self.target_shape = target_shape
        self.patch_size = patch_size
        
        D, H, W = target_shape
        pD, pH, pW = patch_size
        
        self.num_patches = (D // pD) * (H // pH) * (W // pW)
        
        # 3D Patch Embedding
        patch_dim = in_channels * pD * pH * pW
        self.patch_embed = nn.Sequential(
            Rearrange('b c (d pd) (h ph) (w pw) -> b (d h w) (pd ph pw c)', 
                      pd=pD, ph=pH, pw=pW),
            nn.Linear(patch_dim, embed_dim)
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, dim_feedforward=embed_dim*4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, dim_feedforward=embed_dim*4, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        
        # Reconstruction Head
        self.decoder_pred = nn.Linear(embed_dim, patch_dim)
        
    def random_masking(self, x, mask_ratio=0.75):
        """
        Perform random masking on patches.
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # x is (B, C, D, H, W) where D is frames/depth
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        x_encoded = self.encoder(x_masked)
        return x_encoded, mask, ids_restore
        
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        x_ = x_ + self.decoder_pos_embed
        
        x_decoded = self.decoder(x_)
        pred = self.decoder_pred(x_decoded)
        return pred

    def forward(self, imgs, mask_ratio=0.75):
        # T dimension can be treated as Depth for 3D convolutions.
        # In a real model, T and D are different, but for simplification we map T -> D.
        # Shape expected: (B, C, D, H, W)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        
        # Calculate loss (MSE) only on masked patches
        target = self.patch_embed[0](imgs) # Get raw patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # Mean over patch dim
        loss = (loss * mask).sum() / mask.sum() # Mean over masked patches
        
        return loss, pred, mask
