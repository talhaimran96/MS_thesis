import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, tube_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tube_size = tube_size
        
        self.proj = nn.Conv3d(in_chans, embed_dim, 
                              kernel_size=(tube_size, patch_size, patch_size), 
                              stride=(tube_size, patch_size, patch_size))

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        x = self.proj(x)
        # x shape: (B, embed_dim, T//tube_size, H//patch_size, W//patch_size)
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        return x

class VideoMAE(nn.Module):
    """ VideoMAE V2 architecture for pretraining """
    def __init__(self, img_size=224, patch_size=16, tube_size=2, in_chans=3, 
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6):
        super().__init__()
        self.patch_embed = PatchEmbed3D(img_size, patch_size, tube_size, in_chans, embed_dim)
        
        # Num patches
        self.num_frames_per_tube = 16 // tube_size
        self.num_spatial_patches = (img_size // patch_size) ** 2
        self.num_patches = self.num_frames_per_tube * self.num_spatial_patches
        
        # ENCODER
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # DECODER
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=decoder_num_heads, dim_feedforward=decoder_embed_dim*4, batch_first=True, norm_first=True)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # PREDICTOR
        self.pred_dim = in_chans * tube_size * patch_size * patch_size
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.pred_dim, bias=True)

    def generate_tube_mask(self, B, mask_ratio=0.9, device='cuda'):
        """ Generate 90% tube masking mask """
        # We sample masks for the spatial dimensions and replicate across time (tube)
        # mask shape: (B, num_spatial_patches)
        num_keep = int(self.num_spatial_patches * (1 - mask_ratio))
        
        noise = torch.rand(B, self.num_spatial_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first num_keep patches
        mask = torch.ones([B, self.num_spatial_patches], device=device)
        mask[:, :num_keep] = 0 # 0 means keep, 1 means masked
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # Repeat across temporal dimension (tube_size)
        # mask shape becomes (B, num_frames_per_tube, num_spatial_patches)
        mask = mask.unsqueeze(1).repeat(1, self.num_frames_per_tube, 1)
        # flatten to (B, num_patches)
        mask = mask.reshape(B, self.num_patches)
        
        return mask

    def forward_encoder(self, x, mask_ratio=0.9):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Generate tube mask
        mask = self.generate_tube_mask(B, mask_ratio, x.device)
        
        # Add pos embed without cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking: only keep visible patches
        # mask is 0 for keep, 1 for masked
        # we need to select patches where mask == 0
        
        # To gather properly, we can get ids_keep from mask
        # Since mask is repeated temporally, we can't just use spatial shuffle ids directly for the whole 3D sequence
        # We sort the flattened mask to get the keep indices
        ids_shuffle = torch.argsort(mask, dim=1)
        num_keep = int(self.num_patches * (1 - mask_ratio))
        ids_keep = ids_shuffle[:, :num_keep]
        
        # Gather visible patches
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        
        # Append cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
        x_visible = torch.cat((cls_tokens, x_visible), dim=1)
        
        # Apply Transformer blocks
        for blk in self.blocks:
            x_visible = blk(x_visible)
            
        x_visible = self.norm(x_visible)
        return x_visible, mask, ids_keep

    def forward_decoder(self, x, ids_keep):
        B = x.shape[0]
        
        # Embed from encoder dim to decoder dim
        x = self.decoder_embed(x)
        
        # x is currently [B, 1 + num_keep, D]
        # We need to add back the mask tokens to reconstruct full sequence
        mask_tokens = self.mask_token.repeat(B, self.num_patches, 1)
        
        # Gather operation in reverse: put visible patches back into their original locations
        # we only scatter the patch tokens (skip cls token at index 0)
        x_patches = x[:, 1:, :]
        
        # Un-shuffle using scatter
        x_full = torch.scatter(mask_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]), src=x_patches)
        
        # Prepend cls token
        x_full = torch.cat((x[:, :1, :], x_full), dim=1)
        
        # Add decoder pos embed
        x_full = x_full + self.decoder_pos_embed
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        
        x_full = self.decoder_norm(x_full)
        
        # Predict pixel values
        pred = self.decoder_pred(x_full)
        
        # Remove cls token prediction
        pred = pred[:, 1:, :]
        return pred

    def forward(self, x, mask_ratio=0.9):
        # x shape: (B, C, T, H, W)
        x_encoded, mask, ids_keep = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(x_encoded, ids_keep)
        return pred, mask
