import torch
import torch.nn as nn
import numpy as np

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        # input_size: (T, H, W) in patches, e.g. (8, 14, 14) for 16 frames patch 2 and 224 patch 16
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_mask = int(self.num_patches_per_frame * mask_ratio)
        
    def __call__(self):
        # We generate a 2D mask for one frame and replicate it across time
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask # 1 for masked, 0 for visible

class PatchEmbed3D(nn.Module):
    """ 3D Patch Embedding for Video """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tube_size=2):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tube_size = tube_size
        
        self.grid_size = (num_frames // tube_size, img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.proj = nn.Conv3d(in_chans, embed_dim, 
                              kernel_size=(tube_size, patch_size, patch_size), 
                              stride=(tube_size, patch_size, patch_size))
                              
    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = self.proj(x) # (B, embed_dim, T//tube_size, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        return x

class VideoMAE(nn.Module):
    """ Vision Transformer acting as VideoMAE (Encoder + Decoder) """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_frames=16, tube_size=2,
                 embed_dim=768, depth=12, num_heads=12, 
                 decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6):
        super().__init__()
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim, num_frames, tube_size)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False) 
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=embed_dim*4, activation="gelu", 
                                                   batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=decoder_num_heads, 
                                                   dim_feedforward=decoder_embed_dim*4, activation="gelu", 
                                                   batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Output is predicting the pixels inside the tube patch
        self.patch_dim = in_chans * tube_size * patch_size * patch_size
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim, bias=True)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_encoder(self, x, mask):
        x = self.patch_embed(x) 
        
        B, N, D = x.shape
        x = x + self.pos_embed[:, 1:, :]
        
        x_vis = []
        for i in range(B):
            vis_indices = torch.nonzero(mask[i] == 0).squeeze()
            if vis_indices.dim() == 0:
                vis_indices = vis_indices.unsqueeze(0)
            x_vis.append(x[i, vis_indices, :])
        x_vis = torch.stack(x_vis, dim=0) 
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
        x_vis = torch.cat((cls_tokens, x_vis), dim=1)
        
        x_vis = self.encoder(x_vis)
        x_vis = self.norm(x_vis)
        return x_vis
        
    def forward_decoder(self, x, mask):
        B = x.shape[0]
        N = self.patch_embed.num_patches
        
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(B, N, 1) 
        x_full = mask_tokens.clone()
        
        for i in range(B):
            vis_indices = torch.nonzero(mask[i] == 0).squeeze()
            if vis_indices.dim() == 0:
                vis_indices = vis_indices.unsqueeze(0)
            x_full[i, vis_indices, :] = x[i, 1:, :] 
            
        x_full = torch.cat([x[:, :1, :], x_full], dim=1)
        
        x_full = x_full + self.decoder_pos_embed
        
        x_full = self.decoder(x_full)
        x_full = self.decoder_norm(x_full)
        
        x_full = self.decoder_pred(x_full) 
        
        x_full = x_full[:, 1:, :]
        return x_full

    def forward_loss(self, imgs, pred, mask):
        tube_size = self.patch_embed.tube_size
        patch_size = self.patch_embed.patch_size
        B, C, T, H, W = imgs.shape
        p = patch_size
        t = tube_size
        
        target = imgs.view(B, C, T//t, t, H//p, p, W//p, p)
        target = target.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        target = target.view(B, -1, C * t * p * p)
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) 
        
        loss = (loss * mask).sum() / mask.sum() 
        return loss

    def forward(self, imgs, mask):
        latent = self.forward_encoder(imgs, mask)
        pred = self.forward_decoder(latent, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

class VideoMAEForClassification(nn.Module):
    def __init__(self, videomae_model, num_classes=2):
        super().__init__()
        self.videomae = videomae_model
        embed_dim = self.videomae.cls_token.shape[-1]
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, imgs):
        B = imgs.shape[0]
        N = self.videomae.patch_embed.num_patches
        # No masking for downstream classification task
        mask = torch.zeros(B, N, device=imgs.device) 
        
        # Get features from encoder
        features = self.videomae.forward_encoder(imgs, mask)
        
        # Use cls_token for classification
        cls_token = features[:, 0]
        logits = self.fc(cls_token)
        return logits
