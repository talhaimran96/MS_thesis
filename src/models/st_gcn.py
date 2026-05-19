import torch
import torch.nn as nn
import torch.nn.functional as F

class ST_GCN(nn.Module):
    """
    Simplified Spatial-Temporal Graph Convolutional Network for Skeleton-based Action Recognition.
    Adapted for Contrastive Learning (outputs representations/projections).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_joints, edge_importance_weighting=True):
        super(ST_GCN, self).__init__()
        
        # In a full implementation, we'd use an adjacency matrix (A) for the NTU graph.
        # Here we use a learnable adjacency matrix for simplicity in the mockup.
        self.A = nn.Parameter(torch.randn(3, num_joints, num_joints))
        
        if edge_importance_weighting:
            self.edge_importance = nn.Parameter(torch.ones(self.A.size()))
        else:
            self.register_parameter('edge_importance', None)
            
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=(9, 1), padding=(4, 0))
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=(9, 1), padding=(4, 0))
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 128) # 128-dim projection
        )
        
    def forward(self, x):
        # x shape: (N, C, T, V, M) -> (Batch, Channels, Frames, Joints, Bodies)
        N, C, T, V, M = x.size()
        
        # BN expects (N, C, ...) we can merge V and M for BN
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        # Spatial-Temporal Convolutions
        # Using a simplified graph conv where we multiply by A
        if self.edge_importance is not None:
            A = self.A * self.edge_importance
        else:
            A = self.A
            
        # Simplified: just applying regular convs and a fully connected layer over the joints
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        # Global average pooling over time and joints
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N * M, -1)
        
        # If contrastive learning, we return the projection
        proj = self.projection_head(x)
        return proj
