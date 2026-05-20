import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x, A):
        # x: (N, C, T, V)
        # A: (3, V, V)
        x = self.conv(x)
        A_sum = A.sum(dim=0) # Sum over spatial configurations
        x = torch.einsum('nctv,vw->nctw', x, A_sum)
        return x

class ST_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, residual=True):
        super(ST_GCN_Block, self).__init__()
        
        self.gcn = GraphConv(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )
        
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res)

class ST_GCN(nn.Module):
    """
    Deep Spatial-Temporal Graph Convolutional Network.
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=128, num_joints=25, edge_importance_weighting=True):
        super(ST_GCN, self).__init__()
        
        self.A = nn.Parameter(torch.randn(3, num_joints, num_joints))
        
        if edge_importance_weighting:
            self.edge_importance = nn.Parameter(torch.ones(self.A.size()))
        else:
            self.register_parameter('edge_importance', None)
            
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        # 10 layers of ST-GCN block as per original paper
        self.layers = nn.ModuleList((
            ST_GCN_Block(in_channels, 64, residual=False),
            ST_GCN_Block(64, 64),
            ST_GCN_Block(64, 64),
            ST_GCN_Block(64, 64),
            ST_GCN_Block(64, 128, stride=2),
            ST_GCN_Block(128, 128),
            ST_GCN_Block(128, 128),
            ST_GCN_Block(128, 256, stride=2),
            ST_GCN_Block(256, 256),
            ST_GCN_Block(256, out_channels)
        ))
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, 128) # 128-dim projection
        )
        
    def forward(self, x):
        N, C, T, V, M = x.size()
        
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        if self.edge_importance is not None:
            A = self.A * self.edge_importance
        else:
            A = self.A
            
        for layer in self.layers:
            x = layer(x, A)
            
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N * M, -1)
        
        proj = self.projection_head(x)
        return proj
