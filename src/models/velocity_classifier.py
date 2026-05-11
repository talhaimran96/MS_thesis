import torch
import torch.nn as nn

class VelocityClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super(VelocityClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer (LSTM)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Initialize hidden state and cell state with zeros
        # (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0)) # out shape: (batch_size, seq_len, hidden_size)
        
        # Decode the hidden state of the last time step
        # out[:, -1, :] gets the last time step output for each sequence in the batch
        out = self.fc(out[:, -1, :])
        
        return out
