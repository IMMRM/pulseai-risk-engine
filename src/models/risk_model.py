"""
Fusion model combining LSTM and MLP outputs into a risk score.
"""
import torch
import torch.nn as nn

from src.models.lstm_encoder import LSTMEncoder
from src.models.tabular_encoder import TabularEncoder

class RiskModel(nn.Module):
    def __init__(self, vocab_size, num_features=9, lstm_hidden=64,tabular_hidden=32):
        super().__init__()
        self.lstm_encoder = LSTMEncoder(vocab_size, hidden_size=lstm_hidden)
        self.tabular_encoder = TabularEncoder(num_features=num_features, hidden_dim=tabular_hidden)
        self.fc = nn.Linear(lstm_hidden + tabular_hidden, 1)  
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,sequences,tabular):
        lstm_out=self.lstm_encoder(sequences)
        tabular_out=self.tabular_encoder(tabular)
        combined=torch.cat((lstm_out,tabular_out),dim=1)
        risk_score=self.fc(combined)
        return risk_score.squeeze(1)  # Return a 1D tensor of risk scores
if __name__ == "__main__":
    # fake batch of 4 customers
    dummy_seq = torch.randint(0, 20, (4, 100))   # 4 customers, 100 events
    dummy_tab = torch.randn(4, 9)                # 4 customers, 9 features

    model = RiskModel(vocab_size=20)
    out = model(dummy_seq, dummy_tab)

    print("Output shape:", out.shape)   # want: torch.Size([4, 1])
    print("Sample scores:", out.squeeze().tolist())  # 4 numbers between 0 and 1