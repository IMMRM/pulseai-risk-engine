"""
PyTorch MLP encoder for tabular customer features.
"""
import torch
import torch.nn as nn

class TabularEncoder(nn.Module):
    def __init__(self,num_features=9,hidden_dim=32):
        super().__init__()
        self.layer1=nn.Linear(num_features,hidden_dim)
        self.layer2=nn.Linear(hidden_dim,hidden_dim)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.relu(self.layer1(x))
        x=self.relu(self.layer2(x))
        return x

if __name__ == "__main__":
    dummy = torch.randn(4, 9)          # 4 customers, 9 features each
    model = TabularEncoder()
    out = model(dummy)
    print("Output shape:", out.shape) 