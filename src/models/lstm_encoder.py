"""
PyTorch LSTM sequence encoder for customer event streams.
"""
import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self,vocab_size,embedding_dim=32,hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.lstm= nn.LSTM(embedding_dim,hidden_size,batch_first=True)
    
    def forward(self,x):
        x= self.embedding(x)
        output,(hidden,cell)=self.lstm(x)
        return hidden.squeeze(0)  # Return the last hidden state
    

if __name__ == "__main__":
    # fake batch of 4 customers, 100 events each
    dummy = torch.randint(0, 20, (4, 100))
    model = LSTMEncoder(vocab_size=20)
    out = model(dummy)
    print("Output shape:", out.shape)   # want: torch.Size([4, 64])