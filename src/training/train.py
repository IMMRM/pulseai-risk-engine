"""
Training loop with early stopping, metric logging, checkpointing.
"""
import torch 
import torch.nn as nn
from pathlib import Path
from src.logger import get_logger
from src.models.risk_model import RiskModel

logger = get_logger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def load_tensors():
    processed=ROOT_DIR / "data" / "processed"
    sequences=torch.load(processed / "sequences.pt")
    tabular=torch.load(processed / "tabular.pt")
    labels=torch.load(processed / "labels.pt")
    return sequences,tabular,labels

def save_checkpoint(model,filename="risk_model_checkpoint.pt"):
    checkpoint_dir=ROOT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    filepath=checkpoint_dir / filename
    torch.save(model.state_dict(),filepath)
    logger.info(f"Checkpoint saved to {filepath}")
    return filepath

def train():
    sequences,tabular,labels=load_tensors()
    # find the vocab size for the LSTM encoder
    vocab_size=int(sequences.max().item())+1
    # create the model
    model=RiskModel(vocab_size=vocab_size)
    # define loss and optimizer
    criterion=nn.BCELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    logger.info(f"vocab_size={vocab_size}")
    logger.info(f"Data shapes: {sequences.shape}, {tabular.shape}, {labels.shape}")
    logger.info("Setup complete — model, loss, optimizer ready.")
    
    # split the data into train and validation sets
    n=len(labels)
    split_idx=int(n*0.8)
    train_seq,val_seq=sequences[:split_idx],sequences[split_idx:]
    train_tab,val_tab=tabular[:split_idx],tabular[split_idx:]
    train_labels,val_labels=labels[:split_idx],labels[split_idx:]
    logger.info(f"Train: {len(train_labels)}, Validation: {len(val_labels)}")
    # training loop
    epochs=30
    for epoch in range(1,epochs+1):
        model.train()
        optimizer.zero_grad()
        predictions=model(train_seq,train_tab)
        loss=criterion(predictions,train_labels.float())
        loss.backward()
        optimizer.step()
        
        #--validation (no training)-
        model.eval()
        with torch.no_grad():
            val_predictions=model(val_seq,val_tab)
            val_loss=criterion(val_predictions,val_labels.float())
        #log for every 5th epoch
        if epoch%5==0:
            logger.info(f"Epoch {epoch}/{epochs} — Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
    save_checkpoint(model)

if __name__ == "__main__":
    train()