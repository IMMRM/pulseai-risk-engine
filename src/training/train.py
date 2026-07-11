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
    torch.manual_seed(42)  # for reproducibility
    sequences,tabular,labels=load_tensors()
    # Early stopping setup
    best_val_loss=float('inf')
    patience=7
    patience_counter=0
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
            val_accuracy=compute_accuracy(val_predictions,val_labels.float())
            if val_loss<best_val_loss:
                best_val_loss=val_loss
                patience_counter=0
                save_checkpoint(model,filename="best_model.pt")
                logger.info(f"New best model saved with val_loss={val_loss.item():.4f}")
            else:
                patience_counter+=1
            if patience_counter>=patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
        #log for every 5th epoch
        if epoch%5==0:
            logger.info(
                f"Epoch {epoch}/{epochs} — "
                f"Train Loss: {loss.item():.4f}, "
                f"Val Loss: {val_loss.item():.4f}, "
                f"Val Acc: {val_accuracy:.2%}"      # ← add  (.2% shows as percentage)
            )

def compute_accuracy(predictions,labels,threshold=0.5):
    predicted_classes=(predictions>=threshold).float()
    correct=(predicted_classes==labels).sum().item()
    accuracy=correct/len(labels)
    return accuracy
if __name__ == "__main__":
    train()