from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import torch
from pytorch_lightning.loggers import CSVLogger
import numpy as np
import random
from preprocessing import train_df,val_df,emoji2id,tokenizer
from model import MultimodalSarcasmModel 
import pytorch_lightning as pl
from lit_model import LitSarcasmModel
from preprocessing import SarcasmDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)

set_seed(42)


# Compute class weights (balanced)
cw = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=train_df['label'])
class_weights = torch.tensor(cw, dtype=torch.float)


# dataloader
train_data = SarcasmDataset(train_df, tokenizer, emoji2id)
val_data = SarcasmDataset(val_df, tokenizer, emoji2id)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,num_workers=2,worker_init_fn=seed_worker,generator=g)
val_loader = DataLoader(val_data, batch_size=32,num_workers=2,worker_init_fn=seed_worker,generator=g)

# Lightning model
lit_model = LitSarcasmModel(emoji_vocab_size=len(emoji2id), class_weights=class_weights)
logger = CSVLogger("logs", name="sarcasm")

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    log_every_n_steps=10,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
    ],
    logger=CSVLogger("logs", name="sarcasm")
)


trainer.fit(lit_model, train_loader, val_loader)

