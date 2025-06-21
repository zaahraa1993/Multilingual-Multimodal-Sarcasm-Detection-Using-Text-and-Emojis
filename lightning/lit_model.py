import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from model import MultimodalSarcasmModel 


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)

        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LitSarcasmModel(pl.LightningModule):
    def __init__(self, emoji_vocab_size, class_weights=None, lr=1e-5):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.clone().detach().to(self.device)) # Move class_weights to the correct device
        else:
            self.register_buffer("class_weights", torch.tensor([1.0, 1.0], dtype=torch.float))

        self.loss_fn = FocalLoss(alpha=self.class_weights, gamma=2)
        self.model = MultimodalSarcasmModel(emoji_vocab_size=emoji_vocab_size)


    def forward(self, input_ids, attention_mask, emoji_ids):
        return self.model(input_ids, attention_mask, emoji_ids)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"], batch["emoji_ids"])
        loss = self.loss_fn(logits, batch["label"])
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(batch["label"].cpu().long(), preds.cpu())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"], batch["emoji_ids"])
        loss = self.loss_fn(logits, batch["label"])
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(batch["label"].cpu().long(), preds.cpu())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
