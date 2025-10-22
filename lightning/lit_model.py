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

        # move alpha to same device
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)

        # compute cross-entropy loss
        cross_entropy_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')

        # compute modulating factor
        prob_correct = torch.exp(-cross_entropy_loss)
        focal_loss = ((1 - prob_correct) ** self.gamma) * cross_entropy_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LitSarcasmModel(pl.LightningModule):
    def __init__(self, emoji_vocab_size, class_weights=None, lr=1e-5):
        super().__init__()
        self.learning_rate = lr
        self.save_hyperparameters()

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.clone().detach().to(self.device)) 
        else:
            self.register_buffer("class_weights", torch.tensor([1.0, 1.0], dtype=torch.float))

        self.loss_fn = FocalLoss(alpha=self.class_weights, gamma=2)
        self.model = MultimodalSarcasmModel(emoji_vocab_size=emoji_vocab_size)

    def training_step(self, b, idx):
        ids = b["ids"]
        mask = b["mask"]
        emoji = b["emoji"]
        labels = b["label"]

        #forward
        outputs = self(ids, mask, emoji)
        
        #compute preds
        preds = torch.argmax(outputs, dim=1)
        
        #check shapes
        assert preds.shape == labels.shape, "pred or label shape mismatch"

        #compute acc 
        acc = (preds == labels).float().mean().item()

        #compute loss
        loss = self.loss_fn(outputs, labels)

        if idx % 100 == 0: 
            print(f"[train] batch={idx} loss={loss.item():.4f} acc={acc:.3f}")

    # log metrics
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, b, idx):
        ids = b["ids"]
        mask = b["mask"]
        emoji = b["emoji"]
        labels = b["label"]

        outputs = self(ids, mask, emoji)

        preds = torch.argmax(outputs, dim=1)

        
        assert preds.shape == labels.shape, "pred or label shape mismatch"

        acc = (preds == labels).float().mean().item()

        loss = self.loss_fn(outputs, labels)

        if idx % 100 == 0:
            print(f"[val] step={idx} loss={loss.item():.4f} acc={acc:.3f}")


        self.log("val_loss", loss, prog_bar=True,on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True,  on_epoch=True, on_step=False)

        return loss



   
    def forward(self, ids, mask, emoji):
        return self.model(ids, mask, emoji)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.85)
        return [optimizer], [scheduler]
