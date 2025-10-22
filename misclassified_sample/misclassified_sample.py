import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
from preprocessing import val_df
from train import lit_model,val_loader

def get_misclassified_samples(model, dataloader, df_raw, class_names=["non-sarcastic", "sarcastic"]):
    model.eval()
    preds, targets, ids = [], [], []

    with torch.no_grad():
        for i, b in enumerate(dataloader):
            ids = b["ids"].to(model.device)
            mask = b["mask"].to(model.device)
            emoji = b["emoji"].to(model.device)
            labels = b["label"].to(model.device)

            outputs = model(ids, mask, emoji)
            pred_labels = torch.argmax(outputs, dim=1)

            preds.extend(pred_labels.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            

    df_eval = df_raw.copy()
    df_eval["true"] = targets
    df_eval["pred"] = preds
    df_wrong = df_eval[df_eval["true"] != df_eval["pred"]]

    
    df_wrong["true_label"] = df_wrong["true"].map(lambda x: class_names[x])
    df_wrong["pred_label"] = df_wrong["pred"].map(lambda x: class_names[x])

    return df_wrong[["text", "true_label", "pred_label"]]


df_val = val_df.copy()
df_val["text"] = val_df["clean_text"]
wrong_df = get_misclassified_samples(lit_model, val_loader, val_df)
wrong_df.head(30)

