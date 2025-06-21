import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train import lit_model,val_loader

def evaluate_model(model, dataloader, class_names=["non-sarcastic", "sarcastic"]):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            emoji_ids = batch["emoji_ids"].to(model.device)
            labels = batch["label"].to(model.device)

            
            logits = model.model(input_ids, attention_mask, emoji_ids)
            pred_labels = torch.argmax(logits, dim=1)

            preds.extend(pred_labels.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    # 1. Classification Report
    print("\n🧾 Classification Report:\n")
    report = classification_report(targets, preds, target_names=class_names)
    print(report)

    with open("/content/sample_data/classification_report_after.txt", "w") as f:
        f.write(report)

    # === 2. Confusion Matrix
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("/content/sample_data/confusion_matrix_after.png")
    plt.show()

    evaluate_model(lit_model, val_loader)