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
        for b in dataloader:
            ids = b["ids"].to(model.device)
            mask = b["mask"].to(model.device)
            emoji= b["emoji"].to(model.device)
            labels = b["label"].to(model.device)

            
            outputs = model.model(ids, mask, emoji)
            pred_labels = torch.argmax(outputs, dim=1)

            preds.extend(pred_labels.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    # Classification Report
    report = classification_report(targets, preds, target_names=class_names)
    print(report)

    with open("/content/sample_data/classification_report_after.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("/content/sample_data/confusion_matrix_after.png")
    plt.show()

    evaluate_model(lit_model, val_loader)