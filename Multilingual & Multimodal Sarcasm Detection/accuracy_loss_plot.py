import pandas as pd
import matplotlib.pyplot as plt


log_path = "/content/logs/sarcasm/version_7/metrics.csv"  


df = pd.read_csv(log_path)

df = df[df["epoch"].notna()]
df["epoch"] = df["epoch"].astype(int)

# Loss Plot
plt.figure(figsize=(10, 4))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Accuracy Plot
plt.figure(figsize=(10, 4))
plt.plot(df["epoch"], df["train_acc_epoch"], label="Train Acc", marker='o')
plt.plot(df["epoch"], df["val_acc"], label="Val Acc", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

