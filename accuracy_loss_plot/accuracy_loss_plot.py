import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/logs/sarcasm/version_7/metrics.csv")
df = df[df["epoch"].notna()]
df["epoch"] = df["epoch"].astype(int)

plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.legend(); plt.show()

plt.plot(df["epoch"], df["train_acc_epoch"], label="Train Acc")
plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
plt.legend(); plt.show()
