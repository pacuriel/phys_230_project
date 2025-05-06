import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(log_csv="loss_log.csv", out_png="loss_curve.png"):
    df = pd.read_csv(log_csv)
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_loss'], marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"✅ Saved loss plot to {out_png}")

if __name__ == "__main__":
    plot_loss()