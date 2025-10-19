import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def smooth_ema(values, weight=0.9):
    smoothed = []
    last = values[0]
    for val in values:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_error(all_dfs, model_name):
    plt.figure(figsize=(10, 6))
    for df, label, color in all_dfs:
        df['train_acc_smooth'] = smooth_ema(df['train_acc'], weight=0.3)
        df['test_acc_smooth']  = smooth_ema(df['test_acc'], weight=0.3)
        plt.plot(df['x'], 100 - df['test_acc_smooth'] * 100, label=label, color=color, linewidth=2)
        plt.plot(df['x'], 100 - df['train_acc_smooth'] * 100, linestyle='--', label=f"{label} (train)", color=color)
    plt.xlim(0)
    plt.ylim(0, 100)
    xlabel = "Epoch" if model_name.lower() == "cnn" else "Iteration"
    plt.xlabel(xlabel)

    plt.ylabel("Classification Error (%)")
    model_name = model_name.title()
    plt.title(f"{model_name} Training and Validation Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{model_name}_error.png", dpi=300)
    plt.show()

def plot_loss(all_dfs, model_name):
    plt.figure(figsize=(10, 6))
    for df, label, color in all_dfs:
        df['train_loss_smooth'] = smooth_ema(df['train_loss'], weight=0.9)
        df['test_loss_smooth']  = smooth_ema(df['test_loss'], weight=0.9)
        plt.plot(df['x'], df['test_loss_smooth'], label=label, color=color, linewidth=2.5)
        plt.plot(df['x'], df['train_loss_smooth'], linestyle='--', label=f"{label} (train)", color=color)
    plt.xlim(0)
    plt.ylim(0, 3)
    xlabel = "Epoch" if model_name.lower() == "cnn" else "Iteration"
    plt.xlabel(xlabel)
    plt.ylabel("Classification Loss")
    model_name = model_name.title()
    plt.title(f"{model_name} Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_loss.png", dpi=300)
    plt.show()

def plot_acc(all_dfs, model_name):
    plt.figure(figsize=(10, 6))
    for df, label, color in all_dfs:
        df['train_acc_smooth'] = smooth_ema(df['train_acc'], weight=0.75)
        df['test_acc_smooth']  = smooth_ema(df['test_acc'], weight=0.75)
        plt.plot(df['x'], df['test_acc_smooth'] * 100, label=label, color=color, linewidth=2.5)
        plt.plot(df['x'], df['train_acc_smooth'] * 100, linestyle='--', label=f"{label} (train)", color=color)
    plt.xlim(0)
    plt.ylim(0, 100)
    xlabel = "Epoch" if model_name.lower() == "cnn" else "Iteration"
    plt.xlabel(xlabel)
    plt.ylabel("Classification Accuracy (%)")
    model_name = model_name.title()
    plt.title(f"{model_name} Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_acc.png", dpi=300)
    plt.show()

def plot_all(model_name: str):
    model_name = model_name.lower()
    all_dfs = []

    if model_name == "cnn":
        log_path = f"logs/{model_name}_log.csv"
        if os.path.exists(log_path):
            df = pd.read_csv(log_path).dropna()
            df.rename(columns={"epoch": "x"}, inplace=True)  # 👈 epoch → x
            all_dfs.append((df, "CNN", "blue"))
        else:
            print(f"[!] Log file not found: {log_path}")
    else:
        depths = [20, 56, 110]
        colors = ["red", "cyan", "gold"]
        for depth, color in zip(depths, colors):
            log_path = f"logs/{model_name}{depth}_log.csv"
            if os.path.exists(log_path):
                df = pd.read_csv(log_path).dropna()
                df.rename(columns={"iteration": "x"}, inplace=True)  # 👈 iteration → x
                all_dfs.append((df, f"{model_name.capitalize()}-{depth}", color))
            else:
                print(f"[!] Log file not found: {log_path}")


    if all_dfs:
        plot_error(all_dfs, model_name)
        plot_loss(all_dfs, model_name)
        plot_acc(all_dfs, model_name)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_logs.py [model_name]")
    else:
        plot_all(sys.argv[1])
