import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_learning_curve(df, window=100, output_path=None):
    """
    Plots the learning curve: total reward per episode and a moving average.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards with low alpha
    sns.lineplot(x='episode', y='total_reward', data=df, alpha=0.3, label='Raw Reward')
    
    # Calculate moving average
    df['moving_avg_reward'] = df['total_reward'].rolling(window=window).mean()
    
    # Plot moving average
    sns.lineplot(x='episode', y='moving_avg_reward', data=df, color='red', linewidth=2, label=f'Moving Average (w={window})')
    
    plt.title('Learning Curve: Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")
    
    plt.close()

def plot_generic(df, x, y, title, xlabel, ylabel, output_path=None, hue=None, moving_avg_window=None):
    """
    Generic plotting function for other analyses.
    """
    plt.figure(figsize=(12, 6))
    
    if moving_avg_window:
        sns.lineplot(x=x, y=y, data=df, alpha=0.3, label='Raw Data', hue=hue)
        df['moving_avg'] = df[y].rolling(window=moving_avg_window).mean()
        sns.lineplot(x=x, y='moving_avg', data=df, color='red', linewidth=2, label=f'Moving Average (w={moving_avg_window})')
    else:
        sns.lineplot(x=x, y=y, data=df, hue=hue)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    
def plot_comparison(results_dict, x, y, title, xlabel, ylabel, output_path, window=1000):
    """
    Plots multiple curves for comparison (e.g., different alphas or gammas).
    results_dict: { 'label': dataframe }
    """
    plt.figure(figsize=(12, 6))
    
    for label, df in results_dict.items():
        # Calculate moving average
        df_copy = df.copy()
        df_copy['moving_avg'] = df_copy[y].rolling(window=window).mean()
        sns.lineplot(x=x, y='moving_avg', data=df_copy, label=label, linewidth=2)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
def plot_dual_axis(df, x, y1, y2, title, xlabel, y1label, y2label, output_path, window=1000):
    """
    Plots two metrics with different Y-axes (e.g., Reward and Epsilon).
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # First axis (Reward)
    color1 = 'tab:blue'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label, color=color1)
    
    # Moving average for y1
    df_copy = df.copy()
    df_copy['y1_moving_avg'] = df_copy[y1].rolling(window=window).mean()
    sns.lineplot(x=x, y='y1_moving_avg', data=df_copy, ax=ax1, color=color1, label=y1label)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Second axis (Epsilon)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(y2label, color=color2)
    sns.lineplot(x=x, y=y2, data=df, ax=ax2, color=color2, label=y2label, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(title)
    fig.tight_layout()
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Dual-axis plot saved to: {output_path}")
        
    plt.close()
