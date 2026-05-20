import pandas as pd
import os
import sys

# Adiciona o diretório raiz ao path para importar train e utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import train
from utils.plotter import plot_comparison
from configs.default_config import CONFIG

def run_sensitivity_analysis():
    # Reduzimos o número de episódios para que o teste seja rápido, mas ainda assim perceptível
    # O Taxi-v4 costuma convergir bem em 20k-40k episódios.
    EPISODES = 30000
    
    # 1. Análise de Alpha (Learning Rate)
    alphas = [0.01, 0.1, 0.5]
    alpha_results = {}
    
    print("--- Iniciando Sensibilidade de Alpha ---")
    for a in alphas:
        df = train({
            "alpha": a, 
            "train_episodes": EPISODES, 
            "model_path": None # Não queremos salvar os modelos intermediários
        })
        alpha_results[f"Alpha={a}"] = df
        
    plot_comparison(
        alpha_results, 
        x='episode', 
        y='total_reward', 
        title='Sensitivity Analysis: Learning Rate (Alpha)', 
        xlabel='Episode', 
        ylabel='Average Reward (Moving Avg)', 
        output_path="output/sensitivity_alpha.png",
        window=1000
    )
    
    # 2. Análise de Gamma (Discount Factor)
    gammas = [0.1, 0.6, 0.99]
    gamma_results = {}
    
    print("\n--- Iniciando Sensibilidade de Gamma ---")
    for g in gammas:
        df = train({
            "gamma": g, 
            "train_episodes": EPISODES, 
            "model_path": None
        })
        gamma_results[f"Gamma={g}"] = df
        
    plot_comparison(
        gamma_results, 
        x='episode', 
        y='total_reward', 
        title='Sensitivity Analysis: Discount Factor (Gamma)', 
        xlabel='Episode', 
        ylabel='Average Reward (Moving Avg)', 
        output_path="output/sensitivity_gamma.png",
        window=1000
    )
    
    print("\nSensitivity Analysis complete.")

if __name__ == "__main__":
    run_sensitivity_analysis()
