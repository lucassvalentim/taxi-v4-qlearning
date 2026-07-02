import pandas as pd
import os
import sys

# Adiciona o diretório raiz ao path para importar train e utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import train
from utils.plotter import plot_comparison_sensitivity
from configs.default_config import CONFIG

def run_sensitivity_analysis(agent_type="q_learning"):
    # Reduzimos o número de episódios para que o teste seja rápido, mas ainda assim perceptível
    # O Taxi-v4 costuma convergir bem em 20k-40k episódios.
    EPISODES = 30000
    agent_label = agent_type.upper()
    
    # Garante que a pasta de output exista
    os.makedirs("output", exist_ok=True)
    
    # 1. Análise de Alpha (Learning Rate)
    alphas = [0.01, 0.1, 0.5]
    alpha_results = {}
    
    print(f"--- Iniciando Sensibilidade de Alpha ({agent_label}) ---")
    for a in alphas:
        df = train(
            agent_type=agent_type,
            config_override={
                "alpha": a, 
                "train_episodes": EPISODES, 
                "model_path": None # Não queremos salvar os modelos intermediários
            }
        )
        alpha_results[f"Alpha={a}"] = df
        
    plot_comparison_sensitivity(
        alpha_results, 
        x='episode', 
        y='total_reward', 
        title=f'Sensitivity Analysis ({agent_label}): Learning Rate (Alpha)', 
        xlabel='Episode', 
        ylabel='Average Reward (Moving Avg)', 
        output_path=f"output/{agent_type}_sensitivity_alpha.png",
        window=1000
    )
    
    # 2. Análise de Gamma (Discount Factor)
    gammas = [0.1, 0.6, 0.99]
    gamma_results = {}
    
    print(f"\n--- Iniciando Sensibilidade de Gamma ({agent_label}) ---")
    for g in gammas:
        df = train(
            agent_type=agent_type,
            config_override={
                "gamma": g, 
                "train_episodes": EPISODES, 
                "model_path": None
            }
        )
        gamma_results[f"Gamma={g}"] = df
        
    plot_comparison_sensitivity(
        gamma_results, 
        x='episode', 
        y='total_reward', 
        title=f'Sensitivity Analysis ({agent_label}): Discount Factor (Gamma)', 
        xlabel='Episode', 
        ylabel='Average Reward (Moving Avg)', 
        output_path=f"output/{agent_type}_sensitivity_gamma.png",
        window=1000
    )
    
    print(f"\nSensitivity Analysis for {agent_label} complete.")

if __name__ == "__main__":
    # Executa a análise para o Q-Learning
    run_sensitivity_analysis(agent_type="q_learning")
    
    print("\n" + "="*50 + "\n")
    
    # Executa a análise para o SARSA
    run_sensitivity_analysis(agent_type="sarsa")