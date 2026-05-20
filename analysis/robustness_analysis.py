import pandas as pd
import os
import sys

# Adiciona o diretório raiz ao path para importar train e utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import train
from utils.plotter import plot_robustness
from configs.default_config import CONFIG

def run_robustness_analysis():
    # Vamos rodar 5 sessões com sementes diferentes
    seeds = [42, 10, 2024, 999, 12345]
    results = []
    EPISODES = 40000 # 40k episódios é suficiente para ver a convergência e estabilidade
    
    print(f"--- Iniciando Análise de Robustez ({len(seeds)} runs) ---")
    
    for i, seed in enumerate(seeds):
        print(f"\nRun {i+1}/{len(seeds)} - Semente: {seed}")
        df = train({
            "seed": seed, 
            "train_episodes": EPISODES, 
            "model_path": None
        })
        results.append(df)
        
    print("\nCalculando métricas agregadas e gerando gráfico...")
    plot_robustness(
        results, 
        x='episode', 
        y='total_reward', 
        title=f'Model Robustness: Average Reward across {len(seeds)} different seeds', 
        xlabel='Episode', 
        ylabel='Average Reward (Moving Avg)', 
        output_path="output/model_robustness.png",
        window=1000
    )
    
    print("\nRobustness and Generalization analysis complete.")

if __name__ == "__main__":
    run_robustness_analysis()
