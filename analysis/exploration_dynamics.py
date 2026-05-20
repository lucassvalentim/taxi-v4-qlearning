import pandas as pd
import os
import sys

# Adiciona o diretório raiz ao path para importar train e utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import train
from utils.plotter import plot_dual_axis, plot_comparison
from configs.default_config import CONFIG

def run_exploration_analysis():
    # Caminho para as métricas originais
    base_metrics_path = "results/training_metrics.csv"
    
    # 1. Correlação Recompensa vs Epsilon (Base)
    if os.path.exists(base_metrics_path):
        print("--- Analisando Correlação Recompensa vs Epsilon (Base) ---")
        df_base = pd.read_csv(base_metrics_path)
        plot_dual_axis(
            df_base, 
            x='episode', 
            y1='total_reward', 
            y2='epsilon', 
            title='Exploration Dynamics: Reward vs Epsilon', 
            xlabel='Episode', 
            y1label='Average Reward', 
            y2label='Epsilon (Exploration Rate)', 
            output_path="output/reward_vs_epsilon.png",
            window=1000
        )
    
    # 2. Comparação de diferentes taxas de decaimento (Decay Rates)
    # Vamos testar um decaimento rápido, o padrão e um lento.
    decays = [0.005, 0.001, 0.0001] # Rápido, Padrão, Lento
    decay_results = {}
    EPISODES = 40000
    
    print("\n--- Comparando diferentes taxas de Epsilon Decay ---")
    for d in decays:
        print(f"Treinando com Epsilon Decay = {d}...")
        df = train({
            "epsilon_decay": d, 
            "train_episodes": EPISODES, 
            "model_path": None
        })
        decay_results[f"Decay={d}"] = df
        
    plot_comparison(
        decay_results, 
        x='episode', 
        y='total_reward', 
        title='Impact of Epsilon Decay Rate on Learning', 
        xlabel='Episode', 
        ylabel='Average Reward (Moving Avg)', 
        output_path="output/sensitivity_epsilon_decay.png",
        window=1000
    )
    
    print("\nExploration Dynamics analysis complete.")

if __name__ == "__main__":
    run_exploration_analysis()
