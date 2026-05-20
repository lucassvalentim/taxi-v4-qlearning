import pandas as pd
import os
import sys

# Adiciona o diretório raiz ao path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.plotter import plot_generic

def run_analysis():
    # Caminho para as métricas de treino
    metrics_path = "results/training_metrics.csv"
    output_steps = "output/steps_per_episode.png"
    output_penalties = "output/penalties_per_episode.png"
    
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found. Please run train.py first.")
        return
    
    print(f"Loading metrics from {metrics_path}...")
    df = pd.read_csv(metrics_path)
    
    # 1. Análise de Passos por Episódio
    print("Generating steps per episode plot...")
    plot_generic(
        df, 
        x='episode', 
        y='steps', 
        title='Efficiency: Steps per Episode', 
        xlabel='Episode', 
        ylabel='Steps', 
        output_path=output_steps,
        moving_avg_window=1000
    )
    
    # 2. Análise de Penalidades
    print("Generating penalties per episode plot...")
    plot_generic(
        df, 
        x='episode', 
        y='penalties', 
        title='Rule Adherence: Penalties per Episode', 
        xlabel='Episode', 
        ylabel='Penalties', 
        output_path=output_penalties,
        moving_avg_window=1000
    )
    
    print("Analysis complete.")

if __name__ == "__main__":
    run_analysis()
