import pandas as pd
import os
import sys

# Adiciona o diretório raiz ao path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.plotter import plot_learning_curve

def run_analysis():
    # Caminho para as métricas de treino
    metrics_path = "results/training_metrics.csv"
    output_image = "output/learning_curve.png"
    
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found. Please run train.py first.")
        return
    
    print(f"Loading metrics from {metrics_path}...")
    df = pd.read_csv(metrics_path)
    
    print("Generating learning curve plot...")
    plot_learning_curve(df, window=1000, output_path=output_image)
    
    print("Analysis complete.")

if __name__ == "__main__":
    run_analysis()
