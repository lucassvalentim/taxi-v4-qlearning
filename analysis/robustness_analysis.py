import pandas as pd
import os
import sys

# Adiciona o diretório raiz ao path para importar train e utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import train
from utils.plotter import plot_robustness

def run_robustness_analysis():
    # 5 sementes diferentes para garantir validade estatística
    seeds = [42, 10, 2024, 999, 12345]
    EPISODES = 40000 
    output_image = "output/comparison_model_robustness.png"
    
    # Definição dos agentes que serão testados
    agents_to_test = {
        "Q-Learning": "q_learning",
        "SARSA": "sarsa"
    }
    
    # Dicionário estruturado para armazenar os resultados de ambos
    all_results = {}
    
    print(f"--- Iniciando Análise Comparativa de Robustez ({len(seeds)} sementes por agente) ---")
    
    for label, agent_type in agents_to_test.items():
        print(f"\n================ Treinando {label} ================")
        agent_runs = []
        
        for i, seed in enumerate(seeds):
            print(f"Run {i+1}/{len(seeds)} | Semente Atual: {seed}")
            df = train(
                agent_type=agent_type,
                config_override={
                    "seed": seed, 
                    "train_episodes": EPISODES, 
                    "model_path": None # Evita salvar os arquivos de pesos (.npy)
                }
            )
            agent_runs.append(df)
            
        # Guarda a lista com as 5 rodadas do agente correspondente
        all_results[label] = agent_runs
        
    print("\nCalculando métricas agregadas e gerando gráfico de bandas de confiança...")
    plot_robustness(
        data_or_dict=all_results, 
        x='episode', 
        y='total_reward', 
        title=f'Model Robustness: Evaluation across {len(seeds)} Different Seeds', 
        xlabel='Episode', 
        ylabel='Average Reward (Moving Avg)', 
        output_path=output_image,
        window=1000
    )
    
    print("\nRobustness and Generalization analysis complete.")

if __name__ == "__main__":
    run_robustness_analysis()