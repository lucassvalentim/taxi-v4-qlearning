import pandas as pd
import os
import sys

# Adiciona o diretório raiz ao path para importar train e utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import train
from utils.plotter import plot_comparison
from configs.default_config import CONFIG

def run_factorial_analysis(agent_type="q_learning"):
    # Mantendo 30k episódios para ver a convergência
    EPISODES = 20000
    agent_label = agent_type.upper()
    
    os.makedirs("output", exist_ok=True)
    
    # Grade de hiperparâmetros (2x2x2 = 8 combinações)
    # Dica: Se quiser testar mais, basta adicionar elementos nas listas
    alphas = [0.1, 0.5]
    gammas = [0.6, 0.99]
    epsilons = [0.5, 1.0]
    
    factorial_results = {}
    
    print(f"=== Iniciando Análise Fatorial para {agent_label} ===")
    
    # Loops aninhados para cobrir todas as combinações (fatorial completo)
    for a in alphas:
        for g in gammas:
            for e in epsilons:
                # Criando um rótulo claro para a legenda do gráfico
                combo_label = f"α={a} | γ={g} | ε={e}"
                
                print(f"-> Treinando combinação: {combo_label}...")
                
                df = train(
                    agent_type=agent_type,
                    config_override={
                        "alpha": a, 
                        "gamma": g,
                        "epsilon_start": e,
                        "train_episodes": EPISODES, 
                        "model_path": None # Evita entupir o disco com arquivos .npy
                    }
                )
                # Salva o dataframe indexado pelo nome da combinação
                factorial_results[combo_label] = df
                
    print(f"\nGerando gráfico comparativo final para {agent_label}...")
    
    # Como a função plot_comparison aceita um dicionário de DataFrames,
    # ela vai plotar automaticamente as 8 curvas no mesmo gráfico!
    plot_comparison(
        factorial_results, 
        x='episode', 
        y='total_reward', 
        title=f'Factorial Analysis ({agent_label}): Hyperparameter Interaction', 
        xlabel='Episode', 
        ylabel='Average Reward (Moving Avg)', 
        output_path=f"output/{agent_type}_factorial_analysis.png",
        window=1000
    )
    
    print(f"Análise Fatorial de {agent_label} concluída com sucesso.\n")

if __name__ == "__main__":
    # Executa para o Q-Learning
    # run_factorial_analysis(agent_type="q_learning")
    
    # print("\n" + "="*60 + "\n")
    
    # Executa para o SARSA
    run_factorial_analysis(agent_type="sarsa")