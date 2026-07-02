import pandas as pd
import os
import sys

# Adiciona o diretório raiz ao path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.plotter import plot_generic

def run_analysis():
    # Caminhos esperados para os arquivos de métricas de ambos os agentes
    q_learning_path = "results/q_learning_training.csv"
    sarsa_path = "results/sarsa_training.csv"
    
    # Caminhos de saída para os gráficos comparativos
    output_steps = "output/comparison_steps_per_episode.png"
    output_penalties = "output/comparison_penalties_per_episode.png"
    
    # Dicionário que vai acumular os dados existentes
    data_to_compare = {}
    
    # Tenta carregar os dados do Q-Learning
    if os.path.exists(q_learning_path):
        print(f"Carregando métricas de: {q_learning_path}")
        data_to_compare["Q-Learning"] = pd.read_csv(q_learning_path)
    else:
        print(f"Aviso: {q_learning_path} não foi encontrado.")
        
    # Tenta carregar os dados do SARSA
    if os.path.exists(sarsa_path):
        print(f"Carregando métricas de: {sarsa_path}")
        data_to_compare["SARSA"] = pd.read_csv(sarsa_path)
    else:
        print(f"Aviso: {sarsa_path} não foi encontrado.")
        
    # Se nenhum arquivo for encontrado, interrompe a execução
    if not data_to_compare:
        print("Erro: Nenhum arquivo de métricas encontrado em results/. Certifique-se de salvar os CSVs após o treino.")
        return
    
    # 1. Análise Comparativa de Passos por Episódio (Eficiência)
    print("\nGerando gráfico comparativo de passos por episódio...")
    plot_generic(
        df_or_dict=data_to_compare, 
        x='episode', 
        y='steps', 
        title='Efficiency Comparison: Steps per Episode (Q-Learning vs SARSA)', 
        xlabel='Episode', 
        ylabel='Steps', 
        output_path=output_steps,
        moving_avg_window=1000
    )
    
    # 2. Análise Comparativa de Penalidades (Aderência às regras)
    print("Gerando gráfico comparativo de penalidades por episódio...")
    plot_generic(
        df_or_dict=data_to_compare, 
        x='episode', 
        y='penalties', 
        title='Rule Adherence Comparison: Penalties per Episode (Q-Learning vs SARSA)', 
        xlabel='Episode', 
        ylabel='Penalties', 
        output_path=output_penalties,
        moving_avg_window=1000
    )
    
    print("\nAnálise concluída com sucesso.")

if __name__ == "__main__":
    run_analysis()