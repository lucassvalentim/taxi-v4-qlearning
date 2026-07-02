import pandas as pd
import os
import sys

# Adiciona o diretório raiz ao path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.plotter import plot_learning_curve

def run_analysis():
    # Caminhos para as métricas individuais de treino
    q_learning_path = "results/q_learning_training.csv"
    sarsa_path = "results/sarsa_training.csv"
    
    # Caminho de saída da imagem comparativa
    output_image = "output/comparison_learning_curve.png"
    
    data_to_compare = {}
    
    # Carrega dados do Q-Learning se existirem
    if os.path.exists(q_learning_path):
        print(f"Carregando métricas de: {q_learning_path}")
        data_to_compare["Q-Learning"] = pd.read_csv(q_learning_path)
    else:
        print(f"Aviso: {q_learning_path} não encontrado.")
        
    # Carrega dados do SARSA se existirem
    if os.path.exists(sarsa_path):
        print(f"Carregando métricas de: {sarsa_path}")
        data_to_compare["SARSA"] = pd.read_csv(sarsa_path)
    else:
        print(f"Aviso: {sarsa_path} não encontrado.")
        
    # Interrompe se não houver dados
    if not data_to_compare:
        print("Erro: Nenhum arquivo de métricas encontrado em results/. Execute o treinamento primeiro.")
        return
    
    print("\nGenerating comparative learning curve plot...")
    # Usando janela 1000 para suavizar o ruído característico do Taxi-v4
    plot_learning_curve(data_to_compare, window=1000, output_path=output_image)
    
    print("Analysis complete.")

if __name__ == "__main__":
    run_analysis()