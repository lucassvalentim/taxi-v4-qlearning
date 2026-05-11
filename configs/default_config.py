import os

CONFIG = {
    # Hiperparâmetros do Q-Learning
    "alpha": 0.1,                # Taxa de aprendizado
    "gamma": 0.6,                # Fator de desconto
    "epsilon_start": 1.0,        # Exploração inicial (100% aleatório)
    "epsilon_min": 0.01,         # Exploração mínima
    "epsilon_decay": 0.001,      # Taxa de decaimento do epsilon por episódio
    
    # Configurações de Treinamento
    "train_episodes": 100000,    # Número total de episódios para treinamento   
    "max_steps_per_episode": 99, # Número máximo de passos por episódio (evitar loops infinitos) 
    "seed": 42,
    
    # Avaliação
    "eval_episodes": 100,
    
    # Caminhos
    "model_path": os.path.join("models", "q_table.npy"),
    "results_dir": "results"
}