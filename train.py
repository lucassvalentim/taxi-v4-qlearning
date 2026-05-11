import gymnasium as gym
import numpy as np
import pandas as pd
import os
import random
from configs.default_config import CONFIG
from agent.q_learning import QLearningAgent

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def train():
    set_seed(CONFIG["seed"])
    env = gym.make("Taxi-v4")
    
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        alpha=CONFIG["alpha"],
        gamma=CONFIG["gamma"],
        epsilon=CONFIG["epsilon_start"],
        epsilon_min=CONFIG["epsilon_min"],
        epsilon_decay=CONFIG["epsilon_decay"]
    )
    
    training_data = [] # Para armazenar o espectro completo do treino
    
    print("Iniciando Treinamento...")
    for episode in range(CONFIG["train_episodes"]):
        state, _ = env.reset(seed=CONFIG["seed"] + episode)
        total_reward = 0
        penalties = 0
        done = False
        steps = 0
        
        while not done and steps < CONFIG["max_steps_per_episode"]:
            action = agent.choose_action(state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.learn(state, action, reward, next_state, done)
            
            if reward == -10:
                penalties += 1
                
            state = next_state
            total_reward += reward
            steps += 1
            
        agent.update_epsilon()
        
        # Coleta os dados do episódio
        training_data.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "penalties": penalties,
            "epsilon": agent.epsilon
        })
        
        if (episode + 1) % 5000 == 0:
            print(f"Episódio: {episode + 1} | Epsilon: {agent.epsilon:.4f} | Recompensa Média (últimos 100): {np.mean([d['total_reward'] for d in training_data[-100:]]):.2f}")

    # Persistência
    os.makedirs(os.path.dirname(CONFIG["model_path"]), exist_ok=True)
    agent.save(CONFIG["model_path"])
    
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    df_metrics = pd.DataFrame(training_data)
    df_metrics.to_csv(os.path.join(CONFIG["results_dir"], "training_metrics.csv"), index=False)
    
    print("Treinamento concluído. Tabela Q e métricas salvas.")
    env.close()

if __name__ == "__main__":
    train()