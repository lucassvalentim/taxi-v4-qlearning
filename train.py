import gymnasium as gym
import numpy as np
import pandas as pd
import os
import random
from configs.default_config import CONFIG
from agent.q_learning import QLearningAgent
from agent.sarsa import SARSAAgent

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def train(agent_type="q_learning", config_override=None):
    current_config = CONFIG.copy()
    if config_override:
        current_config.update(config_override)

    set_seed(current_config["seed"])
    env = gym.make("Taxi-v4")
    
    agent_kwargs = {
        "state_size": env.observation_space.n,
        "action_size": env.action_space.n,
        "alpha": current_config["alpha"],
        "gamma": current_config["gamma"],
        "epsilon": current_config["epsilon_start"],
        "epsilon_min": current_config["epsilon_min"],
        "epsilon_decay": current_config["epsilon_decay"]
    }
    
    if agent_type.lower() == "q_learning":
        agent = QLearningAgent(**agent_kwargs)
    elif agent_type.lower() == "sarsa":
        agent = SARSAAgent(**agent_kwargs)
    else:
        raise ValueError(f"Tipo de agente inválido: {agent_type}. Escolha 'q_learning' ou 'sarsa'.")
    
    training_data = []
    
    print(f"Iniciando Treinamento com {agent_type.upper()} (Alpha={current_config['alpha']}, Gamma={current_config['gamma']})...")
    for episode in range(current_config["train_episodes"]):
        state, _ = env.reset(seed=current_config["seed"] + episode)
        total_reward = 0
        penalties = 0
        done = False
        steps = 0
        
        action = agent.choose_action(state, explore=True)
        
        while not done and steps < current_config["max_steps_per_episode"]:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Escolhe a próxima ação baseado no estado em que acabou de chegar
            next_action = agent.choose_action(next_state, explore=True)
            
            # Ambos os métodos recebem os mesmos parâmetros agora
            agent.learn(state, action, reward, next_state, next_action, done)
            
            if reward == -10:
                penalties += 1
                
            state = next_state
            action = next_action  # A próxima ação vira a ação atual para a próxima iteração
            total_reward += reward
            steps += 1
            
        agent.update_epsilon()
        
        training_data.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "penalties": penalties,
            "epsilon": agent.epsilon
        })
        
        if (episode + 1) % 10000 == 0:
            print(f"Episódio: {episode + 1} | Recompensa Média (últimos 100): {np.mean([d['total_reward'] for d in training_data[-100:]]):.2f}")

    if current_config.get("model_path"):
        base_path, ext = os.path.splitext(current_config["model_path"])
        actual_path = f"{base_path}_{agent_type}{ext}"
        os.makedirs(os.path.dirname(actual_path), exist_ok=True)
        agent.save(actual_path)
    
    env.close()
    return pd.DataFrame(training_data)

if __name__ == "__main__":
    df_sarsa = train(agent_type="sarsa")
    df_sarsa.to_csv("results/sarsa_training.csv", index=False)