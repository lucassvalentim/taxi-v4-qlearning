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

def train(config_override=None):
    # Merge default config with overrides
    current_config = CONFIG.copy()
    if config_override:
        current_config.update(config_override)

    set_seed(current_config["seed"])
    env = gym.make("Taxi-v4")
    
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        alpha=current_config["alpha"],
        gamma=current_config["gamma"],
        epsilon=current_config["epsilon_start"],
        epsilon_min=current_config["epsilon_min"],
        epsilon_decay=current_config["epsilon_decay"]
    )
    
    training_data = []
    
    print(f"Iniciando Treinamento (Alpha={current_config['alpha']}, Gamma={current_config['gamma']})...")
    for episode in range(current_config["train_episodes"]):
        state, _ = env.reset(seed=current_config["seed"] + episode)
        total_reward = 0
        penalties = 0
        done = False
        steps = 0
        
        while not done and steps < current_config["max_steps_per_episode"]:
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
        
        training_data.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "penalties": penalties,
            "epsilon": agent.epsilon
        })
        
        if (episode + 1) % 10000 == 0:
            print(f"Episódio: {episode + 1} | Recompensa Média (últimos 100): {np.mean([d['total_reward'] for d in training_data[-100:]]):.2f}")

    # Persistência (opcional, pode ser suprimida se estivermos apenas comparando)
    if current_config.get("model_path"):
        os.makedirs(os.path.dirname(current_config["model_path"]), exist_ok=True)
        agent.save(current_config["model_path"])
    
    env.close()
    return pd.DataFrame(training_data)

if __name__ == "__main__":
    train()