import gymnasium as gym
import pandas as pd
from configs.default_config import CONFIG
from agent.q_learning import QLearningAgent

def evaluate():
    env = gym.make("Taxi-v4")
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, 0, 0)
    agent.load('models/q_table_sarsa.npy')
    
    eval_data = []
    
    print("Iniciando Avaliação Determinística...")
    for episode in range(CONFIG["eval_episodes"]):
        state, _ = env.reset(seed=CONFIG["seed"] + 1000 + episode) # Sementes diferentes do treino
        total_reward, penalties, steps = 0, 0, 0
        done = False
        
        while not done and steps < CONFIG["max_steps_per_episode"]:
            action = agent.choose_action(state, explore=False) # Sem exploração!
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if reward == -10:
                penalties += 1
                
            state = next_state
            total_reward += reward
            steps += 1
            
        eval_data.append({"episode": episode, "reward": total_reward, "penalties": penalties, "steps": steps})
    
    df = pd.DataFrame(eval_data)
    print("\nResultados da Avaliação:")
    print(f"Recompensa Média: {df['reward'].mean():.2f}")
    print(f"Passos Médios por Episódio: {df['steps'].mean():.2f}")
    print(f"Total de Penalidades: {df['penalties'].sum()}")

if __name__ == "__main__":
    evaluate()