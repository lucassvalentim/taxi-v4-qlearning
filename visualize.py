import gymnasium as gym
import time
from configs.default_config import CONFIG
from agent.q_learning import QLearningAgent

def visualize(episodes=3, delay=0.3):
    """
    Executa episódios no ambiente Taxi-v4 renderizando as ações graficamente.
    
    Args:
        episodes (int): Quantidade de episódios para visualizar.
        delay (float): Tempo de pausa (em segundos) entre cada passo para visualização humana.
    """
    env = gym.make("Taxi-v4", render_mode="human")
    
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, alpha=0, gamma=0)
    
    try:
        agent.load(CONFIG["model_path"])
        print(f"Tabela Q carregada com sucesso de: {CONFIG['model_path']}")
    except FileNotFoundError:
        print(f"Erro: Modelo não encontrado em {CONFIG['model_path']}.")
        print("Por favor, execute 'python train.py' primeiro para gerar o modelo.")
        env.close()
        return

    print(f"\nIniciando visualização gráfica de {episodes} episódios...")
    
    for episode in range(episodes):
        state, _ = env.reset(seed=CONFIG["seed"] + 2200 + episode)
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episódio {episode + 1} ---")
        
        while not done and steps < CONFIG["max_steps_per_episode"]:
            env.render()
            
            time.sleep(delay)
            
            action = agent.choose_action(state, explore=False)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            steps += 1
            
        env.render()
        time.sleep(1)
        
        print(f"Episódio {episode + 1} concluído | Passos: {steps} | Recompensa Total: {total_reward}")

    env.close()

if __name__ == "__main__":
    visualize(episodes=4, delay=0.3)