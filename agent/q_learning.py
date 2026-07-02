import numpy as np
from agent.base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def learn(self, state, action, reward, next_state, next_action, done):
        """Atualiza a tabela Q usando a equação do Q-Learning"""
        old_value = self.q_table[state, action]
        
        next_max = 0 if done else np.max(self.q_table[next_state])
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value