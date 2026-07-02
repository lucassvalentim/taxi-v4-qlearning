from agent.base_agent import BaseAgent

class SARSAAgent(BaseAgent):
    def learn(self, state, action, reward, next_state, next_action, done):
        """Atualiza a tabela Q usando a equação do SARSA (On-Policy)."""
        old_value = self.q_table[state, action]
        
        next_q = 0 if done else self.q_table[next_state, next_action]
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_q)
        self.q_table[state, action] = new_value