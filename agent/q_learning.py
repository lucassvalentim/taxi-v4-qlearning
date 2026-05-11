import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha, gamma, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Inicializa a tabela Q com zeros
        self.q_table = np.zeros([state_size, action_size])

    def choose_action(self, state, explore=True):
        """Escolhe ação baseada na política epsilon-greedy."""
        if explore and random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """Atualiza a tabela Q usando a equação de Bellman adaptada."""
        old_value = self.q_table[state, action]
        next_max = 0 if done else np.max(self.q_table[next_state])
        
        # Equação do Q-Learning
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

    def update_epsilon(self):
        """Decai o epsilon gradualmente para focar em explotação ao longo do tempo."""
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def save(self, filepath):
        np.save(filepath, self.q_table)

    def load(self, filepath):
        self.q_table = np.load(filepath)