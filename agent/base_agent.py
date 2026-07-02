import numpy as np
import random

class BaseAgent:
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

    def update_epsilon(self):
        """Decai o epsilon gradualmente para focar em explotação."""
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def save(self, filepath):
        np.save(filepath, self.q_table)

    def load(self, filepath):
        self.q_table = np.load(filepath)

    def learn(self, state, action, reward, next_state, next_action, done):
        """Método abstrato que deve ser implementado pelas subclasses."""
        raise NotImplementedError("O método learn deve ser implementado pela subclasse específica.")