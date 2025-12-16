"""
Deep Reinforcement Learning - Enhanced with Deep Q-Network (DQN)

Improvements:
- Neural network for Q-value approximation
- Experience replay for stable learning
- Target network for better convergence
- Epsilon-greedy exploration with decay
- Better visualization and metrics
"""

import numpy as np
import pylab as pl
import networkx as nx
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    """
    Deep Q-Network Agent with experience replay and target network
    """
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """Build neural network for Q-value approximation"""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu',
                  kernel_initializer='he_uniform'),
            Dropout(0.2),
            Dense(64, activation='relu', kernel_initializer='he_uniform'),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_initializer='he_uniform'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update target network weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, available_actions):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)
        
        state = np.array(state).reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)[0]
        
        # Only consider available actions
        valid_q_values = {action: q_values[action] for action in available_actions}
        return max(valid_q_values, key=valid_q_values.get)
    
    def replay(self, batch_size):
        """Train on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict Q-values for starting states
        q_values = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values with Bellman equation
        for i in range(batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        self.model.fit(states, q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

edges = [(0, 1), (1, 5), (5, 6), (5, 4), (1, 2), 
         (1, 3), (9, 10), (2, 4), (0, 6), (6, 7),
         (8, 9), (7, 8), (1, 7), (3, 9)]

goal = 10
MATRIX_SIZE = 11

# Build reward matrix
M = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
M *= -1

for point in edges:
    if point[1] == goal:
        M[point] = 100
    else:
        M[point] = 0
    
    if point[0] == goal:
        M[point[::-1]] = 100
    else:
        M[point[::-1]] = 0

M[goal, goal] = 100

# Traditional Q-learning for comparison
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))
gamma = 0.75
initial_state = 1

# Determines the available actions for a given state
def available_actions(state):
    current_state_row = M[state, ]
    available_action = np.where(current_state_row >= 0)[1]
    return available_action

available_action = available_actions(initial_state)

# Chooses one of the available actions at random
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_action, 1))
    return next_action


action = sample_next_action(available_action)

def update(current_state, action, gamma):

  max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1]
  if max_index.shape[0] > 1:
      max_index = int(np.random.choice(max_index, size = 1))
  else:
      max_index = int(max_index)
  max_value = Q[action, max_index]
  Q[current_state, action] = M[current_state, action] + gamma * max_value
  if (np.max(Q) > 0):
    return(np.sum(Q / np.max(Q)*100))
  else:
    return (0)
# Updates the Q-Matrix according to the path chosen

update(initial_state, action, gamma)

# Traditional Q-learning training
print("Training Traditional Q-Learning...")
scores = []
for i in range(1000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_action = available_actions(current_state)
    action = sample_next_action(available_action)
    score = update(current_state, action, gamma)
    scores.append(score)

# Testing traditional Q-learning
current_state = 0
steps_traditional = [current_state]

while current_state != 10:
    next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state, ]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    steps_traditional.append(next_step_index)
    current_state = next_step_index

print("Traditional Q-Learning - Most efficient path:", steps_traditional)

# Deep Q-Network training
print("\nTraining Deep Q-Network...")
state_size = MATRIX_SIZE
action_size = MATRIX_SIZE
agent = DQNAgent(state_size, action_size)

episodes = 500
batch_size = 32
scores_dqn = []
epsilon_history = []

for episode in range(episodes):
    state = np.random.randint(0, MATRIX_SIZE)
    state_vector = np.zeros(state_size)
    state_vector[state] = 1
    
    total_reward = 0
    steps = 0
    max_steps = 50
    
    while state != goal and steps < max_steps:
        # Get available actions
        available_action = available_actions(state)
        
        # Choose action
        action = agent.act(state_vector, available_action)
        
        # Get reward
        reward = M[state, action]
        
        # Next state
        next_state = action
        next_state_vector = np.zeros(state_size)
        next_state_vector[next_state] = 1
        
        # Check if done
        done = (next_state == goal)
        
        # Remember experience
        agent.remember(state_vector, action, reward, next_state_vector, done)
        
        # Update state
        state = next_state
        state_vector = next_state_vector
        total_reward += reward
        steps += 1
    
    # Train agent
    agent.replay(batch_size)
    
    # Update target network every 10 episodes
    if episode % 10 == 0:
        agent.update_target_model()
    
    scores_dqn.append(total_reward)
    epsilon_history.append(agent.epsilon)
    
    if episode % 50 == 0:
        avg_score = np.mean(scores_dqn[-50:]) if len(scores_dqn) >= 50 else np.mean(scores_dqn)
        print(f"Episode: {episode}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

# Testing DQN
agent.epsilon = 0  # Use greedy policy for testing
current_state = 0
steps_dqn = [current_state]
state_vector = np.zeros(state_size)
state_vector[current_state] = 1

for _ in range(20):  # Max 20 steps
    if current_state == goal:
        break
    available_action = available_actions(current_state)
    action = agent.act(state_vector, available_action)
    steps_dqn.append(action)
    current_state = action
    state_vector = np.zeros(state_size)
    state_vector[current_state] = 1

print("DQN - Most efficient path:", steps_dqn)

# Visualization
fig, axes = pl.subplots(2, 2, figsize=(15, 12))

# Plot 1: Graph structure
G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, ax=axes[0, 0])
nx.draw_networkx_edges(G, pos, ax=axes[0, 0])
nx.draw_networkx_labels(G, pos, ax=axes[0, 0])
axes[0, 0].set_title('Graph Structure (Goal: Node 10)', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Plot 2: Traditional Q-learning scores
axes[0, 1].plot(scores, linewidth=2, color='blue', alpha=0.7)
axes[0, 1].set_title('Traditional Q-Learning Progress', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Iteration', fontsize=10)
axes[0, 1].set_ylabel('Reward Gained', fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: DQN scores
axes[1, 0].plot(scores_dqn, linewidth=2, color='green', alpha=0.7)
axes[1, 0].set_title('Deep Q-Network Progress', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Episode', fontsize=10)
axes[1, 0].set_ylabel('Total Reward', fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Epsilon decay
axes[1, 1].plot(epsilon_history, linewidth=2, color='red', alpha=0.7)
axes[1, 1].set_title('Epsilon Decay (Exploration Rate)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Episode', fontsize=10)
axes[1, 1].set_ylabel('Epsilon', fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

pl.tight_layout()
pl.savefig('dqn_training_results.png', dpi=300, bbox_inches='tight')
print("\nTraining results saved to 'dqn_training_results.png'")
pl.show()

# Save DQN model
agent.model.save('dqn_pathfinding_model.h5')
print("DQN model saved to 'dqn_pathfinding_model.h5'")

# Additional experiment: Environment-aware DQN with obstacles
print("\n" + "="*60)
print("BONUS: Environment-Aware DQN (with obstacles)")
print("="*60)

police = [2, 4, 5]  # Penalty nodes
drug_traces = [3, 8, 9]  # Bonus nodes

# Modify reward matrix for environment-aware learning
M_env = M.copy()
for p in police:
    for i in range(MATRIX_SIZE):
        if M_env[i, p] >= 0 and M_env[i, p] < 100:
            M_env[i, p] = -50  # Penalty for police
            
for d in drug_traces:
    for i in range(MATRIX_SIZE):
        if M_env[i, d] >= 0 and M_env[i, d] < 100:
            M_env[i, d] = 50  # Bonus for drug traces

# Train environment-aware DQN
agent_env = DQNAgent(state_size, action_size, learning_rate=0.0005)
scores_env = []

print("Training Environment-Aware DQN...")
for episode in range(500):
    state = np.random.randint(0, MATRIX_SIZE)
    state_vector = np.zeros(state_size)
    state_vector[state] = 1
    
    total_reward = 0
    steps = 0
    max_steps = 50
    
    while state != goal and steps < max_steps:
        available_action = available_actions(state)
        action = agent_env.act(state_vector, available_action)
        reward = M_env[state, action]
        
        next_state = action
        next_state_vector = np.zeros(state_size)
        next_state_vector[next_state] = 1
        done = (next_state == goal)
        
        agent_env.remember(state_vector, action, reward, next_state_vector, done)
        
        state = next_state
        state_vector = next_state_vector
        total_reward += reward
        steps += 1
    
    agent_env.replay(batch_size)
    
    if episode % 10 == 0:
        agent_env.update_target_model()
    
    scores_env.append(total_reward)
    
    if episode % 100 == 0:
        avg_score = np.mean(scores_env[-100:]) if len(scores_env) >= 100 else np.mean(scores_env)
        print(f"Episode: {episode}, Avg Score: {avg_score:.2f}, Epsilon: {agent_env.epsilon:.3f}")

# Test environment-aware DQN
agent_env.epsilon = 0
current_state = 0
steps_env = [current_state]
state_vector = np.zeros(state_size)
state_vector[current_state] = 1

for _ in range(20):
    if current_state == goal:
        break
    available_action = available_actions(current_state)
    action = agent_env.act(state_vector, available_action)
    steps_env.append(action)
    current_state = action
    state_vector = np.zeros(state_size)
    state_vector[current_state] = 1

print("Environment-Aware DQN path:", steps_env)

# Check if path avoids police and collects drug traces
police_encountered = len([s for s in steps_env if s in police])
traces_collected = len([s for s in steps_env if s in drug_traces])
print(f"Police encountered: {police_encountered}")
print(f"Drug traces collected: {traces_collected}")

# Visualize labeled graph
G = nx.Graph()
G.add_edges_from(edges)
mapping = {0:'0-Start', 1:'1', 2:'2-Police', 3:'3-Trace',
           4:'4-Police', 5:'5-Police', 6:'6', 7:'7', 8:'8-Trace',
           9:'9-Trace', 10:'10-GOAL'}

H = nx.relabel_nodes(G, mapping)
pos = nx.spring_layout(H, seed=42)

# Color nodes based on type
node_colors = []
for node in range(MATRIX_SIZE):
    if node == 0:
        node_colors.append('lightgreen')
    elif node == goal:
        node_colors.append('gold')
    elif node in police:
        node_colors.append('red')
    elif node in drug_traces:
        node_colors.append('lightblue')
    else:
        node_colors.append('lightgray')

pl.figure(figsize=(10, 8))
nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=700)
nx.draw_networkx_edges(H, pos, width=2, alpha=0.5)
nx.draw_networkx_labels(H, pos, font_size=9, font_weight='bold')
pl.title('Environment Graph\n(Red=Police, Blue=Traces, Gold=Goal)', 
         fontsize=14, fontweight='bold')
pl.axis('off')
pl.tight_layout()
pl.savefig('environment_graph.png', dpi=300, bbox_inches='tight')
print("\nEnvironment graph saved to 'environment_graph.png'")
pl.show()

print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"Traditional Q-Learning path: {steps_traditional}")
print(f"Standard DQN path: {steps_dqn}")
print(f"Environment-Aware DQN path: {steps_env}")
print(f"\nPath lengths - Traditional: {len(steps_traditional)}, " 
      f"DQN: {len(steps_dqn)}, Env-DQN: {len(steps_env)}")
print("="*60)
