import time
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# Setup environment
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# --- FIXED HYPERPARAMETERS ---
alpha = 0.1         # Learning rate
gamma = 0.99        # Discount factor (High importance on future reward)
episodes = 2000     # Train for 2000 games

# Exploration parameters (Epsilon Decay)
epsilon = 1.0                 # Start: 100% Random Exploration
max_epsilon = 1.0             # Max Exploration
min_epsilon = 0.01            # Min Exploration
decay_rate = 0.005            # How fast to reduce exploration

print("❄️ Agent is training...")

history = []
# --- TRAINING PHASE ---
for episode in range(episodes):
    state, _ = env.reset()
    terminated = False
    truncated = False
    reward_per_episode = 0
    while not (terminated or truncated): # Fix: Check for truncation (time limit)
        
        # 1. Choose Action (Epsilon-Greedy with Decay)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore
        else:
            action = np.argmax(q_table[state]) # Exploit

        # 2. Take Action
        next_state, reward, terminated, truncated, _ = env.step(action)

        reward_per_episode += reward

        # add a penality if more steps are taken
        if not terminated:
            reward = -0.01
        elif reward == 0:
            reward = -1 # falls into a hole

        # 3. Update Q-Table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        # Standard Bellman Equation
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        
        state = next_state
    history.append(reward_per_episode)   
    # Decrease epsilon after each episode
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

print("Training completed!")

# 1. Calculate the Rolling Average
window = 100
# This creates an average of the last 100 episodes for every point
rolling_avg = np.convolve(history, np.ones(window)/window, mode='valid')

# 2. Create the Plot
plt.figure(figsize=(10, 5))

# Plot raw rewards with transparency (alpha=0.2)
plt.plot(history, alpha=0.2, label="Raw Rewards", color='blue')

# Plot the rolling average (starts after the first 99 episodes)
plt.plot(np.arange(window - 1, len(history)), rolling_avg, 
         label=f"{window}-Episode Average", color='red', linewidth=2)

plt.xlabel("Episodes 🎮")
plt.ylabel("Reward 🏆")
plt.title("Agent Learning Progress")
plt.legend()
plt.show()

print(f"Final Q-Table Values at Start (State 0): {q_table[0]}") 
# If these are all 0, the agent failed. If you see numbers, it learned!

env.close()

# --- TESTING PHASE ---
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

state, _ = env.reset()
terminated = False
truncated = False

print("❄️ Watch your agent go! ❄️")

while not (terminated or truncated):
    # Always exploit the best action now
    action = np.argmax(q_table[state])
    
    state, reward, terminated, truncated, _ = env.step(action)
    time.sleep(0.5)

print("Game Over!")
env.close()
