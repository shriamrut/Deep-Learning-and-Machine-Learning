import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

class DQN(nn.Module):
	def __init__(self, state_size, action_size):
		super(DQN, self).__init__()
		self.l1 = nn.Linear(state_size, 64)
		self.l2 = nn.Linear(64, 64)
		self.l3 = nn.Linear(64, action_size)

	def forward(self, x):
		x = torch.relu(self.l1(x))
		x = torch.relu(self.l2(x))
		x = self.l3(x)
		return x

def test_agent(num_episodes=10, render=False):
    # If using a local machine, you can change to "human" to watch it
    env_test = gym.make("CartPole-v1", render_mode="human" if render else None)
    test_rewards = []
    print(f"Starting testing for {num_episodes} episodes...")

    policy_net.eval() # Set network to evaluation mode
    
    for eps in range(num_episodes):
        state, _ = env_test.reset()
        state = torch.tensor(state, dtype=torch.float32, device = device).unsqueeze(0)
        total_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                # Always take the best action (Exploitation)
                action = policy_net(state).max(1)[1].view(1, 1)
            
            next_state, reward, terminated, truncated, _ = env_test.step(action.item())
            #time.sleep(0.5)
            total_reward += reward
            done = terminated or truncated
            
            state = torch.tensor(next_state, dtype=torch.float32, device = device).unsqueeze(0)
            
        test_rewards.append(total_reward)
        print(f"Test Episode {eps+1}: Reward = {total_reward}")
    env_test.close()
    print(f"\nAverage Test Reward: {np.mean(test_rewards)}")
    
    # Plotting the test results
    plt.figure(figsize=(10,5))
    plt.bar(range(1, num_episodes + 1), test_rewards, color='orange')
    plt.axhline(y=200, color='r', linestyle='--', label='Solved Threshold (v0)')
    plt.title("Agent Performance (No Exploration)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

# Run the test
env = gym.make("CartPole-v1")
action_size = env.action_space.n # 2 (left, right)
print("Action size: ", action_size)
state_size = env.observation_space.shape[0] # 4 (cart position, cart velocity, pole angle, pole angular velocity)
print("State size: ", state_size)
env = None
device = 'cpu'
policy_net= DQN(state_size, action_size)
policy_net.load_state_dict(torch.load('cart-pole-v1-model_parameters.pth', map_location=torch.device(device)))
test_agent(num_episodes=10, render = True)