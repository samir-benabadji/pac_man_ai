import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torchvision import transforms
from PIL import Image

from model import Network

# Hyperparameters
learning_rate = 5e-4
minibatch_size = 64
discount_factor = 0.99

def preprocess_frame(frame):
    """
    Converts a raw Atari frame to a torch tensor:
      1) Resizes it to 128 x 128
      2) Converts it to a PyTorch tensor
      3) Returns with an extra batch dimension
    """
    frame_pil = Image.fromarray(frame)
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return preprocess(frame_pil).unsqueeze(0)

class Agent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = deque(maxlen=10000)

    def step(self, state, action, reward, next_state, done):
        # Convert frames to tensors
        state_t = preprocess_frame(state)
        next_state_t = preprocess_frame(next_state)
        self.memory.append((state_t, action, reward, next_state_t, done))

        # Learn if enough samples in memory
        if len(self.memory) > minibatch_size:
            experiences = random.sample(self.memory, k=minibatch_size)
            self.learn(experiences)

    def act(self, state, epsilon=0.):
        """
        Returns an action using the epsilon-greedy policy.
        """
        state_t = preprocess_frame(state).to(self.device)

        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state_t)
        self.local_qnetwork.train()

        # Epsilon-greedy
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """
        Update value parameters using a batch of experience tuples.
        """
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        actions = torch.from_numpy(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(1).to(self.device)
        dones = torch.from_numpy(np.array(dones, dtype=np.uint8)).unsqueeze(1).to(self.device)

        # Get max predicted Q values for next states from target model
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        q_targets = rewards + (discount_factor * next_q_targets * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.local_qnetwork(states).gather(1, actions)

        # Compute the loss
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
