import gymnasium as gym
import numpy as np
import torch
from collections import deque

from agent import Agent

def main():
    # Create the environment
    env = gym.make('MsPacmanDeterministic-v0', full_action_space=False)

    # Print env details
    state_shape = env.observation_space.shape
    number_actions = env.action_space.n
    print(f"State shape: {state_shape}")
    print(f"Number of actions: {number_actions}")

    # Create the agent
    agent = Agent(action_size=number_actions)

    # Training parameters
    number_episodes = 2000
    max_steps_per_episode = 10000
    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start
    scores_window = deque(maxlen=100)

    # Training loop
    for episode in range(1, number_episodes + 1):
        state, _ = env.reset()
        score = 0
        for _ in range(max_steps_per_episode):
            action = agent.act(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        average_score = np.mean(scores_window)
        print(f"\rEpisode {episode}\tAverage Score: {average_score:.2f}", end="")

        # Print every 100 episodes
        if episode % 100 == 0:
            print(f"\rEpisode {episode}\tAverage Score: {average_score:.2f}")

        # Save model if a certain threshold is reached
        if average_score >= 500.0:
            print(f"\nEnvironment solved in {episode-100:d} episodes!\tAverage Score: {average_score:.2f}")
            torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
            break

    env.close()

if __name__ == "__main__":
    main()
