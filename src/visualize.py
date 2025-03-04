import gymnasium as gym
import glob
import io
import base64
import imageio
from IPython.display import HTML, display

from agent import Agent  # or load your saved model weights if needed
import torch

def show_video_of_model(agent, env_name='MsPacmanDeterministic-v0'):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data=f'''
            <video alt="test" autoplay loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{encoded.decode('ascii')}" type="video/mp4" />
            </video>'''))
    else:
        print("Could not find video")

if __name__ == "__main__":
    # Example usage (assuming you have a checkpoint to load):
    # 1) Re-create the agent with the same architecture
    test_agent = Agent(action_size=9)  # MsPacmanDeterministic-v0 has 9 actions with full_action_space=False
    # 2) Load your trained weights
    test_agent.local_qnetwork.load_state_dict(torch.load('checkpoint.pth'))

    # 3) Generate and display the video
    show_video_of_model(test_agent, 'MsPacmanDeterministic-v0')
    show_video()
