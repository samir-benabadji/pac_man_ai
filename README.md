# 🕹️ Deep Convolutional Q-Learning for Pac-Man

This repository contains a **Deep Convolutional Q-Network (DCQN)** implementation using **PyTorch** to train an AI to play **Ms. Pac-Man**. The AI learns by interacting with the game environment through **reinforcement learning**, using **convolutional neural networks (CNNs)** for feature extraction and decision-making.

---

## 📌 Features
- **Deep Q-Learning (DQN) with Convolutional Neural Networks (CNNs)**
- **Replay Memory Buffer** to stabilize training
- **\(\epsilon\)-Greedy Policy** for exploration vs. exploitation
- **Target Q-Network** for more stable learning
- Trained on the **`MsPacmanDeterministic-v0`** environment from Gymnasium

---

## 🛠 Installation & Dependencies

Make sure **Python 3.7+** is installed. Then, install the following:

```bash
pip install gymnasium
pip install "gymnasium[atari, accept-rom-license]"
pip install gymnasium[box2d]
pip install torch torchvision

<video width="640" height="360" controls> <source src="assets/video/pacman_rec.mp4" type="video/mp4"> Your browser does not support the video tag. </video>

🏗 Model Architecture
This Deep Q-Learning Agent is built using CNNs to process game frames and predict Q-values.

Neural Network Structure
4 Convolutional Layers for feature extraction from the game frames.
Batch Normalization to improve training stability.
Fully Connected Layers for decision-making.
ReLU Activation Functions for non-linearity.
Adam Optimizer for efficient weight updates.
DCQN Agent Components
Replay Memory Buffer stores past experiences.
𝜖
ϵ-Greedy Policy for balancing exploration & exploitation.
Target Q-Network to stabilize training.
Experience Sampling to train on randomized past data.


⚡ Hardware Used for Training
Training was performed on the following high-performance setup:

💻 CPU: AMD Ryzen 7 7800X3D (8 Cores @ 4.20 GHz)
🎮 GPU: NVIDIA GEFORCE RTX 4070 Ti Super
🧠 RAM: 64 GB DDR5
This setup significantly improved training speed with CUDA acceleration.
