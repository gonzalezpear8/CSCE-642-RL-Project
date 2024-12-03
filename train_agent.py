from rl_environment import RLGridEnvironment
from yolo_inference import predict_image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Paths
model_path = "best.pt"

# for signle image training
# image_folder = "dataset/RGBD/images/single_image"

# for multiple image training
image_folder = "dataset/RGBD/images/train"

# Get all images in the folder
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

# Training parameters
num_episodes_per_image = 1000  # Episodes to train for each image
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1

# Initialize Q-table
q_table = np.zeros((16, 16, 4))
rewards_per_image = []

# Train on each image
for image_path in image_paths:
    print(f"Processing image: {image_path}")
    
    # YOLO inference
    detections = predict_image(model_path, image_path)

    walls = []
    target_position = None
    agent_start = (0, 0)

    # Parse YOLO detections
    for detection in detections:
        if detection["class"] == 1:  # Wall
            walls.append((detection["grid_y"], detection["grid_x"]))
        elif detection["class"] == 0:  # Goal
            target_position = (detection["grid_y"], detection["grid_x"])

    # Skip the image if no egg is found
    if not target_position:
        print(f"No target found in {image_path}. Skipping...")
        continue

    # Create a new environment for the current image
    env = RLGridEnvironment(grid_size=(16, 16), agent_start=agent_start, target_position=target_position, walls=walls)

    rewards_per_episode = []

    for episode in range(num_episodes_per_image):
        state = env.reset()
        total_reward = 0

        for _ in range(env.max_steps):
            # Epsilon-greedy policy
            if np.random.random() < epsilon:
                action = np.random.choice(env.action_space)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Update Q-table
            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])
            q_table[state][action] = old_value + learning_rate * (reward + discount_factor * next_max - old_value)

            if done:
                break

            state = next_state

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # average reward for this image
    rewards_per_image.append(np.mean(rewards_per_episode))
    print(f"Finished training on {image_path}: Avg Reward = {np.mean(rewards_per_episode):.2f}")

# Save the Q-table
np.save("q_table.npy", q_table)
# Plot training rewards
plt.plot(range(len(rewards_per_image)), rewards_per_image)
plt.xlabel("Image Index")
plt.ylabel("Average Reward")
plt.title("Training Progress Across Images")
plt.show()


