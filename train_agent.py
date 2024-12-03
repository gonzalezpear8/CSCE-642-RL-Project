import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from train_logic import train_agent_on_image

# Paths
model_path = "best.pt"
image_folder = "dataset/RGBD/images/train"
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

# Training parameters
num_episodes_per_image = 6000
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1

# Initialize Q-table
q_table = np.zeros((16, 16, 4))
rewards_per_image = []
all_errors = []  # To collect errors across all images

# Train on each image
for image_path in image_paths:
    print(f"Processing image: {image_path}")
    avg_reward, errors = train_agent_on_image(
        image_path, model_path, num_episodes_per_image, learning_rate,
        discount_factor, epsilon, epsilon_decay, epsilon_min, q_table
    )
    if avg_reward is not None:
        rewards_per_image.append(avg_reward)
        all_errors.extend(errors)  # Collect all errors across images
        print(f"Finished training on {image_path}: Avg Reward = {avg_reward:.2f}")

# Save the Q-table
np.save("q_table.npy", q_table)

# Plot training rewards
plt.figure()
plt.plot(range(len(rewards_per_image)), rewards_per_image)
plt.xlabel("Image Index")
plt.ylabel("Average Reward")
plt.title("Training Progress Across Images")
plt.show()

# Plot error values
plt.figure()
plt.plot(range(len(all_errors)), all_errors)
plt.xlabel("Steps")
plt.ylabel("Error in Estimates")
plt.title("Error During Training")
plt.show()
