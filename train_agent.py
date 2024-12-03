import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from train_logic import train_agent_on_image
from rl_environment import RLGridEnvironment
import shutil

# Paths
model_path = "best.pt"
input_path = "dataset/RGBD/images/single_image/"  # Can be a folder of images or a video file
# Training parameters
num_episodes_per_image = 6000
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.1

# Initialize Q-table
q_table = np.zeros((16, 16, 4))
rewards_per_image = []
all_errors = []

def extract_frames_from_video(video_path, output_folder="temp_frames", frame_skip=1):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []

    frame_count, saved_count = 0, 0
    frame_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (frame_skip + 1) == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_paths.append(frame_filename)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from video {video_path} to {output_folder}")
    return frame_paths

def get_image_paths(input_path):
    if os.path.isdir(input_path):
        return glob.glob(os.path.join(input_path, "*.jpg"))
    elif os.path.isfile(input_path) and input_path.endswith((".mp4", ".avi")):
        return extract_frames_from_video(input_path, frame_skip=1)
    else:
        raise ValueError(f"Invalid input path: {input_path}")

def visualize_agent_path(env, q_table, save_as=None):
    state = env.reset()
    path = [state]

    for _ in range(env.max_steps):
        action = np.argmax(q_table[state])
        next_state, _, done = env.step(action)
        path.append(next_state)
        if done:
            break

    env.animate_path(path, save_as=save_as)

parser = argparse.ArgumentParser(description="Train RL agent on images or video")
parser.add_argument("--visualize", action="store_true", help="Visualize the agent's path after training")
parser.add_argument("--save_animation", type=str, help="Path to save the animation (e.g., 'agent_animation.mp4')")
args = parser.parse_args()

image_paths = get_image_paths(input_path)

for image_path in image_paths:
    print(f"Processing image: {image_path}")
    avg_reward, errors = train_agent_on_image(
        image_path, model_path, num_episodes_per_image, learning_rate,
        discount_factor, epsilon, epsilon_decay, epsilon_min, q_table
    )
    if avg_reward is not None:
        rewards_per_image.append(avg_reward)
        all_errors.extend(errors)
        print(f"Finished training on {image_path}: Avg Reward = {avg_reward:.2f}")

np.save("q_table.npy", q_table)

plt.figure()
plt.plot(range(len(rewards_per_image)), rewards_per_image)
plt.xlabel("Image Index")
plt.ylabel("Average Reward")
plt.title("Training Progress Across Images")
plt.show()

plt.figure()
plt.plot(range(len(all_errors)), all_errors)
plt.xlabel("Steps")
plt.ylabel("Error in Estimates")
plt.title("Error During Training")
plt.show()

if args.visualize:
    save_path = args.save_animation or "agent_animation.mp4"
    walls = [(1, 2), (3, 4)]
    target_position = (5, 5)
    env = RLGridEnvironment(grid_size=(16, 16), agent_start=(0, 0), target_position=target_position, walls=walls)
    visualize_agent_path(env, q_table, save_as=save_path)

if os.path.isdir("temp_frames"):
    shutil.rmtree("temp_frames", ignore_errors=True)
    print("Temporary frames cleaned up.")
