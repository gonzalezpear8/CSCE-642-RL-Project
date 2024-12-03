import numpy as np

def train_agent_on_image(
    image_path, model_path, num_episodes, learning_rate, discount_factor,
    epsilon, epsilon_decay, epsilon_min, q_table
):
    from rl_environment import RLGridEnvironment
    from yolo_inference import predict_image
    
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

    if not target_position:
        print(f"No target found in {image_path}. Skipping...")
        return None, None

    # Create a new environment for the current image
    env = RLGridEnvironment(grid_size=(16, 16), agent_start=agent_start, target_position=target_position, walls=walls)
    rewards_per_episode = []
    errors = []  # To store the Q-table errors

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(env.max_steps):
            # Save a copy of the current Q-table
            q_table_prev = q_table.copy()

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

            # Compute the error (absolute difference between current and previous Q-table)
            error = np.abs(q_table - q_table_prev).sum()  # Sum of all absolute differences
            errors.append(error)

            if done:
                break

            state = next_state

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    return np.mean(rewards_per_episode), errors
