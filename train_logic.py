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

    print(f"Image: {image_path}")
    print(f"Walls: {walls}")
    print(f"Target Position: {target_position}")

    # Create environment
    env = RLGridEnvironment(grid_size=(16, 16), agent_start=agent_start, target_position=target_position, walls=walls)
    rewards_per_episode = []
    errors = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(env.max_steps):
            # Save Q-table for error calculation
            q_table_prev = q_table.copy()

            # Epsilon-greedy policy with added noise
            if np.random.random() < epsilon:
                action = np.random.choice(env.action_space)
            else:
                action_probs = np.exp(q_table[state]) / np.sum(np.exp(q_table[state]))
                action = np.random.choice(len(action_probs), p=action_probs)


            next_state, reward, done = env.step(action)

            # Distance-based reward adjustment
            prev_distance = abs(state[0] - target_position[0]) + abs(state[1] - target_position[1])
            next_distance = abs(next_state[0] - target_position[0]) + abs(next_state[1] - target_position[1])
            distance_reward = (prev_distance - next_distance) * 0.5
            reward += distance_reward


            total_reward += reward

            # Update Q-table
            try:
                old_value = q_table[state][action]
                next_max = np.max(q_table[next_state])
                q_table[state][action] = old_value + learning_rate * (
                    reward + discount_factor * next_max - old_value
                )
            except IndexError:
                print(f"Invalid Q-table state: {state} or {next_state}")
                break

            # Calculate Q-table error
            error = np.abs(q_table - q_table_prev).sum()
            errors.append(error)

            if done:
                break

            state = next_state

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Log Q-values periodically
        if episode % 100 == 0:
            print(f"Episode {episode}: Avg Reward: {np.mean(rewards_per_episode[-100:])}")
            print(f"Q-values for state {state}: {q_table[state]}")

    return np.mean(rewards_per_episode), errors
