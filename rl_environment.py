import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

class RLGridEnvironment:
    def __init__(self, grid_size, agent_start, target_position, walls, max_steps=1000):
        self.grid_size = grid_size
        self.agent_start = agent_start
        self.target_position = target_position
        self.walls = walls
        self.agent_position = agent_start
        self.max_steps = max_steps

        # Define the action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = [0, 1, 2, 3]

    def reset(self):
        self.agent_position = self.agent_start
        return self.agent_position

    def step(self, action):
        new_position = list(self.agent_position)

        # Actions: UP, DOWN, LEFT, RIGHT
        if action == 0:  # UP
            new_position[0] -= 1
        elif action == 1:  # DOWN
            new_position[0] += 1
        elif action == 2:  # LEFT
            new_position[1] -= 1
        elif action == 3:  # RIGHT
            new_position[1] += 1

        # Check boundaries and walls
        if (0 <= new_position[0] < self.grid_size[0]) and (0 <= new_position[1] < self.grid_size[1]):
            if tuple(new_position) not in self.walls:
                self.agent_position = tuple(new_position)

        # Calculate reward
        done = self.agent_position == self.target_position
        if done:
            reward = 50.0
        elif self.agent_position == tuple(new_position):
            reward = -1.0
        else:
            reward = -0.1



        return self.agent_position, reward, done


    def animate_path(self, path, save_as=None):
        fig, ax = plt.subplots()
        grid = np.zeros(self.grid_size, dtype=int)

        for wall in self.walls:
            grid[wall] = -1  # Walls
        grid[self.target_position] = 2  # Goal

        def update(frame):
            ax.clear()
            agent_pos = path[frame]
            grid[agent_pos] = 1  # Agent position
            ax.imshow(grid, cmap="coolwarm", interpolation="nearest")
            ax.set_title(f"Step {frame + 1}")
            grid[agent_pos] = 0  # Reset agent for the next frame

        ani = animation.FuncAnimation(fig, update, frames=len(path), repeat=False)

        # Save the animation if save_as is provided
        if save_as:
            if save_as.endswith(".mp4"):
                writer = animation.FFMpegWriter(fps=2, metadata=dict(artist="RL Agent"))
                ani.save(save_as, writer=writer)
                print(f"Animation saved as {save_as}")
            elif save_as.endswith(".gif"):
                writer = animation.PillowWriter(fps=2)
                ani.save(save_as, writer=writer)
                print(f"Animation saved as {save_as}")
        else:
            plt.show()
