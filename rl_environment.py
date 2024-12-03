import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class RLGridEnvironment:
    def __init__(self, grid_size, agent_start, target_position, walls, max_steps=500):
        self.grid_size = grid_size
        self.agent_start = agent_start
        self.target_position = target_position
        self.walls = walls
        self.agent_position = agent_start
        self.max_steps = max_steps

        # Defining the action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
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

        if (0 <= new_position[0] < self.grid_size[0]) and (0 <= new_position[1] < self.grid_size[1]):
            if tuple(new_position) not in self.walls:
                self.agent_position = tuple(new_position)

        done = self.agent_position == self.target_position
        reward = 1 if done else -0.01
        return self.agent_position, reward, done

    def animate_path(self, path, save_as=None):
        fig, ax = plt.subplots()
        grid = np.zeros(self.grid_size, dtype=int)

        # Mark walls and goal
        for wall in self.walls:
            grid[wall] = -1  
        grid[self.target_position] = 2  

        def update(frame):
            ax.clear()
            agent_pos = path[frame]
            grid[agent_pos] = 1  
            ax.imshow(grid, cmap="coolwarm", interpolation="nearest")
            ax.set_title(f"Step {frame + 1}")
            grid[agent_pos] = 0

        ani = animation.FuncAnimation(fig, update, frames=len(path), repeat=False)

        if save_as:
            writer = animation.PillowWriter(fps=2)
            ani.save(save_as, writer=writer)
            print(f"Animation saved as {save_as}")
        else:
            plt.show()
