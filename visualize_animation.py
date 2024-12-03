import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

class RLGridEnvironment:
    # Other methods...

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
