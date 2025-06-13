import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_reward_rates(reward_rates: dict) -> matplotlib.axes.Axes:
    """
    Plots the average reward rate over the course of all episodes
    """
    data = pd.DataFrame()
    for key, value in reward_rates.items():
        if not value:
            continue
        reward_rates = np.array(value)
        df = pd.DataFrame(np.transpose(reward_rates))
        df['episode'] = range(reward_rates.shape[1])
        df = df.melt(id_vars='episode')
        df['agent'] = [key] * len(df)
        data = pd.concat([data, df])
    data = data.rename(columns={'value': 'reward rate'})
    return sns.lineplot(data=data, x='episode', y='reward rate', hue='agent')


class MattarDawMazePlot():
    """
    An object that can plot the agent's position, the Q-values and the replay in the maze
    """

    def __init__(self, current_state: int, Q_values: list):
        """
        We initialize the three plots using data from the agent
        """
        self._maze = np.array([[0, 1, 2, 3, 4, 5, 6, np.nan, 7],
                               [8, 9, np.nan, 10, 11, 12, 13, np.nan, 14],
                               [15, 16, np.nan, 17, 18, 19, 20, np.nan, 21],
                               [22, 23, np.nan, 24, 25, 26, 27, 28, 29],
                               [30, 31, 32, 33, 34, np.nan, 35, 36, 37],
                               [38, 39, 40, 41, 42, 43, 44, 45, 46]])

        self._agent_position = np.copy(self._maze)
        self._agent_position[self._maze >= 0] = 0
        self._agent_position[self._maze == current_state] = 1

        self._replay = np.copy(self._maze)
        self._replay[self._maze >= 0] = 0

        self._Q_max = np.copy(self._maze)
        for idx_x in range(self._Q_max.shape[0]):
            for idx_y in range(self._Q_max.shape[1]):
                if self._maze[idx_x, idx_y] >= 0:
                    self._Q_max[idx_x, idx_y] = Q_values[int(self._maze[idx_x, idx_y])]

        # Preparing the plots
        plt.ion()
        self._fig, self._ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        self._ax[0].set_title("Map")
        self._ax[1].set_title("Updating")
        self._ax[2].set_title("Q-values")
        self._axim = np.array([self._ax[0].imshow(self._agent_position),
                               self._ax[1].imshow(self._replay),
                               self._ax[2].imshow(self._Q_max, vmin=0, vmax=10)])
        self._ax[0].autoscale()
        self._ax[1].autoscale()

        self._txt = np.empty((*self._maze.shape, ), dtype=matplotlib.text.Text)
        for idx_x in range(self._txt.shape[0]):
            for idx_y in range(self._txt.shape[1]):
                self._txt[idx_x, idx_y] = self._ax[2].text(idx_y, idx_x, f"{self._Q_max[idx_x, idx_y]: .2f}",
                                                      ha="center", va="center", color="w")
        plt.show()


    def update_agent_pos(self, state: int):
        """
        Updates the agent's position on the agent position plot
        """
        self._agent_position[self._agent_position > 0] = 0
        self._agent_position[self._maze == state] = 1
        self._axim[0].set_data(self._agent_position)
        self._fig.canvas.flush_events()
        plt.show()

    def update_Q_values(self, state: int, Q_max: float):
        """
        Updates the Q-value corresponding to a given state. If "clear" we erase the plot with the updates
        """
        # Updating the Replay
        max_val = np.nanmax(self._replay)
        self._replay[self._maze == state] = max_val + 1
        self._axim[1].set_data(self._replay)
        self._axim[1].autoscale()

        # Updating the Q-values
        self._Q_max[self._maze == state] = Q_max
        self._axim[2].set_data(self._Q_max)

        # Updating teh text
        [idx_x], [idx_y] = np.where(self._maze == state)
        self._txt[idx_x, idx_y].set_text(f"{Q_max: .2f}")

        # Plotting
        self._fig.canvas.flush_events()
        plt.show()

    def clear(self) -> None:
        """
        Clears the "updating" plot
        """
        self._replay = np.copy(self._maze)
        self._replay[self._maze >= 0] = 0

    
    def plot_state_value_heatmap(self, title: str = "State-Value Heatmap"):
        """
        Use the already-built self._Q_max grid and self._maze layout
        to render a masked heatmap of V(s).
        """
        import matplotlib.pyplot as plt

        # Mask out walls
        mask = np.isnan(self._maze)
        V_masked = np.ma.array(self._Q_max, mask=mask)

        # New figure (or reuse existing axes if you prefer)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(V_masked, origin='lower',
                       vmin=np.nanmin(self._Q_max), vmax=np.nanmax(self._Q_max))
        ax.set_title(title)
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        # Overlay the maze walls lightly
        ax.imshow(mask, cmap='gray', alpha=0.3, origin='lower')
        fig.colorbar(im, ax=ax, label="V(s)")
        plt.show()
        return ax



