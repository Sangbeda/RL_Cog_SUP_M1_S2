import numpy as np


class Environment:
    """
    The parent class of our two different environments
    """

    def __init__(self):
        """
        This is just a placeholder that the children of this class can fill up.
        The transitions will be a big dictionary, mapping:
            - every state to a dictionary that maps
                - each action to a dictionary that contains
                    - a list of possible arriving states, their corresponding probability distribution and rewards
        """
        self._initial_state = None
        self._current_state = None
        self._transitions = None
        return

    # Getters ----------------------------------------------------------------------------------------------------------
    def get_states(self) -> list:
        """
        What are the available states in the environment?
        :return: a list of the available states in the environment
        """
        return list(self._transitions.keys())

    def get_actions(self) -> list:
        """
        What are the available actions in the environment?
        :return: a list of the available actions in the environment
        """
        return list(self._transitions[self._current_state].keys())

    def get_current_state(self) -> int:
        """
        What state is the agent currently occupying?
        :return: the current state
        """
        return self._current_state

    # Actions of the environment ---------------------------------------------------------------------------------------
    def take_action(self, action: str) -> dict:
        """
        Takes a single step in the environment
        :param action: the action we choose to take
        :return: a dictionary containing the arrival state, the corresponding reward, and whether this was the end of
            an episode and whether this was a common transition
        """
        if action not in self._transitions[self._current_state].keys():
            raise ValueError(f'Action "{action}" is not an available action.')

        possible_states = self._transitions[self._current_state][action]['state']
        possible_rewards = self._transitions[self._current_state][action]['reward']
        transition_proba = self._transitions[self._current_state][action]['proba']
        random_index = np.random.choice(range(len(possible_states)), p=transition_proba)
        self._current_state = possible_states[random_index]
        reward = possible_rewards[random_index]
        end_of_episode = reward > 0  # TODO this is very simplistic and does not work for every environment!
        common = transition_proba[random_index] == max(transition_proba)
        return {'state': self._current_state, 'reward': reward, 'terminated': end_of_episode, 'common': common}

    def reset(self) -> int:
        """
        Resets the agent to the start location
        """
        self._current_state = self._initial_state
        return self._current_state


class TolmanMaze(Environment):
    """
    A child to the Environment class.
    This is a discrete instantiation of the Tolman maze following the work of Martinet et al. (2011)
    """

    def __init__(self) -> None:
        """
        Setting up the Tolman Maze.
        :return:
        """
        super().__init__()
        self._initial_state = 0
        self._current_state = self._initial_state
        self._transitions = {0: {'LEFT': {'state': [0], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [2], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [0], 'proba': [1], 'reward': [0]}},
                             1: {'LEFT': {'state': [1], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [6], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [1], 'proba': [1], 'reward': [0]}},
                             2: {'LEFT': {'state': [1], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [5], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [3], 'proba': [1], 'reward': [0]}},
                             3: {'LEFT': {'state': [3], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [3], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [4], 'proba': [1], 'reward': [0]}},
                             4: {'LEFT': {'state': [4], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [9], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [4], 'proba': [1], 'reward': [0]}},
                             5: {'LEFT': {'state': [5], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [8], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [5], 'proba': [1], 'reward': [0]}},
                             6: {'LEFT': {'state': [6], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [6], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [7], 'proba': [1], 'reward': [0]}},
                             7: {'LEFT': {'state': [7], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [7], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [8], 'proba': [1], 'reward': [0]}},
                             8: {'LEFT': {'state': [8], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [10], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [8], 'proba': [1], 'reward': [0]}},
                             9: {'LEFT': {'state': [9], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [12], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [9], 'proba': [1], 'reward': [0]}},
                             10: {'LEFT': {'state': [10], 'proba': [1], 'reward': [0]},
                                  'UP': {'state': [13], 'proba': [1], 'reward': [0]},
                                  'RIGHT': {'state': [10], 'proba': [1], 'reward': [0]}},
                             11: {'LEFT': {'state': [10], 'proba': [1], 'reward': [0]},
                                  'UP': {'state': [11], 'proba': [1], 'reward': [0]},
                                  'RIGHT': {'state': [11], 'proba': [1], 'reward': [0]}},
                             12: {'LEFT': {'state': [11], 'proba': [1], 'reward': [0]},
                                  'UP': {'state': [12], 'proba': [1], 'reward': [0]},
                                  'RIGHT': {'state': [12], 'proba': [1], 'reward': [0]}},
                             13: {'LEFT': {'state': [13], 'proba': [1], 'reward': [0]},
                                  'UP': {'state': [self._initial_state], 'proba': [1], 'reward': [10]},
                                  'RIGHT': {'state': [13], 'proba': [1], 'reward': [0]}}}
        return

    def obstruct(self) -> None:
        """
        Obstructs the middle corridor in the Tolman Maze
        """
        self._transitions[2]['UP'] = {'state': [2], 'proba': [1], 'reward': [0]}
        return


class MattarDawMaze(Environment):
    """
    A child to the Environment class.
    This is a discrete instantiation of the maze from Mattar & Daw (2018)
    """

    def __init__(self) -> None:
        """
        Setting up the Tolman Maze.
        :return:
        """
        super().__init__()
        self._initial_state = 15
        self._current_state = self._initial_state
        self._transitions = {
            0: {
                'UP': {'state': [0], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [1], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [8], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [0], 'proba': [1], 'reward': [0]}
            },
            1: {
                'UP': {'state': [1], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [2], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [9], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [0], 'proba': [1], 'reward': [0]}
            },
            2: {
                'UP': {'state': [2], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [3], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [2], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [1], 'proba': [1], 'reward': [0]}
            },
            3: {
                'UP': {'state': [3], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [4], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [10], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [2], 'proba': [1], 'reward': [0]}
            },
            4: {
                'UP': {'state': [4], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [5], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [11], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [3], 'proba': [1], 'reward': [0]}
            },
            5: {
                'UP': {'state': [5], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [6], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [12], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [4], 'proba': [1], 'reward': [0]}
            },
            6: {
                'UP': {'state': [6], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [6], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [13], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [5], 'proba': [1], 'reward': [0]}
            },
            7: {
                'UP': {'state': [self._initial_state], 'proba': [1], 'reward': [10]},
                'RIGHT': {'state': [7], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [14], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [7], 'proba': [1], 'reward': [0]}
            },
            8: {
                'UP': {'state': [0], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [9], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [15], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [8], 'proba': [1], 'reward': [0]}
            },
            9: {
                'UP': {'state': [1], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [9], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [16], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [8], 'proba': [1], 'reward': [0]}
            },
            10: {
                'UP': {'state': [3], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [11], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [17], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [10], 'proba': [1], 'reward': [0]}
            },
            11: {
                'UP': {'state': [4], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [12], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [18], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [10], 'proba': [1], 'reward': [0]}
            },
            12: {
                'UP': {'state': [5], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [13], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [19], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [11], 'proba': [1], 'reward': [0]}
            },
            13: {
                'UP': {'state': [6], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [13], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [20], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [12], 'proba': [1], 'reward': [0]}
            },
            14: {
                'UP': {'state': [7], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [14], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [21], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [14], 'proba': [1], 'reward': [0]}
            },
            15: {
                'UP': {'state': [8], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [16], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [22], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [15], 'proba': [1], 'reward': [0]}
            },
            16: {
                'UP': {'state': [9], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [16], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [23], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [15], 'proba': [1], 'reward': [0]}
            },
            17: {
                'UP': {'state': [10], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [18], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [24], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [17], 'proba': [1], 'reward': [0]}
            },
            18: {
                'UP': {'state': [11], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [19], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [25], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [17], 'proba': [1], 'reward': [0]}
            },
            19: {
                'UP': {'state': [12], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [20], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [26], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [18], 'proba': [1], 'reward': [0]}
            },
            20: {
                'UP': {'state': [13], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [20], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [27], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [19], 'proba': [1], 'reward': [0]}
            },
            21: {
                'UP': {'state': [14], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [21], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [29], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [21], 'proba': [1], 'reward': [0]}
            },
            22: {
                'UP': {'state': [15], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [23], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [30], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [22], 'proba': [1], 'reward': [0]}
            },
            23: {
                'UP': {'state': [16], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [23], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [31], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [22], 'proba': [1], 'reward': [0]}
            },
            24: {
                'UP': {'state': [17], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [25], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [33], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [24], 'proba': [1], 'reward': [0]}
            },
            25: {
                'UP': {'state': [18], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [26], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [34], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [24], 'proba': [1], 'reward': [0]}
            },
            26: {
                'UP': {'state': [19], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [27], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [26], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [25], 'proba': [1], 'reward': [0]}
            },
            27: {
                'UP': {'state': [20], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [28], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [35], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [26], 'proba': [1], 'reward': [0]}
            },
            28: {
                'UP': {'state': [28], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [29], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [36], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [27], 'proba': [1], 'reward': [0]}
            },
            29: {
                'UP': {'state': [21], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [29], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [37], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [28], 'proba': [1], 'reward': [0]}
            },
            30: {
                'UP': {'state': [22], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [31], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [38], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [30], 'proba': [1], 'reward': [0]}
            },
            31: {
                'UP': {'state': [23], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [32], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [39], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [30], 'proba': [1], 'reward': [0]}
            },
            32: {
                'UP': {'state': [32], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [33], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [40], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [31], 'proba': [1], 'reward': [0]}
            },
            33: {
                'UP': {'state': [24], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [34], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [41], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [32], 'proba': [1], 'reward': [0]}
            },
            34: {
                'UP': {'state': [25], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [34], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [42], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [33], 'proba': [1], 'reward': [0]}
            },
            35: {
                'UP': {'state': [27], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [36], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [44], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [35], 'proba': [1], 'reward': [0]}
            },
            36: {
                'UP': {'state': [28], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [37], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [45], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [35], 'proba': [1], 'reward': [0]}
            },
            37: {
                'UP': {'state': [29], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [37], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [46], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [36], 'proba': [1], 'reward': [0]}
            },
            38: {
                'UP': {'state': [30], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [39], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [38], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [38], 'proba': [1], 'reward': [0]}
            },
            39: {
                'UP': {'state': [31], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [40], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [39], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [38], 'proba': [1], 'reward': [0]}
            },
            40: {
                'UP': {'state': [32], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [41], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [40], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [39], 'proba': [1], 'reward': [0]}
            },
            41: {
                'UP': {'state': [33], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [42], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [41], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [40], 'proba': [1], 'reward': [0]}
            },
            42: {
                'UP': {'state': [34], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [43], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [42], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [41], 'proba': [1], 'reward': [0]}
            },
            43: {
                'UP': {'state': [43], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [44], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [43], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [42], 'proba': [1], 'reward': [0]}
            },
            44: {
                'UP': {'state': [35], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [45], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [44], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [43], 'proba': [1], 'reward': [0]}
            },
            45: {
                'UP': {'state': [36], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [46], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [45], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [44], 'proba': [1], 'reward': [0]}
            },
            46: {
                'UP': {'state': [37], 'proba': [1], 'reward': [0]},
                'RIGHT': {'state': [46], 'proba': [1], 'reward': [0]},
                'DOWN': {'state': [46], 'proba': [1], 'reward': [0]},
                'LEFT': {'state': [45], 'proba': [1], 'reward': [0]}
            }
        }
        return

    def _change_reward_location(self): 

        self._transitions[7]['UP'] = {'state': [7], 'proba': [1], 'reward': [0]}
        self._transitions[46]['DOWN'] = {'state': [46], 'proba': [1], 'reward': [0]}
        self._transitions[46]['UP'] = {'state': [self._initial_state], 'proba': [1], 'reward': [10]}

        
