from environments import *
import random
from plotter import *


class Event:
    """
    This class will represent events that we have experienced
    """

    def __init__(self, state: int, action: str, reward: float, arrival_state: int, priority: float | None) -> None:
        """
        The Event consists of a state and action we experienced, and also a corresponding priority
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.arrival_state = arrival_state
        self.priority = priority

    def __gt__(self, other):
        """
        The > operator between 2 Events. We need this to be able to sort arrays
        """
        return self.priority > other.priority


class RLagent:
    """
    This is the parent class of all agents.
    """

    def __init__(self, environment: Environment, gamma=0.9, epsilon=0., **kwargs) -> None:
        """
        Setting up the agent. The Q-table will be a dictinoary mapping states to dictionaries, where each dict maps
        actions to their corresponding Q-values
        :kwargs replay_type: "random", "forward", "backward", "prioritized" or "trajectory"
                max_replay: the maximum number of replay steps, None meaning no limit
                replay_threshold: the minimum change in Q-values that will still elicit a replay event. A float value
                    between 0 and infinity (None). If max_replay is infinite, this value has to be finite.
                plot_from: [int] Plot from the nth episode. If None, no plotting takes place
        :return:
        """
        self._gamma = gamma
        self._epsilon = epsilon
        self._Q = {state: {action: 0 for action in environment.get_actions()}
                   for state in environment.get_states()}
        self._current_state = environment.get_current_state()

        # About the replay
        self._memory_buffer = []
        self._replay_type = kwargs.get('replay_type', None)
        if self._replay_type not in [None, 'random', 'forward', 'backward', 'prioritized', 'trajectory']:
            raise ValueError(f'{self._replay_type} is not a valid replay type.')
        self._buffer_size = kwargs.get('max_replay', None)
        self._replay_threshold = kwargs.get('replay_threshold', None)
        if self._replay_type is not None and self._buffer_size is None and self._replay_threshold is None:
            raise ValueError(f'When {self._replay_type}-type replay is used, "max_replay" or "replay_threshold" '
                             f'needs to be specified.')
        self._plot_from = kwargs.get('plot_from', None)
        self._episode = 0
        self._plotter = None
        return

    # Private methods --------------------------------------------------------------------------------------------------
    def __take_step__(self, arrival_state: int) -> None:
        """
        Updates the current state in the agent
        """
        self._current_state = arrival_state
        if self._plotter is not None:
            self._plotter.update_agent_pos(self._current_state)
        return

    def __Q_max__(self, state: int, possible_actions=None) -> dict:
        """
        Returns the best Q-value and the corresponding action in a given state
        :param state: the state of interest
        :param possible_actions: the list of action from which we intend to choose the best. If not specified, it means
            all actions can be considered
        :return: a dict containing both the Q-value and the corresponding actions (list)
        """
        if possible_actions is None:
            possible_actions = list(self._Q[state].keys())
        Q_max = {'value': 0, 'action': [possible_actions[0]]}
        for action in possible_actions:
            if self._Q[state][action] > Q_max['value']:
                Q_max['value'] = self._Q[state][action]
                Q_max['action'] = [action]
            elif self._Q[state][action] == Q_max['value']:
                Q_max['action'].append(action)
        return Q_max

    def __known_actions__(self, state:int) -> list:
        """
        Returns all the actions corresponding to a given state
        """
        return list(self._Q[state].keys())
    
    # Private methods related to the replay ----------------------------------------------------------------------------
    def __store_event__(self, state: int, action: str, reward: float, arrival_state: int,
                        priority: float | None) -> None:
        """
        This method will create an Event-type object and either
            - Overwrite an existing memory in the memory buffer; or
            - Create a new memory
        If the memory buffer is full, then this function will delete the oldest memory as well
        """
        # 1) In case of prioritized sweeping, we have to occasionally overwrite an element's priority
        if self._replay_type == 'prioritized':
            for event in self._memory_buffer:
                if event.state == state and event.action == action:
                    event.priority = max(event.priority, abs(priority))
                    return

        # 2) Otherwise, if it is not in the buffer, let's add it to the front
        event = Event(state=state, action=action, reward=reward, arrival_state=arrival_state, priority=abs(priority))
        self._memory_buffer.insert(0, event)

        # 2a) Or if we are prioritized, it will not actually be the front
        if self._replay_type == 'prioritized':
            self._memory_buffer.sort(reverse=True)

        # 3) Delete the oldest / least important element
        if len(self._memory_buffer) > self._buffer_size:
            self._memory_buffer.pop()
        return

    def __classical_replay__(self):
        """
        Implements the classical (random, forward, backwards) replay algorithms
        """
        if self._plotter is not None:
            self._plotter.clear()

        # 1) Prepare the buffer that we are iterating through
        memory_buffer = self._memory_buffer.copy()
        if self._replay_type == 'forward':
            memory_buffer.reverse()
        elif self._replay_type == 'random':
            random.shuffle(memory_buffer)

        # 2) Now we need to iterate until we are under threshold or over the specified number of steps
        replay_steps = 0
        max_Q_change = self._replay_threshold
        while abs(max_Q_change) >= self._replay_threshold:
            max_Q_change = 0
            for event in memory_buffer:
                delta_Q = self.reinforcement_learning(state=event.state, action=event.action, reward=event.reward,
                                                      arrival_state=event.arrival_state)
                if abs(delta_Q) > abs(max_Q_change):
                    max_Q_change = delta_Q
                replay_steps += 1
                if replay_steps >= self._buffer_size:
                    return

    def __prioritized_sweeping__(self):
        """
        Implements the prioritized sweeping algorithm
        """
        # In case of prioritized sweeping, we will update the memory buffer real time
        if self._plotter is not None:
            self._plotter.clear()
        for replay_step in range(self._buffer_size):
            event = self._memory_buffer.pop(0)
            self.reinforcement_learning(state=event.state, action=event.action, reward=event.reward,
                                                  arrival_state=event.arrival_state)
            if not self._memory_buffer:
                return
        return

    # Public methods to instruct the agent -----------------------------------------------------------------------------
    def get_pos(self) -> int:
        """
        Gets the agent's current position
        """
        return self._current_state

    def move(self, state: int) -> None:
        """
        Moves the agent to a given location
        """
        self._current_state = state
        return

    def pre_train(self, action: str, arrival_state: int, reward: float) -> None:
        """
        Placeholder for the pre-training of the MB agents
        """
        self.__take_step__(arrival_state=arrival_state)
        pass
    
    def action_selection(self, state=None) -> dict:
        """
        Chooses what action to take based on the state the agent is in
        :param state: the state in which the action needs to be chosen. If None is specified, the current state is used
            Note: if the state is specified, it is considered to be a virtual step, thus a more conservative action
            selection is used wherein only those actions are considered which have been experienced
        :return: the chosen action
        """
        if state is None:
            state = self._current_state
            possible_actions = list(self._Q[state].keys())
        else:
            possible_actions = self.__known_actions__(state=state)
            if not possible_actions:
                return {}
        if np.random.uniform(0, 1, 1) <= self._epsilon:
            random_index = np.random.choice(range(len(possible_actions)))
            action = possible_actions[random_index]
            return {'action': action, 'Q-value': self._Q[state][action]}
        greedy_actions = self.__Q_max__(state=state, possible_actions=possible_actions)['action']
        action = np.random.choice(greedy_actions)
        return {'action': action, 'Q-value': self._Q[state][action]}

    def reinforcement_learning(self, action: str, arrival_state: int, reward: float, state=None, terminated=False) -> float:
        """
        This is a placeholder function for all the learning algorithms down the line. It does two things. Updates the
        agent's current location and initializes the plot if needed
        :param action: the action taken
        :param arrival_state: the state the agent arrived in after taking the action
        :param reward: the reward received over the state transition
        :param state: if a state is specified, it means I am not learning the current state, ergo it is a replay step
        :param terminated: it is the end of an episode
        """
        if state is None:
            self.__take_step__(arrival_state=arrival_state)
            if terminated:
                self._episode += 1
        if self._plotter is None:
            if self._plot_from == self._episode:
                self._plotter = MattarDawMazePlot(current_state=self._current_state,
                                                  Q_values=[self.__Q_max__(state)['value'] for state in self._Q.keys()])
        if self._plotter is not None and state is None:
            self._plotter.clear()
        return 0

    def memory_replay(self):
        """
        This function will perform the memory replay, by choosing the appropriate replay function.
        """

        if self._replay_type in ['random', 'forward', 'backward']:
            self.__classical_replay__()
        elif self._replay_type == 'prioritized':
            self.__prioritized_sweeping__()
        return


class MFagent(RLagent):
    """
    This is the model-free agent that will use the Q-learning algorithm
    """

    def __init__(self, environment: Environment, gamma=0.9, epsilon=0.05, alpha=0.3, **kwargs):
        """
        Setting up the MF agent. Unlike its parent (and little brother, the MB agent) this agent needs an alpha param
        """
        super().__init__(environment=environment, gamma=gamma, epsilon=epsilon, **kwargs)
        self._alpha = alpha
        return

    # The hidden methods necessary for Q-learning ----------------------------------------------------------------------
    def __Q_learning__(self, action: str, arrival_state: int, reward: float, state: int) -> float:
        """
        The Q-learning algorithm
        If state is None, we are learning the Q-value of the current state
        """
        Q_max = self.__Q_max__(state=arrival_state)['value']
        TD_error = reward + self._gamma * Q_max - self._Q[state][action]
        self._Q[state][action] += self._alpha * TD_error
        if self._plotter is not None:
            self._plotter.update_Q_values(state=state, Q_max=self.__Q_max__(state)['value'])
        return self._alpha * TD_error

    # We overwrite the RL method coming from the parent class ----------------------------------------------------------
    def reinforcement_learning(self, action: str, arrival_state: int, reward: float, state=None,
                               terminated=False) -> float:
        """
        The MF agent will simply use Q-learning to update its Q-values.
        :param action: the action taken
        :param arrival_state: the state the agent arrived in after taking the action
        :param reward: the reward received over the state transition
        :param state: if a state is specified, it means I am not learning the current state, ergo it is a replay step
        :param terminated: is the episode terminated?
        """
        curr_state = state
        if curr_state is None:
            curr_state = self._current_state
        super().reinforcement_learning(action=action, arrival_state=arrival_state, reward=reward, state=state,
                                       terminated=terminated)
        delta_Q = self.__Q_learning__(action=action, arrival_state=arrival_state, reward=reward, state=curr_state)
        if state is None and self._replay_type not in [None, 'trajectory']:
            self.__store_event__(state=curr_state, action=action, reward=reward, arrival_state=arrival_state,
                                 priority=delta_Q)
        return delta_Q


class MBagent(RLagent):
    """
    This is the model-based agent that will use the VI algorithm
    """

    def __init__(self, environment: Environment, gamma=0.9, epsilon=0.05, theta=0.0001, window_length=10,
                 **kwargs) -> None:
        """
        Setting up the MF agent. This agent will also store a model of the environment in the form of a reward- and a
        transition function. This means that this agent will need a convergence threshold for its calculations (theta)
        and a window length for updating its model
        :kwargs predecessors: Should predecessors be added to the prioritized sweeping array?
        """
        self._theta = theta
        self._window_length = window_length
        super().__init__(environment=environment, gamma=gamma, epsilon=epsilon, **kwargs)
        self._reward_history = {state: {action: [] for action in environment.get_actions()}
                                for state in environment.get_states()}
        self._reward_fun = {state: {action: 0 for action in environment.get_actions()}
                            for state in environment.get_states()}
        self._transition_history = {state: {action: []
                                            for action in environment.get_actions()}
                                    for state in environment.get_states()}
        self._transition_function = {state: {action: [1 / len(environment.get_states())] * len(environment.get_states())
                                             for action in environment.get_actions()}
                                     for state in environment.get_states()}
        self._predecessors = kwargs.get('predecessors', False)
        if self._predecessors and self._replay_type != 'prioritized':
            raise ValueError('Predecessor search only makes sense while using prioritized sweeping')
        return
    
    # Hidden methods unique to the MB agent ----------------------------------------------------------------------------
    def __known_actions__(self, state:int) -> list:
        """
        Returns all the known actions corresponding to a given state (i.e. the actions already taken)
        """
        possible_actions = super().__known_actions__(state)
        known_actions = []
        for action in possible_actions:
            cumul_history = [sum(transition) for transition in zip(*self._transition_history[state][action])]
            if cumul_history and sum(cumul_history) > 0:
                known_actions.append(action)
        return known_actions
    
    def __update_model__(self, state: int, action: str, arrival_state: int, reward=None) -> None:
        """
        Updates the world model based on the experienced transition
        :param state: The state in which the update takes place
        :param action: the action taken
        :param arrival_state: the state the agent arrived in after taking the action
        :param reward: the reward received over the state transition. If None, we're in pre-training mode and not
            learning a reward function
        """
        if reward is not None:
            self._reward_history[state][action].append(reward)
            if len(self._reward_history[state][action]) > self._window_length:
                self._reward_history[state][action].pop(0)
            self._reward_fun[state][action] = np.average(self._reward_history[state][action])

        transition = [0] * len(self._Q.keys())
        transition[arrival_state] = 1
        self._transition_history[state][action].append(transition)
        if len(self._transition_history[state][action]) > self._window_length:
            self._transition_history[state][action].pop(0)
        self._transition_function[state][action] = \
            np.sum(np.array(self._transition_history[state][action]), axis=0) / \
            np.sum(np.array(self._transition_history[state][action]))
        return
    
    def __one_step___value_iteration____(self, state: int, action: str) -> float:
        """
        Performs a single step of the value iteration algorithm
        """
        Q_t = self._Q[state][action]  # The current Q-value in (s, a)
        expected_V = 0  # The expected V-function of the arrival state
        for arrival_state in self._Q.keys():
            Q_max = self.__Q_max__(state=arrival_state)['value']
            expected_V += self._transition_function[state][action][arrival_state] * Q_max
        self._Q[state][action] = self._reward_fun[state][action] + self._gamma * expected_V
        if self._plotter is not None:
            self._plotter.update_Q_values(state=state, Q_max=self.__Q_max__(state)['value'])
        return self._Q[state][action] - Q_t

    # Hidden methods related to MB replay ------------------------------------------------------------------------------
    def __value_iteration__(self) -> float:
        """
        The value iteration algorithm
        """
        if self._plotter is not None:
            self._plotter.clear()
        max_Q_change = self._theta  # The biggest absolute Q-value change we experineced in the WHOLE state space
        while abs(max_Q_change) >= self._theta:
            max_Q_change = 0
            for state in self._Q.keys():
                for action in self._Q[state].keys():
                    delta_Q = self.__one_step___value_iteration____(state=state, action=action)
                    if abs(delta_Q) > abs(max_Q_change):
                        max_Q_change = delta_Q
        return max_Q_change

    def __trajectory_sampling__(self, state: int):
        """
        The trajectory sampling method of replay
        :param state: The state in which the trajectories start
        """
        if self._plotter is not None:
            self._plotter.clear()
        max_Q_change = 0
        for step in range(self._buffer_size):
            action = self.action_selection(state=state)['action']
            if not action:
                return
            arrival_state = int(np.random.choice(list(self._Q.keys()), p=self._transition_function[state][action]))
            reward = self._reward_fun[state][action]
            delta_Q = self.reinforcement_learning(state=state, action=action, reward=reward,
                                                  arrival_state=arrival_state)
            max_Q_change = max(abs(delta_Q), max_Q_change)
            if reward > 0:  # If it is the end of a simulated episode
                if max_Q_change < self._replay_threshold:
                    return
                max_Q_change = 0
            state = arrival_state

    def __predecessor_search__(self, state: int, priority: float):
        """
        Performs the predecessor search and adds each one to the memeory buffer
        :param state: The state the predecessors of which we want to find
        :param priority: The priority of the state that we will back propagate to its predecessors
        """
        if abs(priority) < self._replay_threshold:
            return
        # For every possible action ...
        for action in self._Q[state].keys():
            predecessor_states = list(self._Q.keys())
            # And every possible (predecessor) state
            for predecessor in predecessor_states:
                # We check if there is a realistic chance of a transition taking place from the predecessor,
                # via the action to the given state
                cumul_history = [sum(transition) for transition in zip(*self._transition_history[predecessor][action])]
                if cumul_history and cumul_history[state] > 0 and \
                   self._transition_function[predecessor][action][state] > 0:
                    # If there is a possibility that the transition (predecessor, action, state) took place
                    # 1) We back propagate the priority to the predecessor
                    predecessor_priority = priority * self._gamma * \
                                           self._transition_function[predecessor][action][state]
                    # 2) We store the event
                    self.__store_event__(state=predecessor, action=action,
                                         reward=self._reward_fun[predecessor][action],
                                         arrival_state=state, priority=predecessor_priority)
        return

    # Overwriting some of the paren class's public methods -------------------------------------------------------------
    def pre_train(self, action: str, arrival_state: int, reward: float) -> None:
        """
        Placeholder for the pre-training of the different types of agents
        """
        self.__update_model__(state=self._current_state, action=action, arrival_state=arrival_state, reward=None)
        super().pre_train(action=action, arrival_state=arrival_state, reward=reward)
        return

    def reinforcement_learning(self, action: str, arrival_state: int, reward: float, state=None,
                               terminated=False) -> float:
        """
        The MF agent will simply use Q-learning to update its Q-values
        :param action: the action taken
        :param arrival_state: the state the agent arrived in after taking the action
        :param reward: the reward received over the state transition
        :param state: if a state is specified, it means I am not learning the current state, ergo it is a replay step
        :param terminated: is this the end of an episode
        """
        curr_state = state
        if curr_state is None:
            curr_state = self._current_state
        super().reinforcement_learning(action=action, arrival_state=arrival_state, reward=reward, state=state,
                                       terminated=terminated)
        if state is None:  # Meaning that it is a real step
            self.__update_model__(state=curr_state, action=action, arrival_state=arrival_state, reward=reward)

        if self._replay_type is None:
            delta_Q = self.__value_iteration__()
        else:
            delta_Q = self.__one_step___value_iteration____(state=curr_state, action=action)

        if state is None and self._replay_type not in [None, 'trajectory']:
            self.__store_event__(state=self._current_state, action=action, reward=reward, arrival_state=arrival_state,
                                 priority=delta_Q)
        if self._predecessors:
            self.__predecessor_search__(state=curr_state, priority=delta_Q)
        return delta_Q

    def memory_replay(self):
        """
        This function will add the option of trajectory sampling to the parent class's replay function.
        """
        super().memory_replay()
        if self._replay_type == 'trajectory':
            self.__trajectory_sampling__(self._current_state)
        return
