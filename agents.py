from environments import *
import random
from plotter import *
from collections import deque
import copy
import copy


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
        :kwargs replay_type: "random", "forward", "backward", "prioritized" or "trajectory", "bidirectional"
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
        self._buffer_size = kwargs.get('max_replay', None)
        self._memory_buffer = deque(maxlen=self._buffer_size)
        self._replay_type = kwargs.get('replay_type', None)
        if self._replay_type not in [None, 'random', 'forward', 'backward', 'prioritized', 'trajectory', 'bidirectional']:
            raise ValueError(f'{self._replay_type} is not a valid replay type.')
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
    def __store_event__(self, state: int, action: str, reward: float, arrival_state: int, priority: float | None) -> None:
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

        # 2) Otherwise, if it is not in the buffer, let's add it to the front
        event = Event(state=state, action=action, reward=reward, arrival_state=arrival_state, priority=abs(priority))
        self._memory_buffer.appendleft(event)

        # 2a) Or if we are prioritized, it will not actually be the front
        # deque will automatically remove the oldest element if the buffer is full
        if self._replay_type == 'prioritized':
            sorted_events = sorted(self._memory_buffer, key=lambda e: e.priority, reverse=True)
            self._memory_buffer = deque(sorted_events, maxlen=self._buffer_size)

    def __classical_replay__(self):
        """
        Implements the classical (random, forward, backwards) replay algorithms
        """
        if self._plotter is not None:
            self._plotter.clear()

        # 1) Prepare the buffer that we are iterating through
        memory_buffer = copy.deepcopy(self._memory_buffer)
        memory_buffer = copy.deepcopy(self._memory_buffer)
        if self._replay_type == 'forward':
            memory_buffer.reverse()
        elif self._replay_type == 'random':
            tmp = list(memory_buffer)
            tmp = list(memory_buffer)
            random.shuffle(tmp)
            tmp)
            memory_buffer = deque(tmp, maxlen=self._buffer_size = deque(tmp, maxlen=self._buffer_size)

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
        for _ in range(self._buffer_size):
            event = self._memory_buffer.popleft()
            self.reinforcement_learning(state=event.state, action=event.action, reward=event.reward,
                                                  arrival_state=event.arrival_state)
            if not self._memory_buffer:
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
        return
    
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
        for _ in range(self._buffer_size):
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
        #super().memory_replay()
        #if self._replay_type == 'trajectory':

        super().memory_replay()
        if self._replay_type == 'bidirectional':
            self.__bidirectional_search__()
        elif self._replay_type == 'trajectory':
            self.__trajectory_sampling__(self._current_state)
        return


# DONE: Modify the predecessor search to adapt to the bidirectional search
# DONE: implment the _softmax_policy method for the MBagent_BD class
#DONE:  add beta_offline parameter to the constructor of the MBagent_BD class
# DONE:  add _maxLoops parameter to the constructor of the MBagent_BD class
# DONE:  add _budget_ps parameter to the constructor of the MBagent_BD class
# DONE:  add _budget_ts parameter to the constructor of the MBagent_BD class

class MBagent_BD(RLagent):
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
        self._budget_ps = kwargs.get('budget_ps', None) # budget for prioritized sweeping
        self._budget_ts = kwargs.get('budget_ts', None) # budget for trajectory sampling
        self._maxLoops = kwargs.get('maxLoops', None) # budget for trajectory sampling
        self._beta_offline = kwargs.get('beta_offline', None) # budget for trajectory sampling
        self._beta_online = kwargs.get('beta_online', None) # budget for trajectory sampling

        if self._predecessors and self._replay_type != 'prioritized':
            raise ValueError('Predecessor search only makes sense while using prioritized sweeping')
    
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
        for _ in range(self._buffer_size):
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
        priority = delta_Q
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

                    if abs(predecessor_priority) > self._replay_threshold:
                        self.__store_event__(state=predecessor, action=action,
                                         reward=self._reward_fun[predecessor][action],
                                         arrival_state=state, priority=predecessor_priority)

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
        # for bidirectional search, we need to use the softmax policy
        if self._replay_type == 'bidirectional':
            action = self._softmax_policy(state=self._current_state, beta=self._beta_online)

        if curr_state is None:
            curr_state = self._current_state
        super().reinforcement_learning(action=action, arrival_state=arrival_state, reward=reward, state=state,
                                       terminated=terminated)
        if state is None:  # Meaning that it is a real step
            self.__update_model__(state=curr_state, action=action, arrival_state=arrival_state, reward=reward)


        # obtain RPE and update the Q-values
        if self._replay_type is None:
            delta_Q = self.__value_iteration__()
        else:
            delta_Q = self.__one_step___value_iteration____(state=curr_state, action=action)

        # check if we need to store the event in the priority queue
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
        #super().memory_replay()
        #if self._replay_type == 'trajectory':

        super().memory_replay()
        if self._replay_type == 'bidirectional':
            self.__bidirectional_search__()
        elif self._replay_type == 'trajectory':
            self.__trajectory_sampling__(self._current_state)
        return


    def _softmax_policy(self, state:int, beta=10):

        if self._beta_offline is not None:
            beta = self._beta_offline
         
        # Q_values
        all_actions = self._Q[state]
        exp_Q = [np.exp(beta * self._Q[state][action]) for action in all_actions]

        # softmax probabilities
        probabilities = {action: exp_Q[i] / sum(exp_Q) for i, action in enumerate(all_actions)}

        # action selection
        action = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))

        return action

        
    def __bidirectional_search__(self):

        '''
        bidirectional search 
        
        prioritized sweeping with predecessors, followed by trajectory sampling

        '''

        nbLoops = 0
        while nbLoops < self._maxLoops:
            sum_RPE = 0
            nbLoops += 1
            nbPS = 0 # number of prioritized sweeping steps

            # bugeted prioritized sweeping
            while nbPS < self._budget_ps and self._memory_buffer[0].priority > self._replay_threshold:
                event = self._memory_buffer[0]
                state, action  = event.state, event.action
                arrival_state = np.random.choice(list(self._Q.keys()), p=self._transition_function[state][action])
                reward = self._reward_fun[state][action]

                # one-step MB update
                delta_Q = abs(self.__one_step___value_iteration____(state=state, action=action))
                sum_RPE += delta_Q

                # update the priority
                self.__store_event__(state=state, action=action, reward=reward, arrival_state=arrival_state,
                                  priority=delta_Q)

                # prioritized sweeping
                self.__predecessor_search__(state=state, priority=delta_Q)

            # budgeted trajectory sampling
            nbTS = 0
            state_in_buffer = False
            mental_curr_state = self._current_state
            while nbTS < self._budget_ts and not state_in_buffer:

                # check if the current_state in real world is in the memory buffer
                state_in_buffer = any(event.state == mental_curr_state for event in self._memory_buffer)

                # draw action from the softmax policy
                action = self._softmax_policy(mental_curr_state, beta=self._beta_offline)
                # draw arrival state from the transition function
                arrival_state = int(np.random.choice(list(self._Q.keys()), p=self._transition_function[mental_curr_state][action]))

                delta_Q = abs(self.__one_step___value_iteration____(state=mental_curr_state, action=action))
                sum_RPE += delta_Q
                mental_curr_state = arrival_state

            if sum_RPE < self._theta:
                break # break if the sum of RPE is below the threshold
                
                
                

                

                



