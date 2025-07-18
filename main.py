from experiment import *
from plotter import *
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from mplite import TaskManager, Task


def run_with_visuals(exp_params: dict, agent_params: dict, key: list, plot_from: int) -> None:
    """
    Runs a single experiment and visualizes it in a video-like format
    """
    environment = MattarDawMaze()
    agent_params[key[0]][key[1]]['environment'] = environment
    agent_params[key[0]][key[1]]['plot_from'] = plot_from
    agent = None
    if key[0] == 'MF':
        agent = MFagent(**agent_params[key[0]][key[1]])
    elif key[0] == 'MB':
        agent = MBagent(**agent_params[key[0]][key[1]])
    print('Pre-training...')
    run_pre_training(exp_params=exp_params, environment=environment, agent=agent)
    print('\tDone.\nRunning experiment...')
    run_experiment(exp_params=exp_params, environment=environment, agent=agent,
                   replay_threshold=agent_params[key[0]][key[1]]['replay_threshold'])


def pretrain_run(exp_params: dict, environment: Environment, agent: RLagent, replay_threshold=None):
    """
    Runs a single experiment from pre-training to completion, and stores the corresponding data
    """
    run_pre_training(exp_params=exp_params, environment=environment, agent=agent)
    start = time.time()
    reward_rate = run_experiment(exp_params=exp_params, environment=environment, agent=agent,
                                 replay_threshold=replay_threshold)
    run_time = time.time() - start
    return reward_rate, run_time


def run_in_parallel(exp_params: dict, agent_params: dict, keys: list, nr_of_runs: int) -> None:
    """
    Runs nr_of_runs experiments for all agents specified and plots the corresponding reward rates.
    Uses parallel processing to speed it up
    """
    reward_rates = {f'{key[0]}_{key[1]}': [] for key in keys}
    run_times = {f'{key[0]}_{key[1]}': [] for key in keys}
    for key in keys:
        tasks = []
        results = []
        for _ in range(nr_of_runs):
            environment = MattarDawMaze()
            agent_params[key[0]][key[1]]['environment'] = environment
            agent_params[key[0]][key[1]]['plot_from'] = None
            agent = None
            if key[0] == 'MF':
                agent = MFagent(**agent_params[key[0]][key[1]])
            elif key[0] == 'MB':
                agent = MBagent(**agent_params[key[0]][key[1]])
# If you cannot execute code in parallel -------------------------------------------------------------------------------
#             results.append(pretrain_run(exp_params, environment, agent, agent_params[key[0]][key[1]]['replay_threshold']))
# ----------------------------------------------------------------------------------------------------------------------
# If you can execute code in parallel ----------------------------------------------------------------------------------
            ta = Task(pretrain_run,
                      *(exp_params, environment, agent, agent_params[key[0]][key[1]]['replay_threshold']))
            tasks.append(ta)
        with TaskManager() as tm:
            results = tm.execute(tasks)
# ----------------------------------------------------------------------------------------------------------------------
        for res in results:
            reward_rates[f'{key[0]}_{key[1]}'].append(res[0])
            run_times[f'{key[0]}_{key[1]}'].append(res[1])
            print(res)
    for key in keys:
        # print(f'Agent {key[0]} {key[1]}: \t\t\t{np.mean(run_times[f"{key[0]}_{key[1]}"])} sec')
        pass
    ax = plot_reward_rates(reward_rates)
    ax.axhline(y=10 / 14, linewidth=2, color='0.3', ls='--')
    ax.axhline(y=10 / 16, linewidth=2, color='0.3', ls='--')
    plt.show()
    return


def main():
    # The experiment ---------------------------------------------------------------------------------------------------
    exp_params = {
        'nr_of_episodes': 50,  # ------------------- How many episodes we want to model
        'pre_training_steps': 10000,  # -------------- How many steps of pre-training are preformed before learning
        'obstruct_corridor': None  # ---------------- Leftover from the previous TP, used for the TolmanMaze environment
    }

    # The agents -------------------------------------------------------------------------------------------------------
    agent_params = {'MF': {}, 'MB': {}}
    agent_params['MF']['classic'] = {
        'gamma': 0.9,  # ---------------------------- The discount factor
        'epsilon': 0.05,  # ------------------------- For the epsilon-greedy action selection
        'alpha': 0.3,  # ---------------------------- The learning rate of the MF agent
        'replay_type': None,  # --------------------- The type of replay to be used
        'max_replay': 50,  # ------------------------ The number of replay staps allowed
        'replay_threshold': 0.001  # ---------------- The minimum change in Q-values eliciting a replay event
    }
    agent_params['MF']['random'] = agent_params['MF']['classic'].copy()
    agent_params['MF']['random']['replay_type'] = 'random'

    agent_params['MF']['forward'] = agent_params['MF']['classic'].copy()
    agent_params['MF']['forward']['replay_type'] = 'forward'

    agent_params['MF']['backward'] = agent_params['MF']['classic'].copy()
    agent_params['MF']['backward']['replay_type'] = 'backward'

    agent_params['MF']['prioritized'] = agent_params['MF']['classic'].copy()
    agent_params['MF']['prioritized']['replay_type'] = 'prioritized'

    agent_params['MB']['classic'] = {
        'gamma': 0.9,  # ---------------------------- The discount factor
        'epsilon': 0.05,  # ------------------------- For the epsilon-greedy action selection
        'theta': 0.001,  # -------------------------- The threshold of the MB agent
        'window_length': 10,  # --------------------- The window for the model learning (MB)
        'replay_type': None,  # --------------------- The type of replay to be used
        'max_replay': 50,  # ------------------------ The number of replay staps allowed
        'replay_threshold': 0.001  # ---------------- The minimum change in Q-values eliciting a replay event
    }


    agent_params['MB']['bidirectional'] = {
        'gamma': 0.9,  # ---------------------------- The discount factor
        'epsilon': 0.05,  # ------------------------- For the epsilon-greedy action selection
        'theta': 0.001,  # -------------------------- The threshold of the MB agent
        'window_length': 10,  # --------------------- The window for the model learning (MB)
        'replay_type': None,  # --------------------- The type of replay to be used
        'max_replay': 100,  # ------------------------ The number of replay staps allowed
        'replay_threshold': 0.0005  # ---------------- The minimum change in Q-values eliciting a replay even 
    }

    agent_params['MB']['random'] = agent_params['MB']['classic'].copy()
    agent_params['MB']['random']['replay_type'] = 'random'

    agent_params['MB']['forward'] = agent_params['MB']['classic'].copy()
    agent_params['MB']['forward']['replay_type'] = 'forward'

    agent_params['MB']['backward'] = agent_params['MB']['classic'].copy()
    agent_params['MB']['backward']['replay_type'] = 'backward'

    agent_params['MB']['prioritized'] = agent_params['MB']['classic'].copy()
    agent_params['MB']['prioritized']['replay_type'] = 'prioritized'

    agent_params['MB']['predecessor'] = agent_params['MB']['classic'].copy()
    agent_params['MB']['predecessor']['replay_type'] = 'prioritized'
    agent_params['MB']['predecessor']['predecessors'] = True

    agent_params['MB']['trajectory'] = agent_params['MB']['classic'].copy()
    agent_params['MB']['trajectory']['replay_type'] = 'trajectory'

    agent_params['MB']['bidirectional'] = agent_params['MB']['classic'].copy()
    agent_params['MB']['bidirectional']['replay_type']     = 'bidirectional'
    




    # ------------------------------------------------------------------------------------------------------------------
    # # Task 1 -----------------------------------------------------------------------------------------------------------
    # agents_to_run = [
    #     ['MF', 'classic'],
    #     ['MB', 'classic']
    # ]
    # run_in_parallel(exp_params=exp_params, agent_params=agent_params, keys=agents_to_run, nr_of_runs=30)
    # # run_with_visuals(exp_params=exp_params, agent_params=agent_params, key=['MF', 'classic'], plot_from=1)
    # # run_with_visuals(exp_params=exp_params, agent_params=agent_params, key=['MB', 'classic'], plot_from=1)

    # # Task 2 -----------------------------------------------------------------------------------------------------------
    # agents_to_run = [
    #     ['MF', 'classic'],
    #     ['MB', 'classic'],
    #     ['MF', 'random'],
    #     ['MF', 'forward'],
    #     ['MF', 'backward']
    # ]
    # # run_in_parallel(exp_params=exp_params, agent_params=agent_params, keys=agents_to_run, nr_of_runs=30)
    # # run_with_visuals(exp_params=exp_params, agent_params=agent_params, key=['MF', 'random'], plot_from=1)
    # # run_with_visuals(exp_params=exp_params, agent_params=agent_params, key=['MF', 'forward'], plot_from=1)
    # # run_with_visuals(exp_params=exp_params, agent_params=agent_params, key=['MF', 'backward'], plot_from=1)

    # # Task 3 -----------------------------------------------------------------------------------------------------------
    # agents_to_run = [
    #     ['MF', 'classic'],
    #     ['MF', 'random'],
    #     ['MF', 'forward'],
    #     ['MF', 'backward'],
    #     ['MF', 'prioritized']
    # ]
    # # run_in_parallel(exp_params=exp_params, agent_params=agent_params, keys=agents_to_run, nr_of_runs=30)
    # # run_with_visuals(exp_params=exp_params, agent_params=agent_params, key=['MF', 'prioritized'], plot_from=1)

    # # Task 4 -----------------------------------------------------------------------------------------------------------
    # agents_to_run = [
    #     ['MB', 'classic'],
    #     ['MB', 'random'],
    #     ['MB', 'forward'],
    #     ['MB', 'backward'],
    #     ['MB', 'prioritized']
    # ]
    # # run_in_parallel(exp_params=exp_params, agent_params=agent_params, keys=agents_to_run, nr_of_runs=30)

    # Task 5 -----------------------------------------------------------------------------------------------------------
    agents_to_run = [
        # ['MF', 'classic'],
        # ['MB', 'classic'],
        # ['MF', 'prioritized'],
        # ['MB', 'prioritized'],
        # ['MB', 'predecessor'],
        ['MB', 'bidirectional']
    ]
    # run_in_parallel(exp_params=exp_params, agent_params=agent_params, keys=agents_to_run, nr_of_runs=30)
    run_with_visuals(exp_params=exp_params, agent_params=agent_params, key=['MB', 'bidirectional'], plot_from=1)
    # run_with_visuals(exp_params=exp_params, agent_params=agent_params, key=['MB', 'predecessor'], plot_from=1)

    # # Bonus task -------------------------------------------------------------------------------------------------------
    # agents_to_run = [
    #     ['MB', 'classic'],
    #     ['MB', 'forward'],
    #     ['MB', 'trajectory']
    # ]
    # run_in_parallel(exp_params=exp_params, agent_params=agent_params, keys=agents_to_run, nr_of_runs=30)
    # run_with_visuals(exp_params=exp_params, agent_params=agent_params, key=['MB', 'forward'], plot_from=20)
    # run_with_visuals(exp_params=exp_params, agent_params=agent_params, key=['MB', 'trajectory'], plot_from=20)

if __name__ == '__main__':
    main()
