from experiment import *
from plotter import *
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from mplite import TaskManager, Task
from multiprocessing import Pool, TimeoutError
import traceback
import os


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
    elif key[0] == 'BD':
        agent = MBagent_BD(**agent_params[key[0]][key[1]])
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
            agent = create_agent(key, environment, agent_params)

# If you cannot execute code in parallel -------------------------------------------------------------------------------
            # results.append(pretrain_run(exp_params, environment, agent, agent_params[key[0]][key[1]]['replay_threshold']))
# ----------------------------------------------------------------------------------------------------------------------
# If you can execute code in parallel ----------------------------------------------------------------------------------
        #     ta = Task(pretrain_run,
        #               *(exp_params, environment, agent, agent_params[key[0]][key[1]]['replay_threshold']))
        #     tasks.append(ta)
        # with TaskManager() as tm:
        #     results = tm.execute(tasks)
# ----------------------------------------------------------------------------------------------------------------------
# alternative parallel execution using multiprocessing ---------------------------------------------------------------

            args = (exp_params, environment, agent, agent_params[key[0]][key[1]]['replay_threshold'])
            tasks.append(args)
        with Pool(processes=min(8, os.cpu_count()//2)) as pool:  # Safer worker count
            results = []
            # Submit ALL tasks first
            async_results = [
                pool.apply_async(safe_pretrain_run, task_args)
                for task_args in tasks
            ]
            
            # Collect results AS THEY COMPLETE
            for async_res in tqdm(async_results, 
                                desc=f'Running {key[0]} {key[1]}',
                                total=len(tasks)):
                try:
                    # Timeout per task = 10 minutes
                    res = async_res.get(timeout=600)
                    results.append(res)
                except TimeoutError:
                    print(f'Timeout for task')
                except Exception as e:
                    print(f'Error: {e}')
        
        # 3. Store results
        for res in results:
            if res:  # Skip failed tasks
                reward_rates[f"{key[0]}_{key[1]}"].append(res[0])
                run_times[f"{key[0]}_{key[1]}"].append(res[1])


    for key in keys:
        # print(f'Agent {key[0]} {key[1]}: \t\t\t{np.mean(run_times[f"{key[0]}_{key[1]}"])} sec')
        pass
    ax = plot_reward_rates(reward_rates)
    ax.axhline(y=10 / 14, linewidth=2, color='0.3', ls='--')
    ax.axhline(y=10 / 16, linewidth=2, color='0.3', ls='--')
    plt.show()

def safe_pretrain_run(exp_params, env, agent, threshold):
    try:
        return pretrain_run(exp_params, env, agent, threshold)
    except Exception as e:
        traceback.print_exc()
        return None

# Helper function for agent creation
def create_agent(key, env, agent_params):
    params = agent_params[key[0]][key[1]].copy()
    params.update(environment=env, plot_from=None)
    
    if key[0] == 'MF':
        return MFagent(**params)
    elif key[0] == 'MB':
        return MBagent(**params)
    elif key[0] == 'BD':
        return MBagent_BD(**params)

def main():
    # The experiment ---------------------------------------------------------------------------------------------------
    exp_params = {
        'nr_of_episodes': 50,  # ------------------- How many episodes we want to model
        'pre_training_steps': 10000,  # -------------- How many steps of pre-training are preformed before learning
        'obstruct_corridor': None  # ---------------- Leftover from the previous TP, used for the TolmanMaze environment
    }

    # The agents -------------------------------------------------------------------------------------------------------
    agent_params = {'MF': {}, 'MB': {}, 'BD':{}}
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


    agent_params['BD']['bidirectional'] = {
        'gamma': 0.9,  # ---------------------------- The discount factor
        'epsilon': 0.05,  # ------------------------- For the epsilon-greedy action selection
        'theta': 0.1,  # -------------------------- The threshold of the MB agent
        'window_length': 10,  # --------------------- The window for the model learning (MB)
        'replay_type': 'bidirectional',  # --------------------- The type of replay to be used
        'max_replay': 100,  # ------------------------ The number of replay staps allowed
        'replay_threshold': 0.0001,  # ---------------- The minimum change in Q-values eliciting a replay event
        'maxLoops': 10,  # ------------------ The maximum number of loops to be performed in the bidirectional replay
        'budget_ps': 10, # ------------------- The budget for the predecessor sampling
        'budget_ts': 10, # ----------------- The budget for the trajectory sampling
        'beta_offline': 10, # offline exploration/exploitation trade-off
        'beta_online': 20, # online exploration/exploitation trade-off
    }

    agent_params['BD']['random'] = agent_params['MB']['classic'].copy()
    agent_params['BD']['random']['replay_type'] = 'random'

    agent_params['BD']['forward'] = agent_params['MB']['classic'].copy()
    agent_params['BD']['forward']['replay_type'] = 'forward'

    agent_params['BD']['backward'] = agent_params['MB']['classic'].copy()
    agent_params['BD']['backward']['replay_type'] = 'backward'

    agent_params['BD']['prioritized'] = agent_params['MB']['classic'].copy()
    agent_params['BD']['prioritized']['replay_type'] = 'prioritized'

    agent_params['BD']['predecessor'] = agent_params['MB']['classic'].copy()
    agent_params['BD']['predecessor']['replay_type'] = 'prioritized'
    agent_params['BD']['predecessor']['predecessors'] = True

    agent_params['BD']['trajectory'] = agent_params['MB']['classic'].copy()
    agent_params['BD']['trajectory']['replay_type'] = None

    agent_params['BD']['bidirectional']['replay_type']     = 'bidirectional'
    




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
        ['BD', 'bidirectional']
    ]
    run_in_parallel(exp_params=exp_params, agent_params=agent_params, keys=agents_to_run, nr_of_runs=30)
    # run_with_visuals(exp_params=exp_params, agent_params=agent_params, key=['BD', 'bidirectional'], plot_from=1)
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
