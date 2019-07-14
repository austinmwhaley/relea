import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import pandas as pd
from tqdm import tqdm #progress bars

def relea_train(model='PPO2', env='CartPole-v1', total_timesteps=1000):
    '''
    Instantiates task between model and environment. Returns fit model and env
    '''
    # Environment Selection
    if env == 'CartPole-v1':
        env = gym.make('CartPole-v1')
    else:
        print('Please input valid environment string')

    # Post Environment Selection
    env = DummyVecEnv([lambda: env])

    # Model Selection
    if model.lower() == 'ppo2':
        model = PPO2(MlpPolicy, env, verbose=0)
    else:
        print('Please input valid model string')

    # Model Training
    model.learn(total_timesteps=total_timesteps)

    return model, env


def relea_run(model, env):
    '''
    Application of relea_train
    '''
    df = pd.DataFrame() #empty dataframe for results
    obs = env.reset() #reset env for start
    done, i = False, 0
    while done == False:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        step = [
            i, #step number
            reward[0], #step reward
            done[0], #step done status
            info[0] #step info
        ]

        # add environment observation information (from array)
        for o in obs[0]:
            step.append(o)

        df = pd.concat([df, pd.DataFrame([step])]) # add step data to running df
        done = done[0] # reset done status
        i+=1 #increment index counter
        # env.render() #optional render

    return df #of logging info


def relea_log(results, env_name):

    # LOGGING FINALIZATION - POST RUN
    if env_name == 'CartPole-v1':
        columns = [
            'step_num',
            'reward',
            'done',
            'info',
            'cart_position',
            'cart_velocity',
            'pole_angle',
            'pole_velocity_at_tip'
        ]

        results.columns = columns

        return results

# https://gym.openai.com/envs/CartPole-v1/
model, env = relea_train(model='PPO2', env='CartPole-v1', total_timesteps=100000)
r = []
for i in tqdm(range(1000)):
    results = relea_run(model=model, env=env)
    results = relea_log(results=results, env_name='CartPole-v1')
    r.append(len(results))

max(r)
