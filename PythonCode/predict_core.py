from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C


import os
from FakeNewsEnv import FakeNewsEnv

from datetime import datetime

model_name = 'PPO'
#brain_name = 'brain - 03 29 2023, 17 44 59'
#brain_name = 'brain - 03 29 2023, 19 17 48'
#brain_name = 'brain - 04 11 2023, 16 23 31'
#brain_name = 'brain - 04 27 2023, 16 48 10'
#brain_name = 'brain - 05 03 2023, 15 14 32'
#brain_name = 'brain - 05 01 2023, 13 18 59'
#brain_name = 'brain - 05 03 2023, 22 21 48'

#mejor brain
#brain_name = 'brain - 05 05 2023, 10 07 41'
brain_name = 'brain - 05 18 2023, 00 24 44'

#2do mejor brain
#brain_name = 'brain - 05 08 2023, 17 57 19'

brain_checkpoint = 'checkpoint 400000 - 500000'
brain_replay = 'replay 400000 - 500000'



eval_outcome_logs = f"./models/{model_name}/{brain_name}/outcome_logs/eval/"

model_info = {'model_name': model_name,
              'brain_name' : brain_name,
            'brain_checkpoint' : 'checkpoint 400 - 500',
            'flags' : "00100",
            'brain_replay' : 'replay 400 - 500',
            'eval_outcome_logs' : eval_outcome_logs}

if not os.path.exists(model_info['eval_outcome_logs'] + '/bin/'):
    os.makedirs(model_info['eval_outcome_logs'] + 'bin/')

if not os.path.exists(model_info['eval_outcome_logs'] + '/plain/'):
    os.makedirs(model_info['eval_outcome_logs'] + 'plain/')

flags = model_info['flags']
vec_env = FakeNewsEnv(flags,train_mode = False,
    model_name = model_info['model_name'], local_data_path='./data/set_test.csv')

brain_path = f"./models/{model_info['model_name']}/{model_info['brain_name']}.zip"
model = PPO.load(brain_path,vec_env)


"""
replay_path = f"./models/{model_info['model_name']}/{model_info['brain_name']}/{model_info['brain_replay']}.pkl"
model.load_replay_buffer(replay_path)
"""


obs = vec_env.reset()
counter = 0

while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    
    if dones:
        counter = counter + 1
        vec_env.reset()

    if counter == 40:
        break

#fecha de la creacion del modelo
now = datetime.now() # current date and time
date_time = now.strftime("%m %d %Y, %H-%M-%S")

vec_env.saveOutcomelog(path_name_binary=model_info['eval_outcome_logs']+'/bin/'+date_time,
                       path_name_plain=model_info['eval_outcome_logs']+'/plain/'+date_time)
