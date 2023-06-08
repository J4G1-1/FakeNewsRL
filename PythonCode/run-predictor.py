from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C


import os
from FakeNewsEnv import FakeNewsEnv

from datetime import datetime

model_name = 'PPO'

brain_name = 'brain - 05 20 2023, 00 57 10'

brain_checkpoint = 'checkpoint 400000 - 500000'
brain_replay = 'replay 400000 - 500000'


eval_outcome_logs = f"./models/{model_name}/{brain_name}/outcome_logs/eval/"

model_info = {'model_name': model_name,
              'brain_name' : brain_name,
            'brain_checkpoint' : 'checkpoint 400 - 500',
            'flags' : "11110",
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
we send the agent to investigate 20 randomly selected news that the agent does not know in advance
"""

obs = vec_env.reset()
counter = 0

while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    
    if dones:
        counter = counter + 1
        vec_env.reset()

    if counter == 20:
        break

#fecha de la creacion del modelo
now = datetime.now() # current date and time
date_time = now.strftime("%m %d %Y, %H-%M-%S")

vec_env.saveOutcomelog(path_name_binary=model_info['eval_outcome_logs']+'/bin/'+date_time,
                       path_name_plain=model_info['eval_outcome_logs']+'/plain/'+date_time)
