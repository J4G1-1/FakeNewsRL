#Esta clase funciona para definir los parametros y procesos
#Para entrenar el modelo 

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C


import os
from FakeNewsEnv import FakeNewsEnv

from datetime import datetime


model_name = 'A2C'
#si usas windows cambia el formato de 11:13:10 y escribe
#en su lugar 11_13_10
brain_version = 'brain - 10 15 2022, 11:13:10.zip'
path = f"./models/{model_name}/{brain_version}"
model_info = {'name': model_name,'path':None}

##Carpetas para hacer guardar logs para tensorboard y modelos
models_dir = f"models/{model_name}"
logdir = "logs"
logcustom = "logscustom"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(logcustom):
    os.makedirs(logcustom)

#Creacion del ambiente
# 0: Similitud al insertar
# 1: Longitud de sent insertado
# 2: Descision diff
# 3: Reward por repetir step
# 4: Reward por el tamaño de cada lista

flags = '11111'
env = FakeNewsEnv(flags,True,model_name)

#Creación de modelo de RL PPO
if model_info['name'] == 'PPO':
    try:
        model = PPO.load(model_info['path'],env)
        print(f'Modelo {brain_version} PPO se cargo')
    except:
        model = PPO("MlpPolicy",env,verbose = 1, tensorboard_log=logdir)
        print(f'Modelo PPO nuevo Creado')

elif model_info['name'] == 'DQN':
    try:
        model = DQN.load(model_info['path'],env)
        print(f'Modelo {brain_version} DQN se cargo')
    except:
        model = DQN("MlpPolicy",env,verbose = 1, tensorboard_log=logdir)
        print(f'Modelo DQN nuevo Creado')

elif model_info['name'] == 'A2C':
    try:
        model = A2C.load(model_info['path'],env)
        print(f'Modelo {brain_version} A2C se cargo')
    except:
        model = A2C("MlpPolicy",env,verbose = 1, tensorboard_log=logdir)
        print(f'Modelo A2C nuevo Creado')


##Estos timesteps serán el número de pasos de una evaluación
##para hacer un reporte en tensorboard
TIMESTEPS_PER_EVALUATION = 100


#Es el numero de evaluación (puntos en tensorboard)
NUMBER_OF_EVALUATIONS = 5500

#Total de pasos por hacer
TIMESTEPS = TIMESTEPS_PER_EVALUATION * NUMBER_OF_EVALUATIONS

#Loop para ir guardando los modelos

#fecha de la creacion del modelo
now = datetime.now() # current date and time
date_time = now.strftime("%m %d %Y, %H:%M:%S")

#input('Presiona enter para iniciar entrenamiento')

#COmenzar el entrenamiento del modelo 
log_name = f'{model_name} - {date_time} - {flags}'
model.learn(total_timesteps=TIMESTEPS, tb_log_name=log_name)

#Guardar el checkpoint del modelo
print("saving the model...")
model.save(f"{models_dir}/brain - {date_time}")

env.WriteCurrentLog(log_name)
