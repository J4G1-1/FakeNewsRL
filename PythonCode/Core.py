#Esta clase funciona para definir los parametros y procesos
#Para entrenar el modelo 

from stable_baselines3 import PPO
import os
from FakeNewsEnv import FakeNewsEnv

from datetime import datetime

##Carpetas para hacer guardar logs para tensorboard y modelos
models_dir = "models/PPO"
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

flags = "11111"
env = FakeNewsEnv(flags,True)

##Estos timesteps serán el número de pasos de una evaluación
##para hacer un reporte en tensorboard
TIMESTEPS_PER_EVALUATION = 40

#Creación de modelo de RL PPO
model = PPO("MlpPolicy",env,verbose = 1, tensorboard_log=logdir, n_steps=TIMESTEPS_PER_EVALUATION)

#Es el numero de evaluación (puntos en tensorboard)
NUMBER_OF_EVALUATIONS = 1000

#Total de pasos por hacer
TIMESTEPS = TIMESTEPS_PER_EVALUATION * NUMBER_OF_EVALUATIONS

#Loop para ir guardando los modelos

#fecha de la creacion del modelo
now = datetime.now() # current date and time
date_time = now.strftime("%m %d %Y, %H:%M:%S")

#COmenzar el entrenamiento del modelo 
model.learn(total_timesteps=TIMESTEPS, tb_log_name=f'PPO - {date_time} - {flags}')

#Guardar el checkpoint del modelo
print("saving the model...")
model.save(f"{models_dir}/brain - {date_time}")