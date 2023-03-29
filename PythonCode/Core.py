#Esta clase funciona para definir los parametros y procesos
#Para entrenar el modelo 

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C

#importamos sys para leer argumentos desde la terminal
import sys

import os
from FakeNewsEnv import FakeNewsEnv

from datetime import datetime


"""
Al ejecutar este archivo, se pueden recibir 3 argumentos opcionales.

1.- El nombre del modelo a usar {"PPO", "DQN", "A2C"}

2.- La flag a usar, es una cadena de 5 caracteres conformada 
    por 0's y 1's e.g. "11101".
    
    En la flag el caracter en la posición i (las posiciones se cuentan desde
    0 a 4) de la cadena indica si la opcion i en la creación del ambiente
    se activará.

    #opciones de creación del ambiente
    # 0: Similitud al insertar
    # 1: Longitud de sent insertado
    # 2: Descision diff
    # 3: Reward por repetir step
    # 4: Reward por el tamaño de cada lista


3.- El modelo a usar("brain - version.zip"), este deberia estar guardado 
    en la carpeta ./models/{model_name}/ donde model_name in {"PPO", "DQN", "A2C"},
    sino se encuentra, entonces se creara uno nuevo.

"""



try:
    # sys.argv[1] in {"PPO", "DQN", "A2C"}
    model_name = sys.argv[1]
except:
    #entonces se usar por default, "DQN"
    model_name = 'DQN'


try:
    model_name = sys.argv[3]
except:
    #entonces se usar uno por default.
    brain_version  = 'brain - 10 26 2022, 02_48_12.zip'




#ruta para localizar el modelo seleccionado
path = f"./models/{model_name}/{brain_version}"

model_info = {'name': model_name,'path':path}

##Carpetas para hacer guardar logs para tensorboard y modelos
models_dir = f"./models/{model_name}"
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

try:
    flags = str(sys.argv[2])
except:#se usaria por default la flag "10111"
    flags = '10111'


#creacion del ambiente
env = FakeNewsEnv(flags,False, model_name)

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
NUMBER_OF_EVALUATIONS = 12

#Total de pasos por hacer
TIMESTEPS = TIMESTEPS_PER_EVALUATION * NUMBER_OF_EVALUATIONS

#Loop para ir guardando los modelos

#fecha de la creacion del modelo
now = datetime.now() # current date and time
date_time = now.strftime("%m %d %Y, %H %M %S")

#input('Presiona enter para iniciar:')

#Comenzar el entrenamiento del modelo 
log_name = f'{model_name} - {date_time} - {flags}'
model.learn(total_timesteps=TIMESTEPS, tb_log_name=log_name)

#Guardar el checkpoint del modelo
print("saving the model...",f"{models_dir}/brain - {date_time}.zip")
model.save(f"{models_dir}/brain - {date_time}")

env.WriteCurrentLog(log_name)

"""
try:
    
    print(env.reward_for_dates[0],'_'*20)
    print(env.reward_for_dates[1])
except:pass
"""
