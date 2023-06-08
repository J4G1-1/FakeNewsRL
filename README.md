# FakeNewsRL
Author: Aguilera Valderrama Alexis Fernando

Este programa implementa un agente de reinforcement learning para detectar noticias falsas


Para su uso se tienen que hacer lo siguiente (En linux, ubuntu):

1. Instalar geckodriver PARA FIREFOX en su máquina.

2. En la carpeta principal del proyecto, crear un ambiente virtual con python con el siguiente código:
   python3 -m venv ./FakeNewEnv
  y activarlo con el siguiente comando: source ./FakeNewEnv/bin/activate
  
3. Para hacer la instalación de los paquetes de python correr: pip install -r requirements.txt
 Se tiene que estar en la carpeta principal
 
4. Para usarlo, primero en la carpeta PythonCode debe descomprimir chunk_0-800-reload.zip
5. Para usarlo en modo entrenamiento: ejecute python Core.py (también se puede con python3 ./Core.py)
6. Para usarlo en modo prediccion: ejecute python run-predictor.py (también se puede con python3 ./run-predictor.py)
7. Para usarlo en la investigacion de una noticia en particular: ejecute el archivo research.py

8. (opcional) utilice el comando nice, e introduzca el nombre del modelo a usar y la flag deseada (revise Core.py para mas información sobre esto).

Comando de ejemplo: nice python3 ./Core.py PPO 11101



# Posibles errores en la instalación

 1. A la hora de instalar geckodriver y de correr el programa, se puede obtener una notificación
 de que el firefox profile no fue detectado, este error se da por que o no se tiene firefox instalado o
 se instaló la versión de la tienda snap en vez de la de apt.

 Solución:
 Instalar la versión firefox de apt y no la de la tienda snap
