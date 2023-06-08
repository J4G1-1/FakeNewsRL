from stable_baselines3 import PPO


import os
from FakeNewsEnv import FakeNewsEnv
import pandas as pd
from datetime import datetime

new_csv = False

if not os.path.exists('./'):
			os.makedirs('Results_Investigation')


def investigar():
	titulo = input('introduce el titulo de la noticia a investigar: ')
	news = {'title':titulo, 'label':-1}
	data_frame = pd.DataFrame([news])
	data_frame.to_csv('new_frame.csv')
	new_csv = True
	model_name = 'PPO'
	brain_name = 'brain - 05 20 2023, 00 57 10'
	
	brain_checkpoint = 'checkpoint 400000 - 500000'
	brain_replay = 'replay 400000 - 500000'
		
	eval_outcome_logs = f"./Results_Investigation/{model_name}/{brain_name}/outcome_logs/"
	
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
		model_name = model_info['model_name'], local_data_path='./new_frame.csv')

	brain_path = f"./models/{model_info['model_name']}/{model_info['brain_name']}.zip"
	model = PPO.load(brain_path,vec_env)

	obs = vec_env.reset()
	counter = 0

	while True:
		action, _states = model.predict(obs, deterministic=False)
		obs, rewards, dones, info = vec_env.step(action)
		
		if dones:
			counter = counter + 1
			vec_env.reset()

		if counter == 1:
			break
	#fecha de la creacion del modelo
	now = datetime.now() # current date and time
	date_time = now.strftime("%m %d %Y, %H-%M-%S")
	vec_env.saveOutcomelog(path_name_binary=model_info['eval_outcome_logs']+'/bin/'+date_time,
						   path_name_plain=model_info['eval_outcome_logs']+'/plain/'+date_time)

	final_outcome = vec_env.outcome_log[0]
	print('title: ',final_outcome['title'])
	print('agree_list: ',final_outcome['agree_list'])
	print('\n')
	print('disagree_list: ',final_outcome['disagree_list'])
	print('outcome: ',final_outcome['outcome'])
	print('certainty: ',final_outcome['certainty'])
	print('\n')
	
	print('si outcome: 1, entonces se concluyo que la notica es verdadera')
	print('si outcome: 0, entonces se concluyo que la notica es falsa')
	
	print(f"""
	    _______________________________________________
		Los datos de la investigacion se guardaron en
		PythonCode/Results_Investigation/{model_name}/{brain_name}/outcome_logs/plain
		________________________________________________
		""")
	vec_env.cleanOutcomeLog()
	vec_env.close()

if input('desea investigar una noticia:[s/n] ') in ['s','S']:
	investigar()



if new_csv:
	os.system(f'rm new_frame.csv')
else:pass
