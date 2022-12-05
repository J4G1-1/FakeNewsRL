from cgitb import reset

import gym
from gym import spaces

import numpy as np
import pandas as pd

import spacy
from scipy import stats
from math import e

import os
from LocalDataManager import LocalDataManager
from WebDataManager import WebDataManager

from WebScrapper import WebScrapper
from ArgumentList import ArgumentList

#La clase FakeNewsEnv es una clase que implementa un gym environment
#Su objetivo es definir 

class FakeNewsEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self,flags, train_mode = False, model_name = ''):
    super(FakeNewsEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:

    self.action_counter=[0,0,0,0]
    self.flags = flags
    self.model_name = model_name

    #Crear carpetas de logs escritos por el programador
    self.logcustom = "logscustom"
    if not os.path.exists(self.logcustom):
      os.makedirs(self.logcustom)
    
    self.log = ''

    #Actions
    #El agente tiene 4 acciones:
    #0.Ignorar la sent actual
    #1.Agregar el sent a la lista de agree
    #2.Agregar el sent a la lista de disagree
    #3. Ir a la siguiente fuente de informacion
    self.action_space = spaces.Discrete(4)

    #Definiendo el espacio de 
    #Al usar spacy en_core_web_md los embbedings tiene una longitud de
    #300 cada uno ya que se toma en cuenta lo siguiente:
    #[0,299]: La suma de los vectores de todos los sents contenidos en agreelist
    #[300,599]: La suma de los vectores de todos los sents contenidos en disagreelist
    #[600,899]: El vector del sent actual (puede ser una lista)
    self.observation_space = spaces.Box(low=-100.0, high=100.0,
                                        shape=(900,), dtype=np.float64)

    #Carga de spacy para sent analysis y embeddings
    self.nlp = spacy.load("en_core_web_md")

    #Creacion del dirver para webscrapping
    if train_mode:
      self.dataManager = LocalDataManager(localdata_path='../chunk_0-600')
    else:
      self.dataManager = WebDataManager(r"./data/DataFakeNews.csv")

    #Creacion de la estrucutra de datos de listas
    self.argumentLists = ArgumentList(self.nlp)
    
    #Contabilizar el total_reward
    self.total_reward = 0

    #-----------------------------------------
    self.GaussianFactorList = []
    mu = 0
    std = 4
    scale = 25
    snd = stats.norm(mu, std)

    counter = 0
    while True:
      number = scale*snd.pdf(counter)
      counter = counter + 1
      if number > 0.1:
        self.GaussianFactorList.append(number)
      else:
        break
    
    self.counter_good = 0
    self.counter_wrong = 0
  
  def WriteCurrentLog(self,logname):
    f = open(f'{self.logcustom}/{logname}', "a")
    f.write(self.log)
    f.close()
  
  def sigmodialFunction(self,value, factor, estrecho, desx, desy):
    return factor/(1+pow(e+estrecho,-(value-desx))) + desy
    
  def GetSents(self,text):
    try:
      doc = self.nlp(text)
      return [sentence for sentence in doc.sents]
    except:
      print("Hubo problemas al parsear el html")
      return []

  def BuildObservation(self):
    #Grab the first sentence
    current_sent = self.argumentLists.GetCurrentSent()

    accumulatedAgree = self.argumentLists.GetAccumalatedAgree()
    accumulatedDisagree = self.argumentLists.GetAccumalatedDisagree()

    observation = np.concatenate((accumulatedAgree,
                                  accumulatedDisagree,
                                  current_sent.vector),
                                  axis=None)

    return observation

  #El método toma encuenta los valores producidos por la funcion gaussiana para calcular el reward
  # En pocas palabras el reward funciona de la siguiente manera:
  # Conforme se van agregando elementos al agree o al disagree list, la recompensa por
  # agregar más elementos es menor
  def SimilarityReward(self, simility_list):
    GaussianFactor = 0
    reward = 0

    if len(simility_list) >= len(self.GaussianFactorList):
      GaussianFactor = self.GaussianFactorList[len(self.GaussianFactorList)-1]
    else:
      GaussianFactor = self.GaussianFactorList[len(simility_list)]

    for simility in simility_list:
      simility_reward = (16*simility - 8)
      reward = GaussianFactor*simility_reward + reward
    
    return reward


  def step(self, action):

    self.action_counter[action] = self.action_counter[action] + 1

    isFinished = False
    reward = 0

    rew_simility = 0
    rew_len = 0
    reward_factor = 0
    reward_diff_step = 0
    reward_len_agree = 0
    reward_len_disagree = 0

    ##Ignore sent
    if action==0:
      #Pasa a la siguiente sent
      more = self.argumentLists.GoToNextSent()

      if more is False:
        reward = -0.0
      else:
        reward = 0.0


    ##Add to affirm
    if action==1:
      present_sent = self.argumentLists.GetCurrentSent()

      more = self.argumentLists.GoToNextSent()
      #Si ya no hay sents se da una rencompensa negativa
      if more is False:
        reward = -0.0
      else:

        simility_list = self.argumentLists.append_agree_list(present_sent)

        ##Calculo de similitud
        if self.flags[0] == '1':
          rew_simility = self.SimilarityReward(simility_list)

        ##Calculo de longitud
        if self.flags[1] == '1':
          rew_len = self.sigmodialFunction(len(present_sent),7,-1.3,13,-3.5)

        reward = reward + rew_len + rew_simility

    ##Add to objection
    if action==2:
      ##Calculo de similitud
      present_sent = self.argumentLists.GetCurrentSent()

      more = self.argumentLists.GoToNextSent()
      #Si ya no hay sents se da una rencompensa negativa
      if more is False:
        reward = -0.0
      else:
        
        simility_list = self.argumentLists.append_disagree_list(present_sent)
        ##Calculo de similitud
        if self.flags[0] == '1':
          rew_simility = self.SimilarityReward(simility_list)

        ##Calculo de longitud
        if self.flags[1] == '1':
          rew_len = self.sigmodialFunction(len(present_sent),7,-1.3,13,-3.5)

        reward = reward + rew_len + rew_simility


    ##Go to the next url
    if action==3:

      status = self.dataManager.GoNextArticle()

      if not status:
        isFinished = True

      if self.flags[2] == '1':
        decision = self.argumentLists.GetDecision()
        agreelist = self.argumentLists.getAgreeList()
        disagreelist = self.argumentLists.getDisagreeList()

        diff = abs(len(agreelist) - len(disagreelist))
        
        #Difference in list reward
        reward_factor = self.sigmodialFunction(diff,3.4,-0.5,4.4,0)

        if not status:

          if decision == -1:
            reward = 0 + reward
          elif self.label == decision:
            self.counter_good = self.counter_good + 1
            reward = reward_factor + reward
          else:
            self.counter_wrong = self.counter_wrong + 1
            reward = -reward_factor + reward
        
        else:
          reward_factor = 0.05*reward_factor
          if decision == -1:
            reward = 0 + reward
          elif self.label == decision:
            reward =  reward_factor+ reward
          else:
            reward = -reward_factor + reward

      #Reward de pasos
      if self.flags[3] == '1':
        current_step_num = sum(self.action_counter)
        diff_step = current_step_num - self.last_step_three

        reward_diff_step = self.sigmodialFunction(diff_step,19.2,-1.4,24.5,-11.3)
        reward = reward_diff_step + reward

        self.last_step_three = current_step_num

      #Reward de tamaño
      if self.flags[4] == '1':
        reward_len_agree = self.sigmodialFunction(len(agreelist),4.5,-0.6,4,-2)
        reward_len_disagree = self.sigmodialFunction(len(disagreelist),4.5,-0.6,4,-2)

      reward = reward_len_agree + reward_len_disagree + reward

    observation = self.BuildObservation()

    info = {}

    self.total_reward = self.total_reward + reward

    self.log = f'-------------------------------------------------- \n' + \
          f'model_name: {self.model_name} \n' + \
          f'flags: {self.flags} \n' + \
          f'Action counter:  {self.action_counter} \n' + \
          f'Action: {action} \n' + \
          f'Good: {self.counter_good} \n' + \
          f'Wrong: {self.counter_wrong} \n' + \
          f'rew_simility: {rew_simility} \n' + \
          f'rew_len: {rew_len} \n' + \
          f'reward_factor: {reward_factor} \n' + \
          f'reward_diff_step: {reward_diff_step} \n' + \
          f'reward_len_agree: {reward_len_agree} \n' + \
          f'reward_len_disagree: {reward_len_disagree} \n' + \
          f'Reward: {reward}, Total: {self.total_reward} \n' + \
          f'Current sent: {self.argumentLists.GetCurrentSent()} \n' + \
          f'Agree list: \n' + \
          f'{self.argumentLists.getAgreeList()} \n' + \
          f'Disagree list \n' + \
          f'{self.argumentLists.getDisagreeList()} \n'

    print(self.log)

    return observation, reward, isFinished, info


  def reset(self):

    self.last_step_three = sum(self.action_counter)

    print("Reinicio")

    #Buscar en la web el termino del titulo
    status = self.dataManager.ChargeNewFromFile()

    #Si no encuentra ningun resultado reinicia
    if not status:
      return self.reset()

    ##Loop que revisa si del html se obtuvieron sents
    while True:
      #Loop que revisa si se cargo y se extrajo un html exitosamente
      while True:
        status = self.dataManager.GoNextArticle()
        if status == 0:
          return self.reset()
        elif status == 1:
          break

      title, self.label, text = self.dataManager.GetLoadedData()
      sents = self.GetSents(text)
      if len(sents)>0:
        break        

    ##Reiniciar y cargar con nuevas sents las listas
    self.argumentLists.resetLists()
    self.argumentLists.ChargeSents(sents)

    #Agregar el titulo al agree list
    doc = self.nlp(title)
    self.argumentLists.append_agree_list(doc)

    #Texto de log
    log = f'\n\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n' + \
          f'title: {title} Label: {self.label} \n' + \
          ' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n'
      
    print(log)
    
    #formar el observation

    observation = self.BuildObservation()

    return observation  # reward, done, info can't be included
 
 
  def render(self, mode='human'):
    pass
 
 
  def close (self):
    pass
