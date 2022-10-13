from cgitb import reset

import gym
from gym import spaces

import numpy as np
import pandas as pd

import spacy
from scipy import stats

import os

from WebScrapper import WebScrapper
from ArgumentList import ArgumentList

#La clase FakeNewsEnv es una clase que implementa un gym environment
#Su objetivo es definir 

class FakeNewsEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(FakeNewsEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:

    self.action_counter=[0,0,0,0]

    #Crear carpetas de logs escritos por el programador
    logcustom = "logscustom"
    if not os.path.exists(logcustom):
      os.makedirs(logcustom)

    #Actions
    #0.El agente tiene 4 acciones:
    #1.Ignorar la sent actual
    #2.Agregar el sent a la lista de agree
    #3.Agregar el sent a la lista de disagree
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
    self.webScrapper = WebScrapper()
    #Creacion de la estrucutra de datos de listas
    self.argumentLists = ArgumentList(self.nlp)
    #Carga de dataset para fakenews
    self.dataFrameNews = pd.read_csv(r"/home/serapf/Desktop/FakeNewsRL/PythonCode/data/DataFakeNews.csv")
    
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

    ##Ignore sent
    if action==0:
      #Pasa a la siguiente sent
      more = self.argumentLists.GoToNextSent()

      if more is False:
        reward = -0.5
      else:
        reward = 0.3


    ##Add to affirm
    if action==1:
      present_sent = self.argumentLists.GetCurrentSent()

      more = self.argumentLists.GoToNextSent()
      #Si ya no hay sents se da una rencompensa negativa
      if more is False:
        reward = -0.1
      else:
        simility_list = self.argumentLists.append_agree_list(present_sent)
        self.SimilarityReward(simility_list)

    ##Add to objection
    if action==2:

      present_sent = self.argumentLists.GetCurrentSent()

      more = self.argumentLists.GoToNextSent()
      #Si ya no hay sents se da una rencompensa negativa
      if more is False:
        reward = -0.1
      else:

        simility_list = self.argumentLists.append_disagree_list(present_sent)
        self.SimilarityReward(simility_list)


    ##Go to the next url
    if action==3:

      status = self.webScrapper.GotoNextWebPage()

      decision = self.argumentLists.GetDecision()

      agreelist = self.argumentLists.getAgreeList()
      disagreelist = self.argumentLists.getDisagreeList()

      diff = abs(len(agreelist) - len(disagreelist))
      
      reward_factor = pow(diff,2)/3

      if not status:

        if decision == -1:
          reward = 0
        elif self.label == decision:
          reward = reward_factor
        else:
          reward = -reward_factor

        isFinished = True
      
      else:

        if decision == -1:
          reward = 0
        elif self.label == decision:
          reward = 0.1*reward_factor
        else:
          reward = -0.1*reward_factor
    
    observation = self.BuildObservation()

    info = {}

    self.total_reward = self.total_reward + reward

    log = f'-------------------------------------------------- \n' + \
          f'Action counter:  {self.action_counter} \n' + \
          f'Action: {action} \n' + \
          f'Current sent: {self.argumentLists.GetCurrentSent()} \n' + \
          f'Reward: {reward}, Total: {self.total_reward} \n' + \
          f'Agree list: \n' + \
          f'{self.argumentLists.getAgreeList()} \n' + \
          f'Disagree list \n' + \
          f'{self.argumentLists.getDisagreeList()} \n'

    print(log)

    return observation, reward, isFinished, info


  def reset(self):

    print("Reinicio")

    #Se busca el encabezado de una notica cualquiera
    randomSample = self.dataFrameNews.sample()
    title = randomSample["title"].iloc[0]
    self.label = randomSample["label"].iloc[0]

    #Buscar en la web el termino del titulo
    status = self.webScrapper.ChargeFromWeb(title)

    #Si no encuentra ningun resultado reinicia
    if not status:
      return self.reset()

    ##Loop que revisa si del html se obtuvieron sents
    while True:
      #Loop que revisa si se cargo y se extrajo un html exitosamente
      while True:
        status = self.webScrapper.GotoNextWebPage()
        if status == 0:
          return self.reset()
        elif status == 1:
          break

      html = self.webScrapper.GetLoadedHTML()
      sents = self.GetSents(html)
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