import numpy as np
import RewardForDates as RD

#usando esta libreria ya no sera necesario usar sents sin stopwords.
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

#Esta clase define la estructura de datos que se encarga de 
#gestionar las listas de agree, disagree y el sent que se está leyendo
# 
# Sent list: Guarda los sent a analizar (por lo general los sents de una pagina)

# agree list: Guarda todos los sents que apunten que la noticia sea verdadera
# disagree list Guarda todos los sents que apunten que la noticia sea falsa
#
# accumulatedAgree: Es la suma de todos los vectores del agree list
# accumulatedDisagree: Es la suma de todos los vectores del disagree list
# Los dos anteriores sirven para hacer la obersvación que se le pasará a la red neuronal.

class ArgumentList:

  # Metodo constructor
  def __init__(self, nlp): 
    self.resetLists()
    self.nlp = nlp

  # Vacia toda la estrucutra y la deja lista para
  # recibir otro conjunto de sent
  def resetLists(self):
    self.current_sent = None
    self.sent_list = []
    self.sent_index = 0    

    self.agreelist = []
    self.accumulatedAgree = np.zeros(300)
    self.disagreelist = []
    self.accumulatedDisagree = np.zeros(300)
    
    self.dates_agree = []
    self.dates_disagree = []

  #Carga un nuevo conjunto de sents para su analisis.
  def ChargeSents(self, sent_list):
    self.sent_list = list(set(sent_list))
    self.sent_index = 0
    self.current_sent = self.sent_list[0]
    
  def cal_similarity(self, sent1, sent2):
    """
    calcula la similitud entre la sentencia 1 y la sentencia 2
    usando la libreria sentence_transformers.

    Parameters
    ----------
    sent1 : numpy.ndarray
        sentencia 1.
    sent2 : numpy.ndarray
        sentencia 2.

    Returns
    -------
    Float64
    """
    cos_sim = util.cos_sim(sent1, sent2)
    return cos_sim.item()
    

  #Agregar el sent al agree list
  #Regresa un vector que contiene la similitud que tiene el sent en cuestión
  #con los sent que ya estaban en la lista, esto para calcular recompensa.
  
  def append_agree_list(self, sent):
    #extraemos las fechas dentro del sent
    dates_in_sent = RD.extract_datetime(sent, self.nlp)
    if len(dates_in_sent)!=0:
        self.dates_agree.append(dates_in_sent)
    
    if len(self.dates_agree)>1:
        #formamos tuplas de fechas, usando las obtenidas en el
        #actual sent y el anterior
        date_list = [(i,j) for i in self.dates_agree[-1] for j in self.dates_agree[-2]]
        #calculamos la similaridad entre las fechas
        reward_dates = RD.similarity_dates(date_list, self.nlp)
    else:
        reward_dates=0
    
    #Se suma el vector del sent a agregar al accumulated agree
    self.accumulatedAgree = np.add(self.accumulatedAgree, sent.vector)
        
    #Calculo de la similitud del sent nuevo con los sent de la lista
    similarity_list = []
    
    #embedding1, codificacion del sent usando SentenceTransformer
    emb1 = model.encode(str(sent))
    
    #calculamos la similaridad del sent con los demas
    #sents en agreelist
    for sent_in_list in self.agreelist:
        emb2 = model.encode(str(sent_in_list))
            
        similarity_list.append(self.cal_similarity(emb1, emb2))
    
    #agregamos el sent al agreelist
    self.agreelist.append(sent)
        
    return [similarity_list, reward_dates]

  #Mismo proceso que agreelist pero usando el disagree list
  def append_disagree_list(self, sent):
    #extraemos las fechas dentro del sent
    dates_in_sent = RD.extract_datetime(sent, self.nlp)
    if len(dates_in_sent)!=0:
        self.dates_disagree.append(dates_in_sent)
    
    if len(self.dates_disagree)>1:
        #formamos tuplas de fechas, usando las obtenidas en el
        #actual sent y el anterior
        date_list = [(i,j) for i in self.dates_disagree[-1] for j in self.dates_disagree[-2]]
        #calculamos la similaridad entre las fechas
        reward_dates = RD.similarity_dates(date_list, self.nlp)
    else:
        reward_dates=0
    
    #Se suma el vector del sent a agregar al accumulated Disagree
    self.accumulatedDisagree = np.add(self.accumulatedDisagree, sent.vector)
    
    #Calculo de la similitud del sent nuevo son los sent de la lista
    similarity_list = []
        
    #embedding1, codificacion del sent usando SentenceTransformer
    emb1 = model.encode(str(sent))
    
    #calculamos la similaridad del sent con los demas
    #sents en disagreelist
    for sent_in_list in self.disagreelist:            
        emb2 = model.encode(str(sent_in_list))
        
        similarity_list.append(self.cal_similarity(emb1, emb2))                         
    
    #agregamos el sent al disagreelist
    self.disagreelist.append(sent)

    return [similarity_list, reward_dates]
  
  #De la lista de sent va al siguiente sent para su analisis.
  #Si ya no hay sents se envi auno False
  def GoToNextSent(self):

    if self.sent_index == len(self.sent_list)-1:
        return False

    self.sent_index = self.sent_index + 1

    self.current_sent = self.sent_list[self.sent_index]
  
  #Getters y setters de cada componente
  def GetCurrentSent(self):
        return self.current_sent  

  def GetAccumalatedAgree(self):
        return self.accumulatedAgree
    
  def GetAccumalatedDisagree(self):
        return self.accumulatedDisagree
  
  #Con base al contenido de las listas, se determina
  #si la noticia es falsa o verdadera
  # SI es verdadero 1 si es falso 0
  # !!!Para esta iteración solo se toma encuenta el tamaño de las listas
  #    sin embargo este criterio puede mejorar con creces.!!!!
  def GetDecision(self):
      agree_count = len(self.agreelist)
      disagree_count = len(self.disagreelist)

      if agree_count > disagree_count:
        return 1
      elif disagree_count > agree_count:
        return 0
      else:
          return self.ReValDecision1()

  def ReValDecision2(self, ads_reward_list):
      #Este metodo es para revaluar una desicion tomada
      #se usara este metodo cuando el GetDecision retorne -1
      vote = sum(ads_reward_list)/len(ads_reward_list)
      if vote <=60:
          return 1
      else:
          return 0
  def ReValDecision1(self):
      s1, s2 = 0, 0
      for i in self.agreelist:s1+=len(i)
      for i in self.disagreelist:s2+=len(i)
      if s1> s2:return 1
      else:return 0

  def getAgreeList(self):
    return [sent.text + "\n" for sent in self.agreelist]

  def getDisagreeList(self):
    return [sent.text + "\n" for sent in self.disagreelist]

  #Para imprimir el contenido en texto de cada list
  def PrintList(self):
    print("Agree List: ")

    for element in self.agreelist:
      print(element.text)

    print("Disagree List: ")
    
    for element in self.disagreelist:
      print(element.text)


