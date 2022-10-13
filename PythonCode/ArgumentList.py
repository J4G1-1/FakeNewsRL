import numpy as np

#Esta clase define la estructura de datos que se encarga de 
#gestionar las listas de agree, disagree y el sent que se está leyendo
# 
# Sent list: Guarda los sent a analizar (por lo general los sents de una pagina)

# agree list: Guarda todos los sents que apunten que la noticia sea verdadera
# disagree list Guarda todos los sents que apunten que la noticia sea falsa
# Las dos listas anteriores guardan diccionario que tienen la siguiente estrucutra:
#     sent: Contiene al sent completo
#     sent_no_stopwords: Guarda una copia del sent pero sin incluir stop words (para calcular la similaridad)
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

  #Carga un nuevo conjunto de sents para su analisis.
  def ChargeSents(self,sent_list):
    self.sent_list = sent_list
    self.sent_index = 0
    self.current_sent = self.sent_list[0]

  #Metodo auxiliar para elaborar la copia sent_no_stopwords
  def GetStopWordsSent(self,sent):
    text = ""

    for token in sent:
        if not token.is_stop:
            text = text + token.text + " "
        
    no_stopword_doc = self.nlp(text)

    return no_stopword_doc

  #Agregar el sent al agree list
  #Regresa un vector que contiene la similitud que tiene el sent en cuestión
  #con los sent que ya estaban el la lista, esto para calcular recompensa.
  #Para calcular la similitud se hace uso del sent pero sin stop words.
  def append_agree_list(self,sent):

    #Se suma el vector del sent a agregar al accumulated agree
    self.accumulatedAgree = np.add(self.accumulatedAgree, sent.vector)

    #Uso de metodo auxiliar para obtener el sent sin stopwords
    sent_no_stopwords = self.GetStopWordsSent(sent)

    #Calculo de la similitud del sent nuevo son los sent de la lista
    similarity_list = []

    for sent_in_list in self.agreelist:
        similarity_list.append(sent_in_list["sent_no_stopwords"].similarity(sent_no_stopwords))

    self.agreelist.append({"sent": sent,
                            "sent_no_stopwords" : sent_no_stopwords})
    
    return similarity_list

  #Mismo proceso que agreelist pero usando el disagree list
  def append_disagree_list(self,sent):

    self.accumulatedDisagree = np.add(self.accumulatedDisagree, sent.vector)

    #Uso de metodo auxiliar para obtener el sent sin stopwords
    sent_no_stopwords = self.GetStopWordsSent(sent)

    #Calculo de la similitud del sent nuevo son los sent de la lista
    similarity_list = []

    for sent_in_list in self.disagreelist:
        similarity_list.append(sent_in_list["sent_no_stopwords"].similarity(sent_no_stopwords))

    self.disagreelist.append({"sent": sent,
                            "sent_no_stopwords" : sent_no_stopwords})

    return similarity_list
  
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
        return -1
  
  def getAgreeList(self):
    return [sent["sent"].text + "\n" for sent in self.agreelist]

  def getDisagreeList(self):
    return [sent["sent"].text + "\n" for sent in self.disagreelist]

  #Para imprimir el contenido en texto de cada list
  def PrintList(self):
    print("Agree List: ")

    for element in self.agreelist:
      print(element["sent"].text)

    print("Disagree List: ")
    
    for element in self.disagreelist:
      print(element["sent"].text)