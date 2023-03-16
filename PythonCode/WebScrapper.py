from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from bs4 import BeautifulSoup
import re

import random

import time

#La clase de WebScrapper sirve para conseguir información 

# Su funcionamiento es el siguiente:
# 1. Con los resultados de una busqueda almacena los links para ser consultados.
# 2. Se va iterando por cada url y se carga el html
# 3. Se sigue este proceso hasta que se acaben los links
# TODO: implemenatar paginación para que de esa manera pueda conseguir más links más haya de la primera hoja de resultados

class WebScrapper:

  #Constructor: Principalmente inicia el web driver de geckdriver de firefox
  def __init__(self):
    self.urls = []
    self.urls_index = 0

    firefoxProfile = webdriver.FirefoxProfile()
    firefoxProfile.set_preference('permissions.default.stylesheet', 2)
    firefoxProfile.set_preference('permissions.default.image', 2)
    firefoxProfile.set_preference('dom.ipc.plugins.enabled.libflashplayer.so','false')
    firefoxProfile.set_preference("http.response.timeout", 5)
    firefoxProfile.set_preference("dom.max_script_run_time", 5)

    self.driver = webdriver.Firefox(firefox_profile=firefoxProfile)
    #Si la página tarda más de 5 segundos, se extrae el html como éste y se sigue
    self.driver.set_page_load_timeout(5)

    self.data = ''

  #El método busca en internet el string phrase y guarda los url devueltos.
  #TODO: El método actualmente usa duckduck, sin embargo con la estrucutra
  #hecha se puede usar google si el debido método es llamado. Por lo tanto,
  # se podría mejorar si da la opción de escoger el buscador a usar. Aparte
  # implementar otros navegadores como yahoo, bing, etc.
  def ChargeFromWeb(self, phrase):
    self.urls_index = 0
    self.urls = self.DuckDuckSeach(phrase)

    if len(self.urls) == 0:
        return False
    else:
        return True
  
  #Los siguiente métodos sirven para usar diferentes navegadores para buscar
  #el query solicitado. Por lo tanto, cada método implementa un navegador diferente.
  #Los parametros son:
  # phrase: es el término a buscar
  # enginelink: Es el url del navegador
  # xpath: Es la expresión xpath a utilizar para encontrar los urls dentro del html de los reusltados
  def GoogleSearch(self, phrase):
    return self.GetLinks(phrase,
                'https://www.google.com/',
                '//div[@class="yuRUbf"]/a[@href]')

  def DuckDuckSeach(self, phrase):
    return self.GetLinks(phrase,
            'https://duckduckgo.com/',
            '//h2[@class="LnpumSThxEWMIsDdAT17 CXMyPcQ6nDv47DKFeywM"]/a[@href]')
  
  # Usando los parametros desctiros en los métodos de los navegadores
  # El método GetLinks hace uso del web driver para buscar el query y extraer los url.
  # Adicionalmente cambia el orden en como los links fueron encontrados para evitar patrones.
  def GetLinks(self, phrase ,enginelink, xpath):

    try:
        self.driver.get(enginelink)
        
        ##Search what the user typed
        search = self.driver.find_element(by=By.NAME,value = 'q')
        search.send_keys(phrase)
        search.send_keys(Keys.RETURN)

        time.sleep(2)

        ## Get the urls of all the results of the current page of the search's results
        webpages = self.driver.find_elements(by=By.XPATH, value = xpath)
        urls = [webpage.get_attribute("href") for webpage in webpages]

        
        
        def is_not_a_document(url)->bool:
          #return True if the url does not correspond to a text document
          extensions = [".pdf",".doc", ".docx", ".txt", "csv", ".xls", ".xlsx", ".xlsm" , ".xlsb",]
          for i in extensions:
            if i in url:return False
          return True
        
        #filtramos las urls para obtener aquellas que no sean de documentos de texto
        urls = list(filter(is_not_a_document, urls))
        
        #ordenamos de manera aleatoria
        random.shuffle(urls)

        return urls

    except:
        print(f"Problema de {enginelink}")
        return []
  
  #El método GotoNextWebPage sirve para ir a la siguiente url de
  # la lista de urls generadas en GetLinks, y carga el html.
  # Si ya no hay url por explorar se regresa un 0.
  # Si hubo errores de al parsear el html al texto y limpiarlo se devuelve -1
  # Si no hay errores se regresa 1
  def GotoNextWebPage(self):

    if len(self.urls) == self.urls_index:
      return 0
    
    #Tolerancia de 5 segundos antes de cargar la pagina
    try:
      self.driver.get(self.urls[self.urls_index])
    except:
      print("No termino la carga, pero sigue")

    self.urls_index = self.urls_index + 1

    #Entrar a la pagina
    try:
      ##Obtener el html
      innerHtml = self.driver.page_source

      ##Use BeautifulSoup to convert the html file to plain text and clean the data
      soup = BeautifulSoup(innerHtml, "html.parser")
      html_text = soup.get_text()
      html_text = re.sub('(?<=\n)\s+\n', '', html_text)
      html_text = re.sub('\n', '.', html_text)
      html_text = re.sub('\s{2,}', '.', html_text)
      html_text = re.sub('\[\d+]', '', html_text)
      self.data = re.sub('\|+|\/+|-+', '\n', html_text)

      return 1

    except:
      return -1

  #Getter del texto obtenido del html de la última pagina cargada exitosamente.
  def GetLoadedData(self):
    return self.data

  def Terminar(self):
    self.driver.quit()
