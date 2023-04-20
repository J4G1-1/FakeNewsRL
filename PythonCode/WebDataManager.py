import pandas as pd
from WebScrapper import WebScrapper

class WebDataManager:

    def __init__(self, titles_data_path = ''):
      #Carga de dataset para fakenews
      self.dataFrameNews = pd.read_csv(titles_data_path)
      self.webScrapper = WebScrapper()
      

    def ChargeNewFromFile(self):
        #Se busca el encabezado de una notica cualquiera
        randomSample = self.dataFrameNews.sample()
        self.title = randomSample["title"].iloc[0]
        self.label = randomSample["label"].iloc[0]
        #Buscar en la web el termino del titulo
        status = self.webScrapper.ChargeFromWeb(self.title)

        return status
    
    def GoNextArticle(self):
        return self.webScrapper.GotoNextWebPage()

    def GetLoadedData(self):
        return [self.title, self.label, self.webScrapper.GetLoadedData()]
    
    def GetNumAds(self):
        return self.webScrapper.GetNumAds()
