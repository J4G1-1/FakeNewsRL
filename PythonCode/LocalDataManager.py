import pickle
import random
#import AdCounter

class LocalDataManager:

    def __init__(self, localdata_path = ''):
      self.dataChunk = []
      self.current_data = ''
      self.datalist = []
      self.ads_counted = []
      self.number_of_ads = 0
      with open(localdata_path, "rb") as fp:   # Unpickling
        self.dataChunk = pickle.load(fp)
      self.data_count = 0
      

    def ChargeNewFromFile(self):
        current_data_dict = random.choice(self.dataChunk)
        self.title = current_data_dict['title']
        self.label = current_data_dict['label']
        self.datalist = current_data_dict['data']
        self.ads_counted = current_data_dict['#ads']
        #self.ads_counted = current_data_dict['no_ads']
        self.data_count = 0
        self.current_data = self.datalist[self.data_count]
        self.number_of_ads = self.ads_counted[self.data_count]
        return True
      
    
    def GoNextArticle(self):
        #self.data_count = self.data_count + 1
        self.data_count += 1
        if len(self.datalist) == self.data_count:
            return 0

        self.current_data = self.datalist[self.data_count]
        self.number_of_ads = self.ads_counted[self.data_count]
        
        return 1        

    def GetLoadedData(self):
        return self.title, self.label, self.current_data
    
    def GetNumAds(self):
        return self.number_of_ads
        
        
