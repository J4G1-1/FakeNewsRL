# import the threading module
import threading

from WebScrapper import WebScrapper
import threading
import pandas as pd
import pickle
import spacy

import itertools

chunkList = []
#Carga de spacy para sent analysis y embeddings
nlp = spacy.load("en_core_web_md")

class thread_news(threading.Thread):
    
    def __init__(self, thread_name, thread_ID, dataFrameNews,begin, end, nlp):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.thread_ID = thread_ID
        self.dataFrameNews = dataFrameNews[begin:end]
        print(self.dataFrameNews is dataFrameNews)
        self.webScrapper = WebScrapper()
        
        self.begin = begin
        self.end = end

        self.chunk = []

        self.nlp = nlp
        
    def GetChunk(self):
        return self.chunk
    
    # helper function to execute the threads
    def run(self):
        print('Comenzo ',self.thread_ID)

        rows = self.dataFrameNews.iterrows()
        for index, row in rows:
            title = row['title']
            label = row['label']
            #Buscar en la web el termino del titulo
            status = self.webScrapper.ChargeFromWeb(title)
            #Si no encuentra ningun resultado reinicia
            if status:
                sents_list = []
                number_of_ads = []
                while True:
                    status = self.webScrapper.GotoNextWebPage()
                    
                    if status == 0:
                        break
                    
                    if status == 1:
                        try:
                            text = self.webScrapper.GetLoadedData()
                            GetNumAds = self.webScrapper.GetNumAds()
                            doc = self.nlp(text)
                            sents = [sentence for sentence in doc.sents]
                            if len(sents)>0:
                                sents_list.append(text)
                                number_of_ads.append(GetNumAds)
                        except Exception as e:
                            print("Hubo problemas al parsear el html ", e)
            
                dict = {'title':title,'label':label,'data':sents_list, 'no_ads':number_of_ads}
                self.chunk.append(dict)
                dict = {}
        print('Termino ',self.thread_ID)
        self.webScrapper.Terminar()

dataFrameNews = pd.read_csv(r"./data/DataFakeNews.csv")

global_begin = 0
global_end = 500

diff = global_end - global_begin

number_thread = 4
chunk_size = int(diff/number_thread)
thread_list = []

for i in range(0,number_thread):
    thread_list.append(thread_news(i, i,dataFrameNews,i*chunk_size + global_begin,(i+1)*chunk_size + global_begin, nlp))
    thread_list[i].start()

for i in range(0,number_thread):
     thread_list[i].join()

for i in range(0,number_thread):
    chunkList.append(thread_list[i].GetChunk())

chunkList = list(itertools.chain.from_iterable(chunkList))

with open(f'chunk_{global_begin}-{global_end}', "wb") as fp:   #Pickling
   pickle.dump(chunkList, fp)


print('terminado')

