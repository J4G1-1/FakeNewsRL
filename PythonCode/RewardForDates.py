

def extract_datetime(text, nlp):
    """
    extraer fechas de un texto y las devuelve 
    en una lista
    
    Returns
    -------
    list.
    """    
    fechas = []
    doc = nlp(str(text))
    for entity in doc.ents:
        if entity.label_=="DATE":
            fechas.append(entity.text)
        
        #print(entity.text, entity.label_)
    return fechas


def similarity_dates(list_dates, nlp):
    """
    Dada una lista de pares de fechas
    calcula la similitud entre las fechas    
    
    Parameters
    ----------
    list_dates : list de tuplas
    
    Returns
    -------
    float64
    """
    sim = []    
    
    for i in list_dates:
        d1 = nlp(i[0])
        d2 = nlp(i[1])
        simi = d1.similarity(d2)
        if simi<0.5:
            sim.append(-simi)
            #print(i,' - ',-simi)
        else:
            sim.append(simi)
            #print(i,' - ',simi)
    return sum(sim)

   


   
