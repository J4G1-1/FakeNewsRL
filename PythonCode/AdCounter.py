#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

def is_in(cadena):
    #si se encuentra una referencia a anuncios en la cedena returna True
    
    patrones = [r"\bad(?!d)[a-zA-Z_0-9]*",r"\bads[a-zA-Z_0-9]*",
    r"\badvertisement[a-zA-Z_0-9]*", r"\bsponsored[a-zA-Z_0-9]*"]
    
    for i in patrones:        
        if len(re.findall(i, cadena))!=0:            
            return True
    return False

def domain(url):
    #obtiene el dominio de una url dada
    a = url.split('/')
    if "http" in a[0]:
        return a[2]
    else:
        return a[0]
        
def enlaces(bloque):
    urls_encontradas = []
    #extraer los enlaces que se encuentre dentro del bloque que está constituido
    #de codigo html
    a = str(bloque).split()
    for i in a:
        if 'href="h' in i:
            urls_encontradas.append(i.replace('href="','').replace('"',''))
                
    return urls_encontradas


def ads_found(soup, url):
    
    #en esta lista se guardaran los bloques de codigo que tenga
    #palabras claves que hagan referencia a anuncios
    anuncios = []
    
    #recuperamos todos los bloques que tenga etiquetas del tipo
    #<div> y <a>
    tags = soup.find_all(['a','div'])
        
    #revisa cada elementos en tags buscado referencias a anuncios
    #y si las encuentra, las añade a la lista: anuncios
    for i in tags:
        if is_in(str(i)):
            anuncios.append(i)
            
    #en esta lista se guardara las urls de los anuncios        
    urls_ads =[]    
    
    #buscaremos los enlaces de los anuncios  
    for bloque in anuncios:
        urls_ads.extend(enlaces(bloque))
    
    #eliminamos aquellas urls que contengan al dominio de la pagina
    #que estamos analizando
    urls_ads = [i for i in urls_ads if domain(url) not in i]
    
    #eliminamos las urls repetidas
    urls_ads = set(urls_ads)
    
    ads_count = len(urls_ads)
    
    #for i in urls_ads:print(i,'\n')
    #print('**'*50)
    #print('cantidad tentativa de anuncios: ',ads_count)        
    
    return ads_count


