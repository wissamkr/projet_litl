
# coding: utf-8

# In[76]:


import re
from tqdm import tqdm
from math import log2
from xml.dom.expatbuilder import Namespaces
from lxml import etree
from lxml.builder import *

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

path_corpus = "Corpus_Reddit_long.xml"
path_corpus_out = "CalculsTfIdf.csv"

def get_messages(path):
    #On utilise la bibliothèque etree pour pouvoir parser le corpus reddit-TIFU qui est au format xml
    #Ici le corpus est appelé "Corpus_reddit.xml" et est placé and un dossier "Corpus_Reddit/"
    document = etree.parse(path)
    root = document.getroot()
    message = root.findall('message')
    #Les différents messages sont stockés dans un itératif "message"
    newroot = E("Corpus_numerote")

    #Initialisation d'une liste où les postes seront stockés. Ici : un poste=un document. 
    liste = [] 
    stats = []
    incrID = 0

    #On itère dans le corpus pour ajouter le texte de chaque message non vide à une liste
    for element in message:
            text = element.text
            if text is not None:
                 #(tokenisation(str(text)))> 200 :if
                    #Formule très lourde pour enlever 15% des post les moins long du calcul
                    #Serait sans doute mieux de le faire sur la base des charachtères (pas de tokenisation)
                message2 = etree.SubElement(newroot, "message", attrib={"id":str(incrID)})
                incrID += 1 #On rajoute 1 au numéro d'id
                text = text.rstrip("\n")
                text = re.sub("\t"," ",text)
                text = re.sub("\n", " ",text)
                text=re.sub(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', '<url>', text) #on remplace les url par une balise <url>
                message2.text = text
                liste.append(text.lower())
    CorpusNumerote = etree.ElementTree(newroot)
    CorpusNumerote.write("Corpus_Redditall_num.xml", pretty_print=True, xml_declaration=True, encoding="utf-8")
    return liste

def get_tfidf(sparse_tf, dense_idf, feature_names):
    dataset = []

    num_row = sparse_tf.shape[0]

    for idDoc in tqdm(range(num_row)):
        dense_tf = pd.DataFrame(sparse_tf[idDoc,].toarray()[0], index = feature_names)

        tf_over_idf = dense_tf/dense_idf
        tf_over_idf.rename(columns={0:"tfidf"}, inplace=True)

        tf_over_idf_non_zero = tf_over_idf[(tf_over_idf != 0).values].copy()
        tf_over_idf_non_zero["logtfidf"] = tf_over_idf_non_zero["tfidf"].apply(log2)

        tf_over_idf_non_zero["idDoc"] = idDoc

        tf_over_idf_non_zero = tf_over_idf_non_zero.reset_index().rename(columns={"index": "3gram"})

        dataset.append(tf_over_idf_non_zero)

    return pd.concat(dataset, ignore_index=True)



messages = get_messages(path_corpus)

#Création du dataframe à vectoriser dans l'étape suivante:
#Pour Disco on utilisait input='filename' pour pouvoir traiter le contenu de notre collection de document txt directement, 
#ici on n'a pas besoin parce que le contenu est accessible directement dans le dataframe créé

df = pd.DataFrame(messages, columns=["Message"])

#sparse_tf cas-où j'ai rajouté une colonne pour avoir le numéro du post
#df['NbDoc'] = np.arange(len(df))

#Initialisation d'un vectoriseur pour le nouveau dataframe (où chaque ligne représente un poste)
count_vectorizer = CountVectorizer(max_df = 0.1,analyzer="char",ngram_range = (3,3))

#création d'un vectoriseur TF.IDF
tfidf_transformer = TfidfTransformer(use_idf = True)
#ET d'un vectoriseur seulement TF : on va combiner les deux pour avoir une matrice TF/IDF
tf_transformer = TfidfTransformer(use_idf = False)

#Fit sur nos données
count_vecto = count_vectorizer.fit_transform(df["Message"])

#On utilise l'algorithme tf_idf sur la matrice de trigrammes de char "count_vecto"
tfidf = tfidf_transformer.fit(count_vecto)
#Dont on prend seulement les poids idf (méthode .idf)

#On enregistre les poids idf pour chaque trigramme dans un DataFrame
#Qu'on transpose : on interverti les axes vertical et horizontal
feature_names = count_vectorizer.get_feature_names()
dense_idf = pd.DataFrame(tfidf.idf_, index = feature_names)

#On a nos poids idf pour le calcul tf/idf, on passe maintenant à la matrice tf :
#Création d'une matrice avec les ngrammes et leur tf (chaque ligne correspond à un document)
sparse_tf = tf_transformer.fit_transform(count_vecto) #matrice creuse

print("getting TF/IDF")
Tfidf = get_tfidf(sparse_tf, dense_idf, feature_names)

print("nouveau fichier CalculsTfIdf créé")
Tfidf.to_csv("CalculsTfIdf.csv",index = False,sep=",")

print("application du quantile")


# In[77]:


#On utilise la bibliothèque etree pour pouvoir parser le corpus reddit-TIFU qui est au format xml
#Ici le corpus est appelé "Corpus_reddit.xml" et est placé and un dossier "Corpus_Reddit/"

#message = newroot.findall('message')
#Les différents messages sont stockés dans un itératif "message"
print("sélection des messages contenant les trigrams les plus rares")
from lxml.html.builder import E
document = etree.parse(path_corpus)
root = document.getroot()
message = root.findall('message')
#Les différents messages sont stockés dans un itératif "message"
newroot = E("Corpus_numerote")
#CorpusNumerote = etree.ElementTree(newroot)

#CorpusNumerote.write("Corpus_Reddit_20k_num.xml", pretty_print=True, xml_declaration=True, encoding="utf-8")

data = pd.read_csv("CalculsTfIdf.csv", encoding="UTF-8", quotechar='"', sep=",", decimal=".", low_memory = False)

x = list(data["logtfidf"])
x = np.array(x)
threshold = float(np.nanquantile(x, 0.02))
data2 = data.query(f"logtfidf < {threshold}")


# In[78]:


print("extraction du contenu des messages")
arbre = etree.parse("Corpus_Redditall_num.xml")   
racine = arbre.getroot()
def recupMess (ligne) :
    expr = "//message[@id= $ID]"
    IdLigne = str(int(ligne["idDoc"]))
    Message = racine.xpath(expr, ID = IdLigne)[0].text
    return(Message)
test = data2.apply(recupMess, axis = 1)
data2["message"] = test
print("extraction du contenu des messages terminée")


# In[81]:


print("extraction des tokens et segmentation des messages")

res_df = pd.DataFrame(columns=["id", "trigram", "message", "extract", "token", "tfidf", "logtfidf"])
ecart = 35

def get_tokens(tri, msg):
    correct_ngram = ""
    for char in tri:
        if char in "$^+?*()[].\\" :
            correct_ngram+="\\"
        correct_ngram+=char
    myregex=rf'((?:^|\W)[\w\"\\\/.?!:\-&]*{correct_ngram}[\w\"\\\/.?!:\-&]*(?:$|\W))'
    match=re.findall(myregex, msg)
    return match

for x in range(0, len(data2['idDoc'])) :   
    tri = data2.iloc[x]["3gram"]
    msg = data2.iloc[x]["message"]
    ide = data2.iloc[x]["idDoc"]
    tfidf = data2.iloc[x]["tfidf"]
    logtfidf = data2.iloc[x]["tfidf"]
    if msg is not None:
        msg = re.sub(r"\s+", " ", msg)
        match = get_tokens(tri, msg)
        for elem in set(match):
            correct_token = ""
            for char in elem:
                if char in "$^+?*()[].\\" :
                    correct_token+="\\"
                correct_token+=char
            ecart = 35
            regex2 = r"(?:\S*\s){0," + str(ecart) + "}" + correct_token.strip() + "(?:\s\S*){0," + str(ecart) + "}"
            for match in re.findall(regex2, msg):
                insert_row = {"id":ide, "trigram":tri, "message":msg, "extract":match, "token":elem, "tfidf":tfidf, "logtfidf":logtfidf}
                res_df = pd.concat([res_df, pd.DataFrame([insert_row])])

print("extraction des tokens et segmentation des messages terminée")
print("nouveau fichier toAnnotate.csv créé")


# In[94]:


byDoc2 = res_df.groupby("id")["logtfidf"].min().reset_index()
DicMini = {}
def add2Dic (ligne) :
    DicMini[int(ligne["id"])] = ligne["logtfidf"]
byDoc2.apply(add2Dic, axis=1)


# In[95]:


def keepMinimum (ligne):
    global min_df
    first = ligne["logtfidf"]
    id1 = ligne["id"]
    if (id1, first) in DicMini.items():
        
        insert_row = {"id":ligne["id"], 
                      "trigram":ligne["trigram"], 
                      "message":ligne["message"], 
                      "extract":ligne["extract"], 
                      "token":ligne["token"], 
                      "tfidf":ligne["tfidf"],
                      "logtfidf":ligne["logtfidf"], 
                     }
        min_df = pd.concat([min_df, pd.DataFrame([insert_row])])
        


# In[96]:


stackCol = list(res_df.columns)
min_df = pd.DataFrame(columns=stackCol)
res_df.apply(keepMinimum, axis= 1)


# In[102]:


print("conversion en jsonl")

import pandas as pd
import json 
import re
ext = dict()  
sortie = open('toAnnotate.jsonl', 'w+', encoding='utf8')

for x in range(0, len(min_df['extract'])) :   
    txt = min_df.iloc[x]["extract"]
    ide = min_df.iloc[x]["id"]
    ext[x] = {'text' : txt,'meta' : {"identifiant" : ide}} 
    json.dump(ext[x], sortie, ensure_ascii=False)    
    
    sortie.write("\n")
    
    txt = ""
    ide= ""

print("conversion en jsonl terminée")
print("nouveau fichier toAnnotate.jsonl créé")

