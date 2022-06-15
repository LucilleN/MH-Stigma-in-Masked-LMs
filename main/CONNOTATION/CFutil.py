#!/usr/bin/env python3

import sys, os, re
from IPython import embed
import pandas as pd
from pprint import pprint
import string
from random import shuffle

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import spacy
  
CF_FILES = {
  "agency": "agency_verb.csv",
  "authority": "authority_verb.csv"
}

def parse_connotation():
  nlp = spacy.load('en') 
  words_to_cats = {}
  dfs = []
  
  for frame in ["agency","authority"]:
    f = os.path.join(os.path.dirname(__file__), CF_FILES[frame])
    df = pd.read_csv(f)
    df = df.iloc[:,:2]
    df.columns = ["verb",frame]
    print(frame,df.shape)
    convert_labels = {"=": "equal", "+": "pos","-": "neg"}
    df[frame] = df[frame].apply(
      lambda x: frame+"_"+convert_labels.get(x,x).lower())

    df.set_index("verb",inplace=True)
    df.index.name = "verb"
    
    dfs.append(df)
    
  df = pd.concat(dfs,axis=1)
  df.index.name = "verb"
  df.reset_index(inplace=True)

  ## Lemmatizing verbs
  # wnl.lemmatize(x.split()[0], pos='v')
  # df["verb"]
  def lemm(x):
    toks = nlp("he "+x)
    return (toks[1].lemma_ + " " + str(toks[2:])).strip()
  
  df["verb_lem"] = df.verb.apply(lemm)
  
  df = df[df.columns.drop("verb")]
  
  for v, r in df.set_index("verb_lem").iterrows():
    words_to_cats[v] = [c for c in r.values.tolist() if not pd.isnull(c)]
    
  return words_to_cats	


def findVerbs(doc, nlp=None, index=False):
  """ """
  if nlp is None:
    nlp = spacy.load('en')
  toks = nlp(doc)
  if index:
    verbs = [(i,t.lemma_.lower()) for i,t in enumerate(toks) if t.pos_ == "VERB"]
  else:
    verbs = [t.lemma_.lower() for t in toks if t.pos_ == "VERB"]
  return verbs

if __name__ =="__main__":
  parse_connotation()
