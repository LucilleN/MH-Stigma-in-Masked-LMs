#!/usr/bin/env python3

import sys, os, re
from IPython import embed
import pandas as pd
from pprint import pprint
import string
from random import shuffle

FILES = {
  "EmoLex": "NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt",
  "OptPess": ["Optimism-Pessimism-Lexicons/unigrams-pmilexicon.txt",
              "Optimism-Pessimism-Lexicons/bigrams-pmilexicon.txt"],
}
HEADER = "......................................................................."

def parse_emolex():
  f = open(os.path.join(os.path.dirname(__file__),FILES["EmoLex"]))
  # File has a header in 2 parts, starts after second set of periods

  header = 0
  words_to_cats = {}
  weird_lines = {}
  for l in f:
    l = l.strip()
    if l == HEADER:
      header += 1
      continue

    if header < 2: continue
    elif not l: continue
    else:
      w, cat, weight = l.split("\t")
      if weight == "1":
        words_to_cats[w] = words_to_cats.get(w,[])
        words_to_cats[w].append(cat)      
  return words_to_cats

def parse_opt():
  fs = FILES["OptPess"]

  words_to_cats = {}
  weird_lines = {}
  skip_lines = ["the list below is for","the lsit below is for"]
  
  for fn in fs:
    f = open(os.path.join(os.path.dirname(__file__),fn))
    header = 0
    for l in f:
      l = l.strip()
      if l == HEADER:
        header += 1
        continue

      if header < 2: continue
      elif not l: continue
      elif any([s in l.lower() for s in skip_lines]): continue
      else:
        out = l.split("\t")
        if len(out) == 1 or not any([str.isdigit(i) for i in l]):
          weird_lines[l] = out
          continue
        else:
          w, weight, _, _ = out
          words_to_cats[w] = words_to_cats.get(w,{})

          words_to_cats[w]["opt"] = float(weight)
          
  return words_to_cats
    
