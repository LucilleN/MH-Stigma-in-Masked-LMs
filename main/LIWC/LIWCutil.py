#!/usr/bin/env python3

import sys, os, re
from IPython import embed
import pandas as pd
from pprint import pprint
import string
from random import shuffle


LIWC_FILES = {
  "2007": "LIWC2007_English100131.dic",
  "2015": "LIWC2015_English.dic",
}
CAT_DELIM = "%"

def parse_liwc(w,whitelist=None):
  if w == "2015":
    return parse_liwc_2015(whitelist=whitelist)
  elif w == "2007":
    return parse_liwc_2007(whitelist=whitelist)



def parse_liwc_2015(whitelist=None):
  f = open(os.path.join(os.path.dirname(__file__),LIWC_FILES["2015"]))
  cats_section = False
  words_to_cats = {}
  id_to_cat = {}
  weird_lines = {}
  for l in f:
    l = l.strip()
    if l == CAT_DELIM:
      cats_section = not cats_section
      continue

    if cats_section:
      try:
        i, cat = l.split("\t")
        cat = cat.split()[0]
        id_to_cat[int(i)] = cat
      except: pass # likely hierarchical category tags
    else:
      w, cats = l.split("\t")[0], l.split("\t")[1:]
      if "(" in w and ")" in w:
        w = w.replace("(","").replace(")","")
      try:
        words_to_cats[w] = [id_to_cat[int(i)] for i in cats]
      except:
        weird_lines[w] = cats

  # Finetuning cause like is weird
  discrep = [w for w,cs in words_to_cats.items() if id_to_cat[53] in cs]
  cs = words_to_cats["53 like*"]
  words_to_cats.update({d+" like*": cs for d in discrep})
  del words_to_cats["53 like*"]

  ## If whitelist
  if whitelist:
    words_to_cats = {w: [c for c in cs if c in whitelist] for w,cs in words_to_cats.items()}
    words_to_cats = {w:cs for w,cs in words_to_cats.items() if cs}
  
  return words_to_cats

def parse_liwc_2007(whitelist=None):
  f = open(os.path.join(os.path.dirname(__file__),LIWC_FILES["2007"]))
  cats_section = False
  words_to_cats = {}
  id_to_cat = {}
  weird_lines = {}
  for l in f:
    l = l.strip()
    if l == CAT_DELIM:
      cats_section = not cats_section
      continue

    if cats_section:
      i, cat = l.split("\t")
      id_to_cat[int(i)] = cat
    else:
      w, cats = l.split("\t")[0], l.split("\t")[1:]
      try:
        words_to_cats[w] = [id_to_cat[int(i)] for i in cats]
      except:
        if w == "like":
          words_to_cats[w] = ["filler", "posemo", "time", "affect"]
        elif w == "kind":
          words_to_cats["kind of"] = ["cogmech", "tentat"]
          words_to_cats["kind"] = ["affect", "posemo"]
        else:
          weird_lines[w] = cats
          
  words_to_cats_d = {w: {c:1 for c in cs} for w,cs in words_to_cats.items()}


  ## If whitelist
  if whitelist:
    words_to_cats = {w: [c for c in cs if c in whitelist] for w,cs in words_to_cats.items()}
    words_to_cats = {w:cs for w,cs in words_to_cats.items() if cs}

  return words_to_cats

def preprocess(doc):
  """Document is a string."""
  better = doc.lower().replace("kind of", "kindof")
  wb = re.compile(r'\b\S+?\b')
  l = len(wb.findall(better))
  return better, l

def extract(liwc, doc, percentage=True):
  """
  Counts all categories present in the document given the liwc dictionary.
  percentage (optional) indicates whether to return raw counts or
  normalize by total number of words in the document"""
  doc, n_words = preprocess(doc)
  extracted = {}

  for w, cats in liwc.items():
    # print(cats)
    if all([c in string.punctuation for c in w]):
      w_re = re.escape(w)
    else:
      w_re = r"\b"+w
      if "*" in w:
        w_re = w_re.replace("*",r"\w*\b")
        if w_re[-2:] != r"\b": w_re += r"\b"
        
        else: w_re += r"\b"
    r = re.compile(w_re)
    matches = r.findall(doc)
    if matches:
      for c in cats:
        extracted[c] = extracted.get(c,0)+len(matches)

  if percentage:
    ## Turn into percentages
    extracted = {k: v/n_words for k,v in extracted.items()}
  return extracted

def reverse_dict(d):
  cats_to_words = {}
  for w, cs in d.items():
    for c in cs:
      l = cats_to_words.get(c,set())
      l.add(w)
      cats_to_words[c] = l
  return cats_to_words

def sample_cat(rev_d, cat,n=10):
  l = list(rev_d[cat])
  shuffle(l)
  return l[:n]

def main(w):
  d = parse_liwc(w)
  rev_d = reverse_dict(d)
  # d = {'kindof': d['kindof'], 'like': d['like'], "reall* like*": d['unlov*']}
  # pprint(d)
  test = "This is an unlovingly sentence, really like, kind of an unloving sentence. " * 2
  print(preprocess(test))
  pprint(extract(d,test,False))

  embed()
  
if __name__ == '__main__':
  which_liwc = sys.argv[1]
  if which_liwc not in ["2007", "2015"]:
    print("Please choose 2007 or 2015")
  main(which_liwc)
