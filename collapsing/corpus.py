import os
import re
import mmap
import utils 
import random
import copy
from collections import defaultdict
from sys import stdout
import pickle as pkl
import pdb

class Sentence:
  def __init__(self):
    return

class RawWord:
  def __init__(self, form, lower):
    self.form = form
    if lower: self.form = self.form.lower()
    return

  def __getitem__(self, key):
    assert(key == 'FORM')
    return self.form


class RawSentence:
  # TO BE IMPLEMENTED
  def __init__(self, sentence):
    if isinstance(sentence,str):
      self.sentence_str = sentence
      self.forms = sentence.split()
      self.positions = [RawWord(form, True) for form in self.forms]
      return

  def __len__(self): return len(self.forms)
  def __getitem__(self,idx): 
    try:
      return self.positions[idx]
    except IndexError:
      raise(IndexError("Index out of range: maximum sentence length is {0}".format(len(self))))
    
###################################################
class UDWord:
  def __init__(self, parts, lower):
    if len(parts) != 10:
      print(parts)
      raise(ValueError)
    self.dictionary = dict(zip(['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC'],parts))

    if lower: 
      self.dictionary['FORM'] = self.dictionary['FORM'].lower()
      self.dictionary['LEMMA'] = self.dictionary['LEMMA'].lower()
    
    #self.dictionary['ID'] = int(self.dictionary['ID'])
    #self.dictionary['HEAD'] = int(self.dictionary['HEAD'])

    # if want to use lemma instead of form
    #self.dictionary['FORM'] = self.dictionary['LEMMA']

    self.dictionary['DEPREL'] = self.dictionary['DEPREL'].split(':')[0]
    #self.dictionary['XPOS'] = self.dictionary['XPOS'].split('-')[0]

    try: 
      intid = int(self.dictionary['ID'])
      intpid = int(self.dictionary['HEAD'])
      self.dictionary['HEADDIRECTION']=(1, -1)[intpid < intid]
    except ValueError:
      #print("value error: cannot cast string to int [{0} {1}]".format(self.dictionary['ID'], self.dictionary['HEAD']))
      self.dictionary['HEADDIRECTION']=None
    self.dictionary['KL'] = 1 # The KL to N(0,1)

    self.dictionary['HEADFORM']=None # this field is set in UDSentence

    if self.dictionary['FEATS'] != '_':
      feats = self.dictionary['FEATS'].split('|')
      for feat in feats:
        try: 
          (f,v) = feat.split('=')
        except ValueError:
          raise ValueError("error in features columns {0} feature {1}".format(self.dictionary['FEATS'],feat))

        self.dictionary[f] = v

  def __str__(self,delim='\t'):
    return delim.join([self.dictionary[key] for key in ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']])

  def __getitem__(self,key):
    try: 
      return self.dictionary[key]
    except KeyError:
      return '_'

  def __setitem__(self, key, value):
    self.dictionary[key] = value

  def is_mwe(self): return re.search('-',self.dictionary['ID'])


class UDSentence:
  def __init__(self, sentence, verbose = 0, lower=True, collapse_rels=None):
    self.verbose = verbose
    self.__current_idx = 0
    self.lower = lower
    self.collapse_rels = collapse_rels
    self.positions, self.mwes, self.forms, self.utags, self.sentence_str, self.dependencies = self._extract(sentence, self.verbose)
    
  def __str__(self,delim='\n'): 
    return delim.join([str(self.positions[p]) for p in range(1,len(self))])

  def collapse(self, positions, rels):
    def recursive(p):
      if p['DEPREL'] in rels:
        parent = [pp for pp in positions if pp['ID'] == p['HEAD']][0]
        parent['FORM'] += '__' + p['FORM']
        if parent['HEAD'] != '0':
          recursive(parent)

    for p in positions: 
      recursive(p)
    return positions
 
  def _extract(self, sentence, verbose = 0, shuffle=False):
    if isinstance(sentence, str):
      tokens = sentence.split(CoNLLUCorpus.EOL)
      tokens = [x for x in tokens if x] # remove th empty tokens. The last token is always redundant
      if shuffle: random.shuffle(tokens)
    elif isinstance(sentence,list): # list of strings
      tokens = sentence

    positions = [UDWord(['0', '*root*', '_', '_', '_', '_', '0', '_', '_', '_'], self.lower)]
    mwe = []
    forms = []
    utags = []
    dependencies={}
    sentence_str = ''
    for token in tokens:
      if (token.startswith(CoNLLUCorpus.COMMENT_MARKER)): continue
      udword = UDWord(token.split(CoNLLUCorpus.DELIM), self.lower)
      if udword.is_mwe():
        mwe.extend(udword['FORM'].split('-'))
        sentence_str += udword['FORM']
        if not re.search('SpaceAfter=No', udword['MISC']): sentence_str += ' '
      elif re.search('^[0-9]+$',udword['ID']): 
        positions.append(udword)
        forms.append(udword['FORM'])
        utags.append(udword['UTAG'])
        dependencies[(int(udword['HEAD']),int(udword['ID']))]=udword['DEPREL']
        if not(udword['ID'] in mwe): 
          sentence_str += udword['FORM']
          if not re.search('SpaceAfter=No', udword['MISC']): sentence_str += ' '
      elif re.search('.',udword['ID']): 
        if verbose > 0:
          print("Warning: the position with ID {0} is ignored".format(udword['ID'])) 
      else:
        raise ValueError("error in ID format: {0}".format(udword['ID']))


# NEW COLLAPSING CODE
    collapsed_positions = self.collapse(positions, self.collapse_rels)

# OLD COLLAPSING CODE
#    collapsed_positions = []
#    for p in positions: 
#      # set head form
#      p['HEADFORM'] = positions[int(p['HEAD'])]['FORM']
#
#      # collapsing
#      if p['DEPREL'] in self.collapse_rels:
#        parent = [pp for pp in positions if pp['ID'] == p['HEAD']][0]
#        parent['FORM'] += '__' + p['FORM']
#        for pp in positions: 
#          if pp['HEAD'] == p['ID']: pp['HEAD'] = parent['ID']
#        for pp in positions:
#          if int(pp['ID']) > int(p['ID']): pp['ID'] = str(int(pp['ID']) - 1)
#          if int(pp['HEAD']) > int(p['ID']): pp['HEAD'] = str(int(pp['HEAD']) - 1)
#        
#          deps = ''
#          for d in pp['DEPS'].split('|'):
#            #if d == '_' or d == '': continue
#            try: 
#              h,r = d.split(':') 
#              if int(h) > int(p['ID']): h = str(int(h)-1)
#              elif int(h) == int(p['ID']): h = parent['ID']
#              deps += h + ':' + r + '|'
#            except ValueError: 
#              continue
#          if deps != '': 
#            pp['DEPS'] = deps[:-1]
#      else:
#        collapsed_positions.append(p)
    
    return collapsed_positions, mwe, forms, utags, sentence_str, dependencies


  def __del__(self):
    return

  def __getitem__(self,x):
    assert(isinstance(x,int))
    return self.positions[x]

  def __len__(self):
    return len(self.positions)

  def __iter__(self):
    self.__current_idx = 0
    return self

  def __next__(self):
    if self.__current_idx >= self.__len__() : raise StopIteration
    current = self[self.__current_idx]
    self.__current_idx += 1
    return current

###################################################

class Corpus:
  def __init__(self, path, max_snt_len=None, encoding='utf-8', verbose=0) :
    self.path = path 
    self.encoding = encoding
    self.num_sentences = 0
    self._sentence_offset = {}
    self.vocabs = {}
    self.max_snt_len = max_snt_len
    try:
      fp = open(self.path,'r+', encoding=self.encoding)
    except FileNotFoundError :
      raise Exception("corpus not found in {0}".format(self.path))
    else :
      self.__fp = fp
    self.size = os.stat(path).st_size
    self.to_save = [self.path, self.encoding, self.num_sentences, self._sentence_offset, self.vocabs]

  def __del__(self): self.__fp.close()
  def __len__(self): return self.num_sentences
  def __str__(self, delim='\n'): return delim.join([str(snt)+delim for snt in self])
  def reset(self): self.__fp.seek(0,os.SEEK_SET)
  def progress(self): return self.__fp.tell()*1.0/self.size
  def tell(self): return self.__fp.tell()
  def seek(self,pos, whence=os.SEEK_SET): self.__fp.seek(pos, whence)
  def handler(self): return self.__fp
  def readline(self): return self.__fp.readline()
  def save(self, path): 
    with open(path, 'w') as fp:
      pkl.dump(self.to_save, fp)

###################################################

# each line is a sentence with no annotation
class RawCorpus(Corpus):
  def __init__(self, path, block_size=1024) :
    self.vocabs = {}
    self.block_size = block_size
    super().__init__(path)
    self.vocabs, self._sentence_offset, self.num_sentences = self.__index_corpus() 
    return

  def __iter__(self):
    self.seek(0)
    return self

  def __next__(self):
    if self.tell() >= self.size : raise StopIteration
    return self._next_sentence()

  def _next_sentence(self):
    return RawSentence(self.readline().strip())

  def __getitem__(self, snt_idx):
    idx = int(snt_idx/self.block_size)
    self.seek(self._sentence_offset[idx])
    for i,l in enumerate(self.handler()):
      if i == snt_idx%self.block_size: 
        return RawSentence(l.strip())
      if i > self.block_size:
        raise IndexError
    raise IndexError

  def __len__(self):
    return self.num_sentences

  def __index_corpus(self) :
    num_tokens = 0
    num_sentences = 0
    vocabs = {}
    sentence_offset = []
    with open(self.path,'r') as f :
      progress = 0
      sentence_offset.append(f.tell())
      line = f.readline()
      while line :
        num_sentences += 1
        if not(num_sentences%self.block_size):
          sentence_offset.append(f.tell())
        if not(num_sentences%10000) :
          progress = f.tell()*1.0 / self.size
          utils.update_progress(progress,"Counting vocabs", 40)

        tokens = line.strip().split()
        num_tokens += len(tokens)
        list(map(lambda x: utils.inc_dict_value(vocabs,x) , tokens))

        line = f.readline()

      if progress < 1: utils.update_progress(1,"Counting vocabs", 40)
    return vocabs, sentence_offset, num_sentences

###################################################

class CoNLLUCorpus(Corpus):
  EOL='\n'
  DELIM='\t'
  COMMENT_MARKER='#'

  def __init__(self,path, max_snt_len, collapse_rels) :
    super().__init__(path, max_snt_len=max_snt_len, encoding="utf-8")
    self.collapse_rels = collapse_rels
    self.vocabs, self._sentence_offset, self.num_sentences, self.tagset = self.__index_corpus() 
    self.to_save.append(self.tagset)
    return 

  def get_vocabs(self): 
    return [i[0] for i in sorted(self.vocabs.items(), key=lambda x: x[1], reverse=True)] 

  def _next_sentence(self):
    sentence_str = ""
    line = self.readline()
    while line:
      if (line.startswith(CoNLLUCorpus.EOL)): break 
      sentence_str += line
      line = self.readline()
    return UDSentence(sentence_str, collapse_rels=self.collapse_rels)
    
  def __getitem__(self,index):
    self.seek(self._sentence_offset[index])
    return self._next_sentence()

  def __iter__(self):
    self.seek(0)
    return self

  # this function does not care about the maximum sentence length max_snt_len
  # if the corpus is scanned by its iterator, all sentences are retrieved
  def __next__(self):
    if self.tell() >= self.size : 
      raise StopIteration
    return self._next_sentence()

  def __index_corpus(self):
    offsets = [] 
    num_sentences = 0

    tagset = defaultdict(set)
    vocabs = {}

    self.seek(0)
    offset = self.tell()

    line = self.readline()
    for snt in self:
      if self.max_snt_len is not None and len(snt) > self.max_snt_len:
        offset = self.tell()
        continue
      offsets.append(offset)
      offset = self.tell()
      for udword in snt:
        for f in udword.dictionary:
          if f not in ['FORM', 'LEMMA', 'HEAD', 'FEATS']: tagset[f].add(udword[f])
          utils.inc_dict_value(vocabs, udword['FORM'])
      num_sentences += 1
      
    return vocabs, offsets, num_sentences, tagset

  def print(self):
    pass


###################################################
class MultiCoNLLUCorpus:
  '''
  This function does not work with tagger.fit(). The reason is that the fit() function is multithreaded but the seek function in CoNNLLUCorpus seem to be problamitic. 
  '''
  def __init__(self, corpora):
    start = 0
    self._corpus_index = {}
    self.vocabs = {}
    self.tagset = {}
    for corpus in corpora:
      self._corpus_index[start] = copy.deepcopy(corpus)
      start += len(corpus)
      self.merg_vocabs(corpus.vocabs)
      self.tagset.update(corpus.tagset)

    self.num_sentences = start

  def merg_vocabs(self, vocabs):
    if self.vocabs:
      key_intersect = list(set(self.vocabs.keys()).intersection(set(vocabs.keys())))
      val_intersect = [self.vocabs[i] + vocabs[i] for i in key_intersect]
      self.vocabs.update(vocabs)
      for key, value in dict(zip(key_intersect, val_intersect)).items():
        self.vocabs[key] = value
    else:
      self.vocabs.update(vocabs)
    return self.vocabs

  def __len__(self): 
    return self.num_sentences

  def __getitem__(self, index):
    for start, corpus in self._corpus_index.items(): 
      if start <= index :
        return corpus[index-start]

    raise IndexError
