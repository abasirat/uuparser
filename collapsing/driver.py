import corpus as CORPUS
import pdb

from os import listdir, mkdir
from os.path import isfile, isdir, join

max_snt_len = 10000
ud_path = '/home/staff/abasirat/Lab/Universal_Dependency_Parsing/ud/ud-treebanks-v2.3'
ut_path = '/home/staff/abasirat/Lab/Universal_Dependency_Parsing/ud/ud-treebanks-v2.3-collpased-keepfunc'

def main():
  for d in listdir(ud_path):
    d_path = join(ud_path, d)
    if isdir(d_path) and d.startswith("UD_"): # and d == 'UD_English-EWT':
      out_dir = join(ut_path, d)

      if not isdir(out_dir):
        mkdir(join(ut_path, d))

      for f in listdir(d_path):
        f_path = join(d_path, f)
        if isfile(f_path):
          if f.endswith("-ud-train.conllu") or f.endswith("-ud-dev.conllu") or f.endswith("-ud-test.conllu"):
            corpus = CORPUS.CoNLLUCorpus(f_path, max_snt_len=max_snt_len, collapse_rels=['cop', 'case', 'aux', 'mark', 'cc', 'det', 'clf'])
            out_path = join(ut_path, d, f)
            print(out_path)
            with open(out_path, 'w') as fp:
              for snt in corpus:
                print(str(snt), file=fp)
                print(file=fp)


if __name__ == '__main__':
  main()

