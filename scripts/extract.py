from glob import glob
import os
import re
import sys

for file in glob("dialogues/*.retrieval.result"):
  lines = open(file, 'rt').readlines()
  bleu1 = re.match('.*bleu2=([\d.]+)', lines[-1])
  if  bleu1 is None:
    continue
  print (file, bleu1.group(1))
