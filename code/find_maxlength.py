from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import logging
import re
import os
import csv
f = '/Users/liuman/Documents/2/WASSA/data/test.csv'
with open(f,'rb') as myFile:
    # lines=csv.reader(myFile, delimiter='\t')
    lines = myFile.readlines()
    # assert len(lines) == 153383
    label_li, text_li = [], []
    for line in lines:
        line = line.strip()
        if line == '': continue
    	# print(line)
        label, text = line.split('\t', 1)
        label_li.append(label)
        text_li.append(text)
print(len(label_li))
print(len(text_li))
foutputlabel = '/Users/liuman/Documents/2/WASSA/data/test_data.label'
foutputtext = '/Users/liuman/Documents/2/WASSA/data/test_data.text'
with open(foutputtext, 'w') as outputfile:
	for line in text_li:
		outputfile.write(line+'\n')

with open(foutputlabel, 'w') as outputfile:
	for line in label_li:
		outputfile.write(line+'\n')