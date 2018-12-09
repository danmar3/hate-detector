"""
@author: Paul Hudgins (hudginspj@vcu.edu)
"""
import gensim
import os
import zipfile
import pandas
import numpy
import datetime

def main1():
    print('hello')
    with open('../nlp516/dataset/development/train_en.tsv', 'rb') as file:
                dataset = pandas.read_csv(file, sep='\t')
    lines = []
    documents = []
    labels = []
    for i in range(dataset.shape[0]):
        line = dataset.iloc[i].text
        lines.append(line)
        label = dataset.iloc[i].HS
        labels.append(label)
    
    sample = []
    print("=====positive======")
    for i in range(len(lines)):
        if len(lines[i]) <50 and labels[i]==1:
            sample.append(lines[i])
            #print(1, lines[i])
    print("\n\n\n\n====negative======")
    for i in range(len(lines)):
        if len(lines[i]) <50 and labels[i]==0:
            sample.append(lines[i])
            #print(0, lines[i])
    for x in sample[:10] +["  "]+ sample[-10:]:
        print(x)
    # for i in range(len(lines)):
    #     lbl = 1
    #     if labels[i]==lbl and len(lines[i]) <100:
    #             if 'is' in str(line).lower():
    #                 print(lbl, lines[i])
        
print('test', ('is' in 'wall'))
main1()
