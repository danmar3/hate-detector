# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from collections import Counter

import random
import operator



with open("t.txt", "rb") as datafile:
    tagged_lines = []
    for line in datafile:
        #print(line.decode())
        words = line.decode().split()
        text = words[1:-3]
        tagged_line = words[-3:]
        tagged_line.append(" ".join(text))
        if tagged_line[0] == '1':
            print(tagged_line)
        tagged_lines.append(tagged_line)


fp = 0
fn = 0
tp = 0
tn = 0

for i in range(10):
    random.shuffle(tagged_lines)
    
    train_data = tagged_lines[:90]
    test_data = tagged_lines[90:]
    
    
    pos_words = []
    neg_words = []
    all_words = []
    for line in train_data:
        all_words += line[3].split()
        if line[0] == '1':
            pos_words += line[3].split()
        else:
            neg_words += line[3].split()
    #print(Counter(pos_words))
    pos_count = Counter(pos_words)
    neg_count = Counter(neg_words)
    
    
    for line in test_data:
        pos_score = 0
        neg_score = 0
        for word in line[3]:
            if word in pos_count:
                pos_score += pos_count[word]
            if word in neg_count:
                neg_score += neg_count[word]
        orig = line[0] == '1'
        pred = False
        if pos_score > neg_score :
            pred = True 
        print(pos_score - neg_score, "pred/orig", pred, orig, line[3])
        
        if pred and orig:
            print("tp")
            tp += 1
        elif pred and not orig:
            print("fp")
            fp += 1
        elif not pred and orig:
            print("fn", pred, orig, (not pred) and orig)
            fn += 1
        elif not pred and not orig:
            print("tn")
            tn += 1
    
    word_scores = {}
    for word in all_words:
        score = 0
        if word in pos_count:
            score += pos_count[word]
        if word in neg_count:
            score -= neg_count[word]
        word_scores[word] = score
    
    
    sorted_scores = sorted(word_scores.items(), key=operator.itemgetter(1))
    print('\n', sorted_scores[:40], sorted_scores[-40:])
print('\ntp', tp, 'fp', fp, 'fn', fn, 'tn', tn, 'accuracy', (tn + tp) / (tp + tn + fp + fn))
   
    
    
 