"""
Filters raw twitter data into a language-specific corpus
@author: Paul Hudgins (hudginspj@vcu.edu)
"""
import os
import bz2
import json
import datetime

CORPUS_FOLDER = 'dataset/corpuses/'

en_words = ['is','the','to','of','and','why','hate','be','that','have','it','for','not','on','at','with']
es_words = '''
    el los del las por una 
    pero sus sobre este entre cuando 
    tambien fue habia 
'''.split()

def file_tweets(filepath, language):
    zipfile = bz2.BZ2File(filepath) 
    data = zipfile.read() 
    lines = data.splitlines()
    tweets = []
    for line in lines:
        parsed_json = json.loads(line)
        
        try:
            tweet = parsed_json['text']
            tweet = str(tweet.encode('ascii', 'ignore').decode())
            if language == 'EN':
                language_words = en_words
            elif language == 'ES':
                language_words = es_words
            for word in tweet.split():
                if word in language_words: # and not tweet.startswith('RT'):
                    tweet = tweet.replace('\n', '')
                    tweet = tweet.replace('\r', '')
                    tweet = tweet + '\n'
                    tweets.append(tweet)
                    break
        except:
            pass
    return tweets


def walk_tweets(outfile, language, limit):
    count = 0
    with open(CORPUS_FOLDER + outfile, 'w') as out_f:
    
        for root, dirs, files in os.walk(CORPUS_FOLDER, topdown=False):
            for name in files:
                if name.endswith(".bz2"):
                    path = os.path.join(root, name)
                    print(path)
                    tweets = file_tweets(path, language)
                    out_f.writelines(tweets)
                    #tweets += file_tweets(path)
                    count += len(tweets)
                    print(count)
                    if count > limit:
                        print("Saved", outfile)
                        return
    print("Saved", outfile)


if __name__ == "__main__":
    start_time = datetime.datetime.now()

    walk_tweets("corpus_en.txt", "EN", 10000000)
    print("runtime", datetime.datetime.now() - start_time)

    walk_tweets("corpus_es.txt", "ES", 10000000)
    print("runtime", datetime.datetime.now() - start_time)