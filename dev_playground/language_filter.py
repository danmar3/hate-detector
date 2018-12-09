import os
import bz2
import json
import datetime

def file_tweets(filepath):
    zipfile = bz2.BZ2File(filepath) 
    data = zipfile.read() 
    lines = data.splitlines()
    tweets = []
    for line in lines:
        parsed_json = json.loads(line)
        
        try:
            # if parsed_json['place'] and parsed_json['place']['country_code'] in ['US','GB']:  #US, GB, ES, MX
            #     tweet = parsed_json['text']
            #     tweet = str(tweet.encode('ascii', 'ignore').decode())
            #     tweets.append(tweet)
            tweet = parsed_json['text']
            tweet = str(tweet.encode('ascii', 'ignore').decode())
            en_words = ['is','the','to','of','and','why','hate','be','that','have','it','for','not','on','at','with']
            es_words = '''
                el los del las por una para
                como mas pero sus sobre este entre cuando esta
                ser tambien fue habia 
            '''.split()
            es_words = '''
                el los del las por una 
                pero sus sobre este entre cuando 
                tambien fue habia 
            '''.split()
            for word in tweet.split():
                if word in es_words: # and not tweet.startswith('RT'):
                    tweet = tweet.replace('\n', '')
                    tweet = tweet.replace('\r', '')
                    tweet = tweet + '\n'
                    tweets.append(tweet)
                    break
        except:
            pass
    return tweets

# tweets = file_tweets("2017\\07\\01\\00\\31.json.bz2")
# print(len(tweets))

def walk_tweets(directory):
    count = 0
    with open("tweets.txt", 'w') as f:
    
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                if name.endswith(".bz2"):
                    path = os.path.join(root, name)
                    print(path)
                    tweets = file_tweets(path)
                    f.writelines(tweets)
                    #tweets += file_tweets(path)
                    count += len(tweets)
                    print(count)
                    if count > 10000000:
                        return


start_time = datetime.datetime.now()
walk_tweets("2017")
# tweets = [str(tweet.encode('ascii', 'ignore').decode()) + '\n' for tweet in tweets]
# for tweet in tweets:
#     print(tweet)



print("runtime", datetime.datetime.now() - start_time)