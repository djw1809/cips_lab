import requests
import json
import pandas as pd
import time
#%%

def get_tweets(limit):
    url = 'http://10.218.106.4:8080/fetch/topActiveTweets'
    payload = {'limit':limit, 'apikey':'dylankey'}
    r = requests.get(url, params = payload)
    if r.json()['status'] == 200:
        with open('twitterCommenting-master/tweet_data/'+time.strftime("%a_%d_%b_%Y_%H:%M:%S", time.localtime()), 'w') as file:
            json.dump(r.json()['response'], file)

        return r.json()['response']

    else:
        print('returned status {}'.format(r.json()['status']))
        return r.json()
#%%

def check_tweets(post_list):
    tweet_objects = [json.loads(i) for i in post_list]
    keep_list = []
    for tweet in tweet_objects:

        if 'retweeted_status' in tweet.keys():
            original_tweet = tweet['retweeted_status']
            if original_tweet['truncated']:
                try:
                    print(original_tweet['extended_tweet']['full_text'])
                except:
                    print(original_tweet['text'])
            else:
                print(original_tweet['text'])


        elif 'quoted_status' in tweet.keys():
            quoted_tweet = tweet['quoted_status']
            if quoted_tweet['truncated']:
                try:
                    print(quoted_tweet['extended_tweet']['full_text'])
                except:
                    print(quoted_tweet['text'])

            else:
                print(quoted_tweet['text'])


        elif tweet['truncated']:
            try:
                print(tweet['extended_tweet']['full_text'])
            except:
                print(tweet['text'])

        else:
            print(tweet['text'])

        keep = input('1 for keep tweet, anything else for discard')
        if keep == '1':
            keep_list.append(tweet)
        else:
            pass

    return keep_list


def create_salvo(keep_list):
    '''need id, username, reply, maybe text'''
    salvo = []
    for tweet in keep_list:
        tweet_dict = {}

        if 'retweeted_status' in tweet.keys():
            original_tweet = tweet['retweeted_status']
            if original_tweet['truncated']:
                try:
                    tweet_dict['text'] = original_tweet['extended_tweet']['full_text']
                except:
                    tweet_dict['text'] = original_tweet['text']

            tweet_dict['username'] = original_tweet['user']['screen_name']
            tweet_dict['id'] = original_tweet['id_str']

        elif 'quoted_status' in tweet.keys():
            original_tweet = tweet['quoted_status']
            if original_tweet['truncated']:
                try:
                    tweet_dict['text'] = original_tweet['extended_tweet']['full_text']
                except:
                    tweet_dict['text'] = original_tweet['text']

            tweet_dict['username'] = original_tweet['user']['screen_name']
            tweet_dict['id'] = original_tweet['id_str']



        elif tweet['truncated']:
            try:
                tweet_dict['text'] = tweet['extended_tweet']['full_text']
            except:
                tweet_dict['text'] = tweet['text']

            tweet_dict['username'] = tweet['user']['screen_name']
            tweet_dict['id'] = tweet['id_str']

        else:
            tweet_dict['text'] = tweet['text']
            tweet_dict['username'] = tweet['user']['screen_name']
            tweet_dict['id'] = tweet['id_str'] 


        print(tweet_dict['text'])
        reply = input('Input a reply to this tweet')

        if reply == '':
            pass

        else:
            tweet_dict['reply'] = reply
            salvo.append(tweet_dict)

    return salvo


def post_salvo(salvo, url):
    output = []
    for tweet in salvo:
        payload = {}
        payload['tid'] = tweet['id']
        payload['message'] = tweet['reply']
        payload['username'] = tweet['username']
        payload['apikey'] = 'dylankey'
        r = requests.post(url, data = payload)
        output.append(r.json())

    return output

#%%
#%% test
