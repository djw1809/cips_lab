import requests
import json
import pandas as pd
import time
#%%

def get_tweets(limit):
    url = 'http://10.218.106.4:8080/fetch/topActiveTweets'
    payload = {'limit':5, 'apikey':'dylankey'}
    r = requests.get(url, params = payload)
    with open('data/'+time.strftime("%a_%d_%b_%Y_%H:%M:%S", time.localtime()), 'w') as file:
        json.dump(r.json()['response'], file)

    return r.json()
#%%

def check_tweets(post_list):
    tweet_objects = [json.loads(i) for i in post_list]
    keep_list = []
    for tweet in tweet_objects:
        try:
            print(tweet['extended_tweet']['full_text'])
        except:
            print(tweet['text'])

        keep = input('1 for keep tweet, anything else for discard')
        if keep == '1':
            keep_list = keep_list.append(tweet)
        else:
            pass

    return keep_list


def create_salvo(keep_list):
    salvo = []
    for tweet in keep_list:
        tweet_dict = {}
        tweet_dict['id'] = tweet['id_str']

        try:
            tweet_dict['text'] = tweet['extended_tweet']['full_text']
        except:
            tweet_dict['text'] = tweet['text']

        tweet_dict['username'] = tweet['user']['screen_name']

        try:
            tweet_dict['url'] = tweet['entities']['urls'][0]['expanded_url']
        except:
            tweet_dict['url'] = ''

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
        r = requests.post(url, data = payload)
        output.append(r.json())

    return output  




with open('data/Mon_17_Aug_2020_16:22:37', 'r') as file:
    test_data = json.load(file)

test_data_tweet_objects = [json.loads(i) for i in test_data]
test_data_tweet_objects[0]['entities']['urls'][0]['expanded_url']
