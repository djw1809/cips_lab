import requests
import json
import pandas as pd
import time
import sys
import traceback
import urllib
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
            else:
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

            else:
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
            url_in = input('Which link should be used anything-access, 1-payless')
            uniqueID = str(int(time.time()))
            if url_in == '1':
                url = 'https://tinyurl.com/y23j5a36'              #'https://www.paylessformeds.us/info/?TW3541122'#+uniqueID #
            else:
                url =  'https://tinyurl.com/y5hebpdr'                  #'https://www.paylessformeds.us/?TW3541122'#+uniqueID
            tweet_dict['reply'] = reply + ' ' + url
            tweet_dict['post_id'] = uniqueID
            salvo.append(tweet_dict)

    return salvo


def post_salvo(salvo, url):
    output = []
    ids = []
    for tweet in salvo:
        payload = {}
        payload['tid'] = tweet['id']
        payload['message'] = tweet['reply']
        payload['username'] = tweet['username']
        payload['apikey'] = 'dylankey'
        r = requests.post(url, data = payload)
        output.append(r.json())
        ids.append(tweet['post_id'])
    return output, ids




class UrlShortenTinyurl:
    URL = "http://tinyurl.com/api-create.php"

    def shorten(self, url_long):
        try:
            url = self.URL + "?" \
                + urllib.parse.urlencode({"url": url_long})
            res = requests.get(url)
            print("STATUS CODE:", res.status_code)
            print("   LONG URL:", url_long)
            print("  SHORT URL:", res.text)
        except Exception as e:
            raise
#%%
#%% test
# url1: 'http://10.218.106.4:8080/twitter/reply'
