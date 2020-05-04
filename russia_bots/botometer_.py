import botometer
import pandas as pd
import numpy as np

#%%

rapid_api_key = 'c2c55e8fd3msh1f8176a1ab668cfp17c7d4jsnc00a09cca80a'

twitter_app_auth = {'consumer_key': 'SLxgPSejspzWjR5VSAKsw5sQh',
                    'consumer_secret': 'sLua9LXWQbN7Wtgw0jdhvPFbaAABfhTEO02ugngsNTHpP3D7EV',
                    'access_token': '907413732384370688-KWSLrewCX6WcWGlPhygqGYZ5G0DKeG0',
                    'access_token_secret': '6MuplKaB8hxfKwHpxqjLLsj9Kwda56yFIxzLHp8xUk3lA'}

bom = botometer.Botometer(wait_on_ratelimit = True, rapidapi_key = rapid_api_key, **twitter_app_auth)
#%%
def compute_botometer_scores(accounts, rapid_key = rapid_api_key, twitter_credentials = twitter_app_auth):
    '''returns a dict of form, {account : botometer score} give a list of form [accounts]'''
    output = {}
    for account, result in botometer.check_accounts_in(accounts):
        score = results['scores']['english']
        output[account] = score

    return output
#%%

def agreement_threshold(data, threshhold):

    agree_bots = data[(data['type'] == 'Bot') & (data['relevant_score'] >= threshhold)]
    agree_users = data[(data['type']  == 'User') & (data['relevant_score'] < threshhold)]

    agree_percent = np.divide(len(agree_bots) + len(agree_users), 60)

    return agree_bots, agree_users, agree_percent
 #%%

sheet1 = pd.read_csv('botometer_assesment_sheet1.csv')
sheet2 = pd.read_csv('botometer_assesment_sheet2.csv')
sheet3 = pd.read_csv('botometer_assesment_sheet3.csv')

all_sheets = sheet1.append(sheet2.append(sheet3))

alg_bots = all_sheets[all_sheets['type'] == 'Bot']
alg_users = all_sheets[all_sheets['type'] == 'User']
alg_suspects = all_sheets[all_sheets['type'] == 'Suspect']

len(alg_suspects[alg_suspects['relevant_score'] >= .7])
len(alg_bots[alg_bots['relevant_score'] >= .7])
len(alg_users[alg_users['relevant_score'] < .7])


agreement_threshold(all_sheets, .3)[2]
agreement_threshold(all_sheets, .4)[2]
agreement_threshold(all_sheets, .5)[2]
agreement_threshold(all_sheets, .6)[2]
agreement_threshold(all_sheets, .7)[2]
agreement_threshold(all_sheets, .8)[2]
agreement_threshold(all_sheets, 1)[2]
