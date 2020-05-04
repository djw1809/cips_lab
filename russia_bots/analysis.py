
import pandas as pd
import numpy
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
#%%

#%% load data
raw_data = pd.read_csv('all_data.csv')
len(raw_data)
raw_likes = raw_data.loc[:, 'likes']
raw_shares = raw_data.loc[:, 'shares']
raw_bots = raw_data.loc[:, 'bots']
raw_data['likes_bot_percent'] = raw_bots.divide(raw_likes)
raw_data['shares_bots_percent'] = raw_bots.divide(raw_shares)
len(raw_data['likes_bot_percent'][raw_data['likes_bot_percent'] > .9])
len(raw_data['shares_bots_percent'][raw_data['shares_bots_percent'] > .9])
raw_data['likes'].value_counts()
raw_data['shares'].value_counts()
len(raw_data[(raw_data['likes'] == 0) & (raw_data['bots'] > 5)])
max(raw_data['shares'].unique())
max(raw_data['likes'].unique())
len(raw_data['shares'][raw_data['shares'] > 1000000])
#%% remove all outliers
data = raw_data[(abs(stats.zscore(raw_data)) < 3.5).all(axis = 1)]
len(data)
likes = data.loc[:, 'likes']
shares = data.loc[:, 'shares']
bots = data.loc[:, 'bots']
bots.unique()

#%% lower bound values/mean/max
bot_values = []
min_shares = []
max_shares = []
mean_shares = []
data.loc[:, 'shares'][data.loc[:, 'bots'] == 1].mean()
max(data.loc[:, 'bots'])
data.loc[:, 'bots'].value_counts().index
for bot_value in bots.unique():
    share_counts = data.loc[:, 'shares'][data.loc[:, 'bots'] == bot_value]
    bot_values.append(bot_value)
    min_shares.append(min(share_counts))
    max_shares.append(max(share_counts))
    mean_shares.append(share_counts.mean())


lower_upper_mean_shares = pd.DataFrame({'bot_values': bot_values, 'min_shares': min_shares, 'max_shares': max_shares, 'mean_shares': mean_shares})
lower_upper_mean_shares

#%% remove outliers

lower_upper_mean_shares = lower_upper_mean_shares[(abs(stats.zscore(lower_upper_mean_shares))< 1).all(axis = 1)]
min_line = lower_upper_mean_shares[lower_upper_mean_shares['min_shares'] == lower_upper_mean_shares['bot_values']]
len(min_line)

lower_upper_mean_shares['bot_min_percent'] = lower_upper_mean_shares['bot_values'].divide(lower_upper_mean_shares['min_shares'])
len(lower_upper_mean_shares)
len(lower_upper_mean_shares[lower_upper_mean_shares['bot_min_percent'] > .9])

len(lower_upper_mean_shares)
bot_values = lower_upper_mean_shares.loc[:, 'bot_values']
min_shares = lower_upper_mean_shares.loc[:, 'min_shares']
max_shares = lower_upper_mean_shares.loc[:, 'max_shares']
mean_shares = lower_upper_mean_shares.loc[:, 'mean_shares']
#%%



#%% set up plots
fig1, (ax11, ax12) = plt.subplots(1,2) #bots v likes raw
fig1.suptitle("bots v. likes raw")
fig2, (ax21, ax22) = plt.subplots(1,2) #bots v shares raw
fig2.suptitle("bots v. shares raw")
fig3, (ax31, ax32) = plt.subplots(1,2) #bots v likes all outliers
fig3.suptitle("bots v. likes outliers")
fig4, (ax41, ax42) = plt.subplots(1,2) #bots v shares all outliers
fig4.suptitle("bots v. shares outliers")

ax11.set_xlabel('bots')
ax11.set_ylabel('likes')
ax12.set_xlabel('bots')
ax12.set_ylabel('likes')
ax12.set_yscale("log", basey = 10)

ax21.set_xlabel('bots')
ax21.set_ylabel('shares')
ax22.set_xlabel('bots')
ax22.set_ylabel('shares')
ax22.set_yscale("log", basey = 10)

ax31.set_xlabel('bots')
ax31.set_ylabel('likes')
ax32.set_xlabel('bots')
ax32.set_ylabel('likes')
ax32.set_yscale("log", basey = 10)

ax41.set_xlabel('bots')
ax41.set_ylabel('shares')
ax42.set_xlabel('bots')
ax42.set_ylabel('shares')
ax42.set_yscale("log", basey = 10)
#ax42.set_xscale("log", basex = 10)
#%%

fig5, ax5 = plt.subplots()
fig5.suptitle("bot vs. min_shares")
fig6, ax6 = plt.subplots()
fig6.suptitle("bot vs. max_shares")
fig7, ax7 = plt.subplots()
fig7.suptitle("bot vs. mean shares")

ax5.set_xlabel('bots')
ax5.set_ylabel('min_shares')
ax6.set_xlabel('bots')
ax6.set_ylabel('max_shares')
ax7.set_xlabel('bots')
ax7.set_ylabel('mean shares')


#%% plot
ax11.scatter(raw_bots, raw_likes)
ax12.scatter(raw_bots, raw_likes)
ax21.scatter(raw_bots, raw_shares)
ax22.scatter(raw_bots, raw_shares)
ax31.scatter(bots, likes)
ax32.scatter(bots, likes)
ax41.scatter(bots, shares)
ax42.scatter(bots, shares)
fig1
fig2
fig3
fig4

#%%
ax5.scatter(bot_values, min_shares)
ax6.scatter(bot_values, max_shares)
ax7.scatter(bot_values, mean_shares)

fig5
fig6
fig7

#%%saving
fig3.savefig("bot_v_likes_outliers.pdf")
fig4.savefig("bot_v_shares_outliers_loglog.pdf")
