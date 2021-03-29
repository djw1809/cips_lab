import pickle as pkl
import csv
import pandas as pd
import matplotlib.pyplot as plt
import operator

topic_index = pkl.load(open('Data/topics_index_bots_fbgroups2.pkl', 'rb'))
clusters = pkl.load(open('Data/clusters.pkl', 'rb'))
# topic_index = pkl.load(open('data/topics_index.pkl', 'rb'))
print(topic_index)
# print(topic_index['1204645532792238085']['topic_links'][0])
# print(topic_index['1204645532792238085'])

# get the list of all topics
all_topics = []
for tweet_id, everything in topic_index.items():
    all_topic = all_topics.extend(topic_index[tweet_id]['topics'])
all_topics = list(set(all_topics))
# print(all_topics)
# print(len(all_topics))

'''
# This part of code checks the number of unique sentiment in the card_sentiment
card_sentiment = []
for tweet_id, everything in topic_index.items():
    card_sentiment.append(topic_index[tweet_id]['card_sentiment'])
card_sentiment = list(set(card_sentiment))
print(card_sentiment)
print(len(card_sentiment))
'''
# tweet_cardplus_insuneg_list = []
# for topic in all_topics:
#     for tweet_id, everything in topic_index.items():
# #         if topic in topic_index[tweet_id]['topics']:
# #             if topic_index[tweet_id]['card_sentiment'] == '+' and topic_index[tweet_id]['insurance_sentiment'] == '-' and topic_index[tweet_id]['tweet'] not in tweet_cardplus_insuneg_list:
# #                 tweet_cardplus_insuneg_list.append(topic_index[tweet_id]['tweet'])
# #                 print(topic_index[tweet_id])
# #
# # print(tweet_cardplus_insuneg_list)
# # print(len(tweet_cardplus_insuneg_list))
# #
# # with open('tweet_cardplus_insuneg_list.txt','w') as f:
# #     for i, twt in enumerate(tweet_cardplus_insuneg_list):
# #         f.write('%d' % (i+1) +' : ' + twt + '\n' + '\n')
# #     f.close()

topic_overall_sentplus_count = {}
topic_overall_sentneg_count = {}
topic_cardplus_count = {}
topic_cardneg_count = {}
topic_insuplus_count = {}
topic_insuneg_count = {}
topic_medication_count = {}
track_of_tweetid = []
for topic in all_topics:
    for tweet_id, everything in topic_index.items():
        if topic in topic_index[tweet_id]['topics'] and tweet_id not in track_of_tweetid:
            track_of_tweetid.append(tweet_id)
            if topic_index[tweet_id]['overall_sentiment'] == '+':
                try:
                    topic_overall_sentplus_count[topic] += 1
                except:
                    topic_overall_sentplus_count[topic] = 1
            elif topic_index[tweet_id]['overall_sentiment'] == '-':
                try:
                    topic_overall_sentneg_count[topic] += 1
                except:
                    topic_overall_sentneg_count[topic] = 1
            if topic_index[tweet_id]['card_sentiment'] == '+':
                try:
                    topic_cardplus_count[topic] += 1
                except:
                    topic_cardplus_count[topic] = 1
            elif topic_index[tweet_id]['card_sentiment'] == '-':
                try:
                    topic_cardneg_count[topic] += 1
                except:
                    topic_cardneg_count[topic] = 1

            if topic_index[tweet_id]['insurance_sentiment'] == '-':
                try:
                    topic_insuneg_count[topic] += 1
                except:
                    topic_insuneg_count[topic] = 1
            elif topic_index[tweet_id]['insurance_sentiment'] == '+':
                try:
                    topic_insuplus_count[topic] += 1
                except:
                    topic_insuplus_count[topic] = 1
            if len(topic_index[tweet_id]['medication_list']) > 0:
                try:
                    topic_medication_count[topic] += 1
                except:
                    topic_medication_count[topic] = 1



print(topic_overall_sentplus_count)
print(sum(topic_overall_sentplus_count.values()))

print(topic_overall_sentneg_count)
print(sum(topic_overall_sentneg_count.values()))

print(topic_cardplus_count)
print(sum(topic_cardplus_count.values()))

print(topic_cardneg_count)
print(sum(topic_cardneg_count.values()))

print(topic_insuplus_count)
print(sum(topic_insuplus_count.values()))

print(topic_insuneg_count)
print(sum(topic_insuneg_count.values()))

df_topic_overall_sentplus_count = pd.DataFrame(data=topic_overall_sentplus_count, index=[0])
df_topic_overall_sentneg_count = pd.DataFrame(data=topic_overall_sentneg_count, index=[0])
df_topic_cardplus_count = pd.DataFrame(data=topic_cardplus_count, index=[0])
df_topic_cardneg_count = pd.DataFrame(data=topic_cardneg_count, index=[0])
df_topic_insuplus_count = pd.DataFrame(data=topic_insuplus_count, index=[0])
df_topic_insuneg_count = pd.DataFrame(data=topic_insuneg_count, index=[0])
df_topic_medication_count = pd.DataFrame(data=topic_medication_count, index=[0])

out_put_name = "fbgroups_"
with pd.ExcelWriter(out_put_name+'frequency.xlsx') as writer:
    df_topic_overall_sentplus_count.T.to_excel(writer, sheet_name='Overall_sentplus_count', header=['count'])
    df_topic_overall_sentneg_count.T.to_excel(writer, sheet_name='Overall_sentneg_count', header=['count'])
    df_topic_cardplus_count.T.to_excel(writer, sheet_name='cardplus_count', header=['count'])
    df_topic_cardneg_count.T.to_excel(writer, sheet_name='cardneg_count', header=['count'])
    df_topic_insuplus_count.T.to_excel(writer, sheet_name='insuplus_count', header=['count'])
    df_topic_insuneg_count.T.to_excel(writer, sheet_name='insuneg_count',header=['count'])
    df_topic_medication_count.T.to_excel(writer, sheet_name='medication_count',header=['count'])




#
# with open('card_count.csv', 'w') as f:
#     f.write("%s,%s,%s\n" % ("Topic", "Topic_cardplus_count", "Topic_cardneg_cout"))
#     for topic in all_topics:
#         if topic in topic_cardplus_count.keys():
#             val_countplus = topic_cardplus_count[topic]
#         else:
#             val_countplus = 0
#
#         if topic in topic_cardneg_count.keys():
#             val_countneg = topic_cardneg_count[topic]
#         else:
#             val_countneg = 0
#
#         f.write("%s,%s,%s\n" % (topic, val_countplus, val_countneg))
#     f.close()
#

'''
with open('topic_countplus.csv', 'w') as f:
    for key in topic_countplus.keys():
        f.write("%s,%s\n" % (key, topic_countplus[key]))
    f.close()
'''
