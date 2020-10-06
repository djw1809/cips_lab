import json
#%%
with open('test_data.json', 'r') as f:
    data1 = json.load(f)

actual_data_1 = [json.loads(i) for i in data1]
len(actual_data_1)

with open('test_data2.json', 'r') as f: 
    data2 = json.load(f)

actual_data_2 = [json.loads(i) for i in data2]
len(actual_data_2)


final_data = actual_data_1 + actual_data_2
len(final_data)

with open('test_data_final.json', 'w') as file:
    json.dump(final_data, file)


with open('test_data_final.json', 'r') as file:
    final_data = json.load(file)


final_data[0].keys()
