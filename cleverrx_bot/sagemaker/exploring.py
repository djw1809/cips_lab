import json
from io import StringIO
import transformers
import bot_models as b
#%%

with open('test_payload.json', 'rb') as file:
    payload = json.load(file)


payload['keywords']
#%%
dict = {'key': 'value'}
out = StringIO()
json.dump(dict, out)
out.getvalue()
#%%
model = b.GPT2Model_bagofctrl.load('../model')

model = b.GPT2Model_bagofctrl.from_pretrained('../model')
