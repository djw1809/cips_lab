import torch
import transformers


#system details
# Memory: 15.4 GiB
# Processor: Intel Core i7-10610U CPU @ 1/10GHz x 12
#Graphics: Intil UHD Graphics (Comet Lacke 3x8 GT2)
#Disk: 987.4 GB

#torch version 1.3.1






class ModelAPI():

    def __init__(self, model_file):
        self.model = ModelClass.load(model_file)


    def process_incoming_comment(comment):
        '''generate prompt for reply to post if condition is met'''

        if condition:
            prompt = 'a prompt'

        else:
            prompt = None 

        return prompt


    def generate_(prompt):
        '''if comment meets condition generate a reply with prompt'''
        comment_reply = model.generate(prompt)

        return comment_reply
