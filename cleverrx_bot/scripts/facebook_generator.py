import torch


class FacebookGenerator():
    '''class to easily generate on fly/pass generations to other applications'''
    def __init__(self, model_path, model_class, keyword_list, tokenizer):
        '''model_path (str): path to folder where model is stored
           model_class (cls): the class of the model instance to be used.  Needs a load method that can accept a folder path and find the model weights in that folder.  Should also have a generate method.
           keyword_list (iterable): a list of keywords that the model was trained on'''
        self.model = model_class.load(model_path)
        self.keywords = keyword_list
        self.tokenizer = tokenizer

    def process_incoming_comment(self, comment):
        '''needs to take as input a comment and return as output the keywords in the keyword list that are in the comment'''

    def generate(prompt, max_length, top_k = None, top_p = None, num_return_sequences = 5, min_keep = 1, filter_value = -float("Inf")):
        output = self.model.generate(self.tokenizer, prompt, max_length, top_k, top_p, num_return_sequences, min_keep, filter_value)
        return output

    def process_list_of_comments(self, comment_list):
        output = [self.process_incoming_comment(comment) for comment in comment_list]
        return output
