# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
from io import StringIO
import sys
import signal
import traceback
import bot_models as models
import subprocess
import flask

import pandas as pd

import transformers
import bot_models as models
#prefix = '/opt/ml/'
model_path = '../ml/model'

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded
    tokenizer = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            #with open() as model:
            cls.model = models.GPT2Model_bagofctrl.load(model_path)


        return cls.model

    @classmethod
    def get_tokenizer(cls):
        if cls.tokenizer == None:
            cls.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

        return cls.tokenizer

    @classmethod
    def predict(cls, model_input, max_length, top_k = None, top_p = None, num_return_sequences = 1):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        model = ScoringService.get_model()
        tokenizer = ScoringService.get_tokenizer()
        generated_sentences = model.generate(tokenizer, model_input, max_length, top_k = top_k, top_p = top_p, num_return_sequences = num_return_sequences)
        return generated_sentences

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/blah', methods=['POST'])
def blah():
    """check how requests are being processed"""
    content_type = flask.request.content_type
    data = flask.request.json
    print(content_type)
    print(data['top_p'])
    return flask.jsonify(content_type = content_type, data = data)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')
    # print(subprocess.run(['ls', '../ml/model'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    # status = 200
    # return flask.Response(response = '\n', status = status)

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'application/json':
        data = flask.request.json #flask.request.data.decode('utf-8')
        #s = StringIO.StringIO(data)
        #data = pd.read_csv(s, header=None)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    #print('Invoked with {} records'.format(data.shape[0]))

    # unpack data and do prediction
    keyword_list = data["keywords"]
    prompt = data["prompt"]
    model_input = (keyword_list, prompt)
    max_len = data["max_len"]
    num_return_sequences = data["num_return_sequences"]

    if "top_k" in data.keys():
        top_k = data["top_k"]
    else:
        top_k = 0

    if "top_p" in data.keys():
        top_p = data["top_p"]
    else:
        top_p = 0

    predictions = ScoringService.predict(model_input, max_len, top_k = top_k, top_p = top_p, num_return_sequences = num_return_sequences)


    # Convert from numpy back to CSV
    #out_write = StringIO.StringIO()
    #json.dump({'results': out}, out_write)#pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    #result = out

    return flask.jsonify(returned_sentences = predictions[0], parameters = {"input": model_input, "max_len": max_len, "num_return_sequences":num_return_sequences, "top_k":top_k, "top_p": top_p})
