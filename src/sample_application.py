import json
from logging.config import dictConfig
from typing import List, Dict

from allennlp.models import load_archive
from allennlp.predictors import Predictor
from fever.api.web_server import fever_web_api

import os
import logging

from readers.reader import FEVERReader
from modeling.esim_rl_ptr_extractor import ESIMRLPtrExtractor
from predictor import Predictor

def my_sample_fever():
    logger = logging.getLogger()
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }},
        'handlers': {'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
            'formatter': 'default'
        }},
        'root': {
            'level': 'INFO',
            'handlers': ['wsgi']
        },
        'allennlp': {
            'level': 'INFO',
            'handlers': ['wsgi']
        },
    })

    logger.info("Columbia FEVER application")
    config = json.load(open(os.getenv("CONFIG_PATH","configs/predict_docker.json")))

    #TODO: insert Tuhin initialization

    logger.info("Load Model from {0}".format(config['model']))        
    archive = load_archive(config['model'], cuda_device=config['cuda_device'])
    
    logger.info("Loading FEVER Reader")
    reader = FEVERReader.from_params(archive.config["dataset_reader"])
    
    predictor = Predictor(archive.model, reader)
            
    # The prediction function that is passed to the web server for FEVER2.0
    def predict(instances):
        documents = getDocsSingle(data,config['api_key'],config['cse_id'])
        predictions = list(predictor.predict(documents, cuda_device=config['cuda_device']))
        return predictions

    return fever_web_api(predict)

