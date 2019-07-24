import os
import logging
import json

from logging.config import dictConfig
from typing import List, Dict

from allennlp.models import load_archive
from allennlp.predictors import Predictor
from fever.api.web_server import fever_web_api
from drqa import retriever

from doc.getDocuments import getDocsSingle, GoogleConfig
from readers.reader import FEVERReader
from modeling.esim_rl_ptr_extractor import ESIMRLPtrExtractor
from predictor import Predictor as ColumbiaPredictor

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
    config = json.load(open(os.getenv("CONFIG_PATH","configs/system_config.json")))

    ner_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")
    google_config = GoogleConfig(**config['retrieval']['google'])
    ranker = retriever.get_class('tfidf')(tfidf_path=config['retrieval']['tfidf']['index'])

    predictors = {}
    for key in ('page_model', 'state_model'):    
        predictors[key] = ColumbiaPredictor(config[key]['path'],
                                            config['cuda_device'],
                                            **config[key])
            
    # The prediction function that is passed to the web server for FEVER2.0
    def predict(instances):
        predictions = getDocsSingle(instances,google_config,ner_predictor,ranker)
        for key in ('page_model', 'state_model'):
            predictions = list(predictors[key].predict(predictions))
        return predictions

    return fever_web_api(predict)

