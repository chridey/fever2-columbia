import os

from copy import deepcopy
from typing import List, Union, Dict, Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from allennlp.common import Params
from allennlp.common.tee_logger import TeeLogger
from allennlp.data import DataIterator, DatasetReader, Tokenizer, TokenIndexer
try:
    from allennlp.data.dataset import Batch as Dataset
except ImportError:
    from allennlp.data import Dataset

from allennlp.models import Model, archive_model, load_archive
from allennlp.service.predictors import Predictor
from allennlp.training import Trainer

from readers.reader import FEVERReader

import argparse
import logging
import sys
import json
import numpy as np

from modeling.esim_rl_ptr_extractor import ESIMRLPtrExtractor

from allennlp.nn import util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def eval_model(args) -> Model:
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device)

    config = archive.config
    ds_params = config["dataset_reader"]

    model = archive.model
    model.eval()

    reader = FEVERReader.from_params(ds_params)

    logger.info("Reading training data from %s", args.in_file)
    data = reader._read(args.in_file, include_metadata=True)

    reverse_labels = {j:i for i,j in reader.label_lookup.items()}
    
    actual = []
    predicted = []

    if args.log is not None:
        f = open(args.log,"w+")

    print({util.get_device_of(param) for param in model.parameters()})
    
    for item in data:
        predicted_sentences = None
        if item.fields["premise"] is None or item.fields["premise"].sequence_length() == 0:
            cls = "NOT ENOUGH INFO"
        else:
            metadata = [i._metadata for i in item.fields['premise'].field_list]
            try:
                prediction = model.forward_on_instance(item)
            except RuntimeError:
                prediction = dict(predicted_sentences=[], label_probs=[0,0,1])
                
            if 'predicted_sentences' in prediction:
                predicted_sentences = [list(metadata[i]) for i in prediction['predicted_sentences']]
                #print([metadata[i.sequence_index] for i in item.fields['evidence'].field_list if i.sequence_index != -1])            
            cls = reverse_labels[int(np.argmax(prediction["label_probs"]))]
            print(cls)
            print(predicted_sentences)
            print(model.get_metrics())
            
        gold = reverse_labels[item.fields["label"].label]
        actual.append(gold)
        predicted.append(cls)

        if args.log is not None:
            f.write(json.dumps({"actual":gold,"predicted":cls,
                                "predicted_sentences":predicted_sentences})+"\n")

    if args.log is not None:
        f.close()

    print(model.get_metrics())
    print(accuracy_score(actual, predicted))
    print(classification_report(actual, predicted))
    print(confusion_matrix(actual, predicted))

    return model

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('archive_file', type=str, help='/path/to/saved/db.db')
    parser.add_argument('in_file', type=str)
    parser.add_argument('--log', required=False, default=None,  type=str, help='/path/to/saved/db.db')

    parser.add_argument("--cuda-device", type=int, default=-1, help='id of GPU to use (if any)')
    
    args = parser.parse_args()

    eval_model(args)
