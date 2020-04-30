import os

from copy import deepcopy
from typing import List, Union, Dict, Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import argparse
import logging
import sys
import json
import numpy as np

from predictor import Predictor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def eval_model(args) -> None:

    predictor = Predictor(args.archive_file,
                          args.cuda_device,
                          args.predicted_pages,
                          args.merge_google,
                          args.score_format,
                          args.verbose)
    
    raw_data = []
    with open(args.in_file) as f:
        for line in f:
            raw_data.append(json.loads(line))
    
    actual = []
    predicted = []

    if args.log is not None:
        f = open(args.log,"w+")

    #print({util.get_device_of(param) for param in model.parameters()})

    for output in predictor.predict(raw_data[args.start:args.end]):
        actual.append(output['actual'] if 'actual' in output else output.get('label', 'NOT ENOUGH INFO'))
        predicted.append(output['predicted_label'] if 'predicted_label' in output else output['predicted'])
        if args.log is not None:
            f.write(json.dumps(output)+"\n")
    if args.log is not None:
        f.close()

    if args.verbose:
        print(accuracy_score(actual, predicted))
        print(classification_report(actual, predicted))
        print(confusion_matrix(actual, predicted))

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('archive_file', type=str, help='/path/to/saved/db.db')
    parser.add_argument('in_file', type=str)
    parser.add_argument('--log', required=False, default=None,  type=str, help='/path/to/saved/db.db')

    parser.add_argument("--cuda-device", type=int, default=-1, help='id of GPU to use (if any)')

    parser.add_argument("--score-format", action="store_true", help="use the format required for score.py")
    parser.add_argument("--predicted-pages", action="store_true", help="use the predicted pages format")
    parser.add_argument("--merge-google", action="store_true", help="add all the pages from the predicted_google key")
    parser.add_argument("--verbose", action="store_true", help="add all the pages from the predicted_google key")            
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2**32)    
    
    args = parser.parse_args()

    eval_model(args)
