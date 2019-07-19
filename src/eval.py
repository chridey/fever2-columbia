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
    '''
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device)

    config = archive.config
    ds_params = config["dataset_reader"]

    model = archive.model
    model.eval()

    ds_params["cuda_device"] = args.cuda_device
    reader = FEVERReader.from_params(ds_params)

            
    logger.info("Reading data from %s", args.in_file)
    data = reader.read(args.in_file, include_metadata=True, start=args.start)

    reverse_labels = {j:i for i,j in reader.label_lookup.items()}
    '''

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

    '''
    for idx,item in enumerate(data):
        predicted_sentences = None
        if item.fields["premise"] is None or item.fields["premise"].sequence_length() == 0:
            cls = "NOT ENOUGH INFO"
        else:
            metadata = item.fields['metadata'] #[i._metadata for i in item.fields['premise'].field_list]
            #try:
            prediction = model.forward_on_instance(item)
            #except RuntimeError as e:
            #    print(e)
            #    prediction = dict(predicted_sentences=[], label_probs=[0,0,1])
                
            if 'predicted_sentences' in prediction:
                predicted_sentences = [list(metadata['evidence'][i]) for i in prediction['predicted_sentences']]
                #print([metadata[i.sequence_index] for i in item.fields['evidence'].field_list if i.sequence_index != -1])

            if "label_sequence_logits" in prediction:
                cls = reverse_labels[int(np.argmax(prediction["label_sequence_logits"].sum(axis=-2)))]
                print([reverse_labels[int(i)] for i in np.argmax(prediction["label_sequence_logits"], axis=-1)])
                
            else:
                cls = reverse_labels[int(np.argmax(prediction["label_probs"]))]
            print(cls)
            print(predicted_sentences)
            print(model.get_metrics())
    '''

    for output in predictor.predict(raw_data[args.start:args.end]):
        actual.append(output['actual'] if 'actual' in output else output['label'])
        predicted.append(output['predicted_label'] if 'predicted_label' in output else output['predicted'])
        '''
        gold = reverse_labels[item.fields["label"].label]
        actual.append(gold)
        predicted.append(cls)
        '''
        if args.log is not None:
            f.write(json.dumps(output)+"\n")
            '''
            if args.predicted_pages:
                if args.merge_google:
                    predicted_sentences = {i[0] for i in predicted_sentences}
                    predicted_sentences.update(raw_data[idx]['predicted_google'])
                    predicted_sentences = [[i] for i in predicted_sentences]
                raw_data[idx].update({"predicted_pages": predicted_sentences})
                f.write(json.dumps(raw_data[idx])+"\n")
            elif args.score_format:
                f.write(json.dumps({"actual":gold,"predicted":cls,
                                    "predicted_sentences":predicted_sentences})+"\n")                
            else:
                f.write(json.dumps({"actual":gold,"predicted_label":cls,
                                    "predicted_evidence":predicted_sentences})+"\n")
            '''
    if args.log is not None:
        f.close()

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
