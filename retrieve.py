import json
import argparse
from allennlp.service.predictors import Predictor
from doc.getDocuments import getDocsBatch

parser = argparse.ArgumentParser()
parser.add_argument('in_file', type=str)
parser.add_argument('out_file', type=str)
parser.add_argument('--config')

args = parser.parse_args()

ner_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")

with open(args.config) as f:
    config = json.load(f)

with open(args.in_file, 'w') as outfile:
    for docs in getDocsBatch(args.out_file, config['api_key'], config['cse_id'],ner_predictor):
        print(json.dumps(docs, file=outfile))
                            
