import json
import argparse
from allennlp.service.predictors import Predictor
from allennlp.models import load_archive
from doc.getDocuments import getDocsBatch, GoogleConfig, getDocumentsFromConstituentParse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument('--config')
    parser.add_argument('--cuda_device', type=int, default=-1)
    
    args = parser.parse_args()
    
    ner_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")

    with open(args.config) as f:
        config = json.load(f)

    google_config = GoogleConfig(**config['retrieval'])
        
    with open(args.out_file, 'w') as outfile:
        for docs in getDocsBatch(args.in_file, google_config, ner_predictor):
            print(json.dumps(docs), file=outfile)
                            
