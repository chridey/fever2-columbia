import json
import argparse

from doc.getDocuments import getDocsBatch

parser = argparse.ArgumentParser()
parser.add_argument('in_file', type=str)
parser.add_argument('out_file', type=str)
parser.add_argument('--config')

args = parser.parse_args()

with open(args.config) as f:
    config = json.load(f)

with open(args.in_file, 'w') as outfile:
    for docs in getDocsBatch(args.out_file, config['api_key'], config['cse_id']):
        print(json.dumps(docs, file=outfile))
                            
