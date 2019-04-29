# For Doc Retrival one of our components depends on Google Custom Search. 

* To be able to use this one needs to create a developer API Key
  More information here : https://support.google.com/googleapi/answer/6158862?hl=en
* Custom Search ID 


# Sample FEVER2.0 builder docker image

The FEVER2.0 shared task requires builders to submit Docker images (via dockerhub) as part of the competition to allow 
for adversarial evaluation. Images must contain a single script to make predictions on a given input file using their model and host a web server (by installing the [`fever-api`](https://github.com/j6mes/fever-api) pip package) to allow for interactive evaluation as part of the _breaker_ phase of the competition.
 
This repository contains an example submission based on an AllenNLP implementation of the system (see [`fever-allennlp`](https://github.com/j6mes/fever-allennlp)). We go into depth for the following key information:

* [Prediction Script](#prediction-script)
* [Entrypoint](#entrypoint)
* [Web Server](#web-server)
* [Common Data](#common-data)

It can be run with the following commands. The first command creates a dummy container with the shared FEVER data that is used by the submission.

```bash
#Set up the data container (run once on first time)
docker create --name fever-common feverai/common

#Start a server for interactive querying of the FEVER system via the web API on port 5000
docker run --rm --volumes-from fever-common:ro -p 5000:5000 feverai/sample

#Alternatively, make predictions on a batch file and output it to `/out/predictions.jsonl` (set CUDA_DEVICE as appropriate)
docker run --rm --volumes-from fever-common:ro -e CUDA_DEVICE=-1 -v $(pwd):/out feverai/sample ./predict.sh /local/fever-common/data/fever-data/paper_dev.jsonl /out/predictions.jsonl
```

### Shared Resources and Fair Use
The FEVER2.0 submissions will be run in a shared environment where resources will be moderated. We urge participants to ensure that these shared resources are respected.

Tensorflow users are asked to implement per-process GPU memory limits: [see this post](https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory). We will set an environment variable `$TF_GPU_MEMORY_FRACTION` that will be tweaked for all systems in phase 2 of the shared task. 


## Prediction Script
The prediction script should take 2 parameters as input: the path to input file to be predicted and the path the output file to be scored:

An optional `CUDA_DEVICE` environment variable should be set  

```bash
#!/usr/bin/env bash

default_cuda_device=0
root_dir=/local/fever-common


python -m fever.evidence.retrieve \
    --index $root_dir/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
    --database $root_dir/data/fever/fever.db \
    --in-file $1 \
    --out-file /tmp/ir.$(basename $1) \
    --max-page 5 \
    --max-sent 5

python -m allennlp.run predict \
    https://jamesthorne.co.uk/fever/fever-da.tar.gz \
    /tmp/ir.$(basename $1) \
    --output-file /tmp/labels.$(basename $1) \
    --predictor fever \
    --include-package fever.reader \
    --cuda-device ${CUDA_DEVICE:-$default_cuda_device} \
    --silent

python -m fever.submission.prepare \
    --predicted_labels /tmp/labels.$(basename $1) \
    --predicted_evidence /tmp/ir.$(basename $1) \
    --out_file $2

``` 

## Entrypoint
The submission must run a flask web server to allow for interactive evaluation. In our application, the entrypoint is a function called `my_sample_fever` in the module `sample_application` (see `sample_application.py`).
The `my_sample_fever` function is a factory that returns a `fever_web_api` object. 

``` python
from fever.api.web_server import fever_web_api

def my_sample_fever(*args):
    # Set up and initialize model
    ...
    
    # A prediction function that is called by the API
    def baseline_predict(instances):
        predictions = []
        for instance in instances:
            predictions.append(...prediction for instance...)
        return predictions

    return fever_web_api(baseline_predict)
```

Your dockerfile can then use the `waitress-serve` method as the entrypoint. This will start a wsgi server calling your factory method

```dockerfile
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--call", "sample_application:my_sample_fever"]
``` 


## Web Server
The web server is managed by the `fever-api` package. No setup or modification is required by participants. We use the default flask port of `5000` and host a single endpoint on `/predict`. We recommend using a client such as [Postman](https://www.getpostman.com/) to test your application.


```
POST /predict HTTP/1.1
Host: localhost:5000
Content-Type: application/json

{
	"instances":[
	    {"id":0,"claim":"this is a test claim"}, 
	    {"id":1,"claim":"this is another test claim"}, 
	]
}
```

## API
In our sample submission, we present a simple method `baseline_predict` method. 

```python 
   def baseline_predict(instances):
        predictions = []
        for instance in instances:
            ...prediction for instance...
            predictions.append({"predicted_label":"SUPPORTS", 
                                "predicted_evidence": [(Paris,0),(Paris,5)]})
            
        return predictions
```

Inputs: 

 * `instances` - a list of dictionaries containing a `claim` 

Outputs:

 * A list of dictionaries containing `predicted_label` (string in SUPPORTS/REFUTES/NOT ENOUGH INFO) and `predicted_evidence` (list of `(page_name,line_number)` pairs as defined in [`fever-scorer`](https://github.com/sheffieldnlp/fever-scorer).


## Testing the Server

After starting the server using waitress, you can test using the requests library:

```
import requests
import json

url = 'http://0.0.0.0:5000/predict'
data = {"instances": [{"id":0, "claim":"this is a test claim"},{"id":1,"claim":"this is another test claim"}]}
headers = {'Content-Type': 'application/json', 'Accept': 'text/plain'}

r = requests.post(url, data=json.dumps(data), headers=headers)

r.json()
```

```
{u'data': {u'predictions': [{u'predicted_evidence': [[u'Theranos', 1],
     [u'One_Million_Dollar_Paranormal_Challenge', 0],
     [u'This', 0],
     [u'Theranos', 0],
     [u'Theranos', 9]],
    u'predicted_label': u'REFUTES',
    u'request_instance': {u'claim': u'this is a test claim',
     u'evidence': [[u'This', 0],
      [u'Theranos', 0],
      [u'Theranos', 1],
      [u'One_Million_Dollar_Paranormal_Challenge', 1],
      [u'Theranos', 8],
      [u'One_Million_Dollar_Paranormal_Challenge', 0],
      [u'Theranos', 7],
      [u'Theranos', 2],
      [u'Theranos', 3],
      [u'Theranos', 4],
      [u'Theranos', 5],
      [u'Theranos', 6],
      [u'This', 4],
      [u'This', 3],
      [u'Theranos', 10],
      [u'One_Million_Dollar_Paranormal_Challenge', 2],                     
      [u'One_Million_Dollar_Paranormal_Challenge', 3],
      [u'This', 1],
      [u'This', 2],
      [u'Theranos', 9]],
     u'id': 0,
     u'predicted_pages': [[u'Theranos'],
      [u'One_Million_Dollar_Paranormal_Challenge'],
      [u'This']]}},
   {u'predicted_evidence': [[u'Turing_test', 12],
     [u'Turing_test', 10],
     [u'Turing_test', 0],
     [u'Turing_test', 8],
     [u'Turing_test', 16]],
    u'predicted_label': u'NOT ENOUGH INFO',
    u'request_instance': {u'claim': u'this is another test claim',
     u'evidence': [[u'Turing_test', 4],
      [u'Turing_test', 2],
      [u'This', 0],
      [u'Turing_test', 0],
      [u'Turing_test', 10],
      [u'Turing_test', 12],
      [u'Turing_test', 11],
      [u'Turing_test', 5],
      [u'Turing_test', 16],
      [u'Turing_test', 8],
      [u'Isabella_of_France', 15],
      [u'Isabella_of_France', 14],
      [u'Isabella_of_France', 13],
      [u'Isabella_of_France', 12],
      [u'Isabella_of_France', 11],
      [u'Isabella_of_France', 10],
      [u'This', 4],
      [u'Isabella_of_France', 9],
      [u'Isabella_of_France', 8],
      [u'Isabella_of_France', 17],
      [u'Isabella_of_France', 7],
      [u'Isabella_of_France', 6],
      [u'Isabella_of_France', 5],
      [u'Isabella_of_France', 4],
      [u'Isabella_of_France', 3],
      [u'Isabella_of_France', 2],
      [u'Isabella_of_France', 1],
      [u'Isabella_of_France', 16],	
      [u'Isabella_of_France', 22],                            
      [u'Isabella_of_France', 18],
      [u'Turing_test', 9],
      [u'This', 2],
      [u'This', 1],
      [u'Turing_test', 17],
      [u'Turing_test', 15],
      [u'Turing_test', 14],
      [u'Turing_test', 13],
      [u'Turing_test', 7],
      [u'Isabella_of_France', 19],
      [u'Turing_test', 6],
      [u'Turing_test', 3],
      [u'Turing_test', 1],
      [u'This', 3],
      [u'Isabella_of_France', 21],
      [u'Isabella_of_France', 20],
      [u'Isabella_of_France', 0]],
     u'id': 1,
     u'predicted_pages': [[u'Isabella_of_France'],
      [u'Turing_test'],
      [u'This']]}}]},
 u'result': u'success'}
```
  
