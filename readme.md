Code for the paper "DeSePtion: Dual Sequence Prediction and Adversarial Examples for Improved Fact-Checking"

To install the software, run the following commands in this order:
```
pip install spacy==2.1.3
python -m spacy download en
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

# Evaluation
To run the full pipeline for inference, use the following sequence of commands:
```
default_cuda_device=0

python -m retrieve $1 /tmp/ir.candidates.$(basename $1) \
    --config configs/system_config.json

python -m eval page_model.tar.gz  \
    /tmp/ir.candidates.$(basename $1) \
    --log /tmp/ir.$(basename $1) \
    --predicted-pages --merge-google \
    --cuda-device ${CUDA_DEVICE:-$default_cuda_device} \
    
python -m eval state_model.tar.gz  \
    /tmp/ir.$(basename $1) \
    --log $2 \
    --cuda-device ${CUDA_DEVICE:-$default_cuda_device}
```

Prior to running the model, the following files are needed:
1) Pre-trained pointer network models (page_model.tar.gz and state_model.tar.gz) and fine-tuned BERT models (bert-pages.tar.gz and bert-fever-gold.tar.gz). Available upon request.
2) FEVER 1.0 and 2.0 data
3) The version of preprocessed Wikipedia for FEVER (can be downloaded from the organizers here: https://s3-eu-west-1.amazonaws.com/fever.public/wiki_index/fever.db)
4) The TF-IDF index and Google search API key

## Document Retrieval 

### TF-IDF
After installing DrQA, run the following command to build the index:
```
DrQA/scripts/retriever/build_tfidf.py fever.db index/
```
This will create a file called fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz.

### Google Custom Search
* To be able to use this one needs to create a developer API Key
  More information here : https://support.google.com/googleapi/answer/6158862?hl=en
* Custom Search ID 
  More information here : https://support.google.com/customsearch/answer/2649143?hl=en

One of the important factors to consider is the first 100 google search api calls is free.
Then onwards one is charged 5$ for every 1000 search api calls (Limit for one day. You cannot call more than 1000 times using one api-key)
Having google cloud credits will be helpful in such cases (Free signup gives 300$)

# Training
To re-train the entire pipeline, follow the same steps as for evaluation (except for running the eval script). Then, fine-tune the BERT models and train a document ranker and joint sentence and relation model.

# BERT Fine-Tuning
To re-train the BERT fine-tuned models, run the huggingface script (https://github.com/huggingface/pytorch-pretrained-BERT):
```
pytorch-pretrained-BERT/examples/run_classifier.py --data_dir INPUT_DIR --bert_model bert-base-uncased --task_name TASK_NAME --output_dir OUTPUT_DIR --do_train --do_eval --do_lower_case --train_batch_size BATCH_SIZE  --learning_rate LEARNING_RATE --num_train_epochs NUM_EPOCHS
```

TASK_NAME for claim/title pairs is "mrpc" and for sentence/title pairs is "mnli."

The datasets of claim/title pairs and sentence/title pairs are available upon request.


# Pointer Network Training
To re-train the pointer network models, run the following command:
```
train.py CONFIG OUTDIR --cuda-device CUDA_DEVICE
```

Example CONFIG files can be found in configs for document ranking (document_ranker.json) and joint sentence and relation prediction (joint_pointer.json).