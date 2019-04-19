#!/usr/bin/env bash

default_cuda_device=0
root_dir=/local/fever-common

ln -s $root_dir/data data

echo "start document retrieval"
python retrieve.py $1 /tmp/ir.$(basename $1) \
    --config docker_config.json

echo "start prediction"
python eval.py http://www1.cs.columbia.edu/nlp/fever/model.tar.gz  \
    /tmp/ir.$(basename $1) \
    --log /tmp/labels.$(basename $1) \
    --cuda-device ${CUDA_DEVICE:-$default_cuda_device} \
    --silent

echo "prepare submission"
python -m fever.submission.prepare \
    --predicted_labels /tmp/labels.$(basename $1) \
    --predicted_evidence /tmp/labels.$(basename $1) \
    --out_file $2
