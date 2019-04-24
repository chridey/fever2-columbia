#!/usr/bin/env bash

default_cuda_device=0

echo "start document retrieval"
python retrieve.py $1 /tmp/ir.$(basename $1) \
    --config configs/system_config.json

echo "start prediction"
python eval.py http://www1.cs.columbia.edu/nlp/fever/model.tar.gz  \
    /tmp/ir.$(basename $1) \
    --log $2 \
    --cuda-device ${CUDA_DEVICE:-$default_cuda_device} \

