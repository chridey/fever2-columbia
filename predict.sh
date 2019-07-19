#!/usr/bin/env bash

default_cuda_device=0

echo "start document candidate retrieval"
python -m retrieve $1 /tmp/ir.candidates.$(basename $1) \
    --config configs/system_config.json

echo "start document re-ranking"
python -m eval http://www1.cs.columbia.edu/nlp/fever/page_model.tar.gz  \
    /tmp/ir.candidates.$(basename $1) \
    --log /tmp/ir.$(basename $1) \
    --predicted-pages --merge-google \
    --cuda-device ${CUDA_DEVICE:-$default_cuda_device} \
    
echo "start prediction"
python -m eval http://www1.cs.columbia.edu/nlp/fever/state_model.tar.gz  \
    /tmp/ir.$(basename $1) \
    --log $2 \
    --cuda-device ${CUDA_DEVICE:-$default_cuda_device} \

