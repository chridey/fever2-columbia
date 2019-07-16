import os

from copy import deepcopy
from typing import List, Union, Dict, Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from allennlp.common import Params
from allennlp.common.tee_logger import TeeLogger
from allennlp.data import Vocabulary,DataIterator, DatasetReader, Tokenizer, TokenIndexer
try:
    from allennlp.data.dataset import Batch as Dataset
except ImportError:
    from allennlp.data import Dataset

from allennlp.models import Model, archive_model, load_archive
from allennlp.service.predictors import Predictor
from allennlp.training import Trainer

from readers.reader import FEVERReader
from modeling.utils.random import SimpleRandom

import argparse
import logging
import sys
import json
import numpy as np

from modeling.esim_rl_ptr_extractor import ESIMRLPtrExtractor

from allennlp.nn import util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def train_model(params: Union[Params, Dict[str, Any]],
                cuda_device:int,
                serialization_dir: str,
                filtering: str) -> Model:
    """
    This function can be used as an entry point to running models in AllenNLP
    directly from a JSON specification using a :class:`Driver`. Note that if
    you care about reproducibility, you should avoid running code using Pytorch
    or numpy which affect the reproducibility of your experiment before you
    import and use this function, these libraries rely on random seeds which
    can be set in this function via a JSON specification file. Note that this
    function performs training and will also evaluate the trained model on
    development and test sets if provided in the parameter json.

    Parameters
    ----------
    params: Params, required.
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """

    SimpleRandom.set_seeds()

    os.makedirs(serialization_dir, exist_ok=True)
    try:
        sys.stdout = TeeLogger(os.path.join(serialization_dir, "stdout.log"), sys.stdout, True)  # type: ignore
        sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), sys.stderr, True)  # type: ignore
    except TypeError:
        sys.stdout = TeeLogger(os.path.join(serialization_dir, "stdout.log"), sys.stdout)  # type: ignore
        sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), sys.stderr)  # type: ignore
    handler = logging.FileHandler(os.path.join(serialization_dir, "python_logging.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    serialization_params = deepcopy(params).as_dict(quiet=True)

    with open(os.path.join(serialization_dir, "model_params.json"), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

    # Now we begin assembling the required parts for the Trainer.
    ds_params = params.pop('dataset_reader', {})
    read_settings = ds_params.pop('read_settings', {})
    dataset_reader = FEVERReader.from_params(ds_params)    

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path,
                                     include_metadata=True,
                                     replace_with_gold=read_settings.pop('replace_gold',
                                                                        False),
                                     pad_with_nearest=read_settings.pop('pad_with_nearest',
                                                                        0))

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = dataset_reader.read(validation_data_path,
                                              include_metadata=True)
    else:
        validation_data = None

    vocab_params = params.pop("vocabulary", {})
    dataset = None
    print(dict(vocab_params), 'directory_path' not in vocab_params)
    assert('directory_path' in vocab_params)    
    vocab = Vocabulary.from_params(vocab_params,
                                   dataset)
    print(vocab)
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    model = Model.from_params(vocab=vocab, params=params.pop('model'))
    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)
    
    trainer_params = params.pop("trainer")
    if cuda_device is not None:
        trainer_params["cuda_device"] = cuda_device
    trainer = Trainer.from_params(model,
                                  serialization_dir,
                                  iterator,
                                  train_data,
                                  validation_data,
                                  trainer_params)

    trainer.train()

    # Now tar up results
    archive_model(serialization_dir)

    return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('param_path',
                           type=str,
                           help='path to parameter file describing the model to be trained')

    parser.add_argument("logdir",type=str)

    parser.add_argument("--filtering", type=str, default=None)
    parser.add_argument("--cuda-device", type=int, default=None, help='id of GPU to use (if any)')

    args = parser.parse_args()

    params = Params.from_file(args.param_path)

    train_model(params,args.cuda_device,args.logdir,args.filtering)
