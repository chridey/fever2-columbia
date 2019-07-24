import logging
from copy import deepcopy

import numpy as np

from allennlp.models import Model, archive_model, load_archive
from allennlp.nn import util
from allennlp.service.predictors import Predictor as AllenNLPPredictor

from readers.reader import FEVERReader
from modeling.esim_rl_ptr_extractor import ESIMRLPtrExtractor
from handlenumericaldateclaims import isClaimEligibleForDateCalculation,getDateClaimLabel

logger = logging.getLogger()

class Predictor:
    def __init__(self, model_path, cuda_device=-1, predicted_pages=False,
                 merge_google=False, score_format=False, verbose=False):
        logger.info("Load model from {1} on device".format(model_path, cuda_device))
        archive = load_archive(model_path, cuda_device=cuda_device)
        logger.info("Loading FEVER Reader")
        ds_params = archive.config["dataset_reader"]
        ds_params["cuda_device"] = cuda_device
        self.reader = FEVERReader.from_params(ds_params)

        self.open_ie_predictor = AllenNLPPredictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
        
        self.model = archive.model
        self.model.eval()        
        self.reverse_labels = {j:i for i,j in self.reader.label_lookup.items()}

        self.predicted_pages = predicted_pages
        self.merge_google = merge_google
        self.score_format = score_format
        self.verbose = verbose

    def predict(self, data):
        raw_data = deepcopy(list(data))
        
        for idx,item in enumerate(self.reader.read('', data=data, include_metadata=True)):
            predicted_sentences = None
            if item.fields["premise"] is None or item.fields["premise"].sequence_length() == 0:
                cls = "NOT ENOUGH INFO"
            else:
                metadata = item.fields['metadata']
                try:
                    prediction = self.model.forward_on_instance(item)
                except RuntimeError as e:
                    if self.verbose:
                        print(e)
                    prediction = dict(predicted_sentences=[], label_probs=[0,0,1])

                if 'predicted_sentences' in prediction:
                    predicted_sentences = [list(metadata['evidence'][i]) for i in prediction['predicted_sentences']]

                reformulated_claim, flag = isClaimEligibleForDateCalculation(raw_data[idx]["claim"])
                if flag:
                    cls = getDateClaimLabel(reformulated_claim,predicted_sentences,self.reader,self.open_ie_predictor).upper()
                elif "label_sequence_logits" in prediction:
                    cls = self.reverse_labels[int(np.argmax(prediction["label_sequence_logits"].sum(axis=-2)))]
                    if self.verbose:
                        print([self.reverse_labels[int(i)] for i in np.argmax(prediction["label_sequence_logits"], axis=-1)])                
                else:
                    cls = self.reverse_labels[int(np.argmax(prediction["label_probs"]))]

            if self.verbose:
                print(cls)
                print(predicted_sentences)
                print(self.model.get_metrics())

            output = {}
            if self.predicted_pages:
                output.update(raw_data[idx])
                if self.merge_google:
                    predicted_sentences = {i[0] for i in predicted_sentences}
                    predicted_sentences.update(raw_data[idx]['predicted_google'])
                    predicted_sentences = [[i] for i in predicted_sentences]
                output.update({"predicted_pages": predicted_sentences,
                               "predicted_label":cls})
            elif self.score_format:
                output = {"actual":raw_data[idx]['label'],"predicted":cls,
                          "predicted_sentences":predicted_sentences}
            else:
                output = {"actual":raw_data[idx]['label'],"predicted_label":cls,
                          "predicted_evidence":predicted_sentences}

            yield output

    def predict_single(self, item):
        prediction = self.predict([item])
        return list(prediction)[0]
                            

