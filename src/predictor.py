import numpy as np

class Predictor:
    def __init__(self, model, reader):
        self.model = model
        self.model.eval()        
        self.reader = reader
        self.reverse_labels = {j:i for i,j in reader.label_lookup.items()}
        
    def predict(self, data, cuda_device=-1):
        for item in self.reader._read('', data=data, include_metadata=True):
            predicted_sentences = None
            if item.fields["premise"] is None or item.fields["premise"].sequence_length() == 0:
                cls = "NOT ENOUGH INFO"
            else:
                metadata = [i._metadata for i in item.fields['premise'].field_list]
                try:
                    prediction = self.model.forward_on_instance(item)
                except RuntimeError:
                    prediction = dict(predicted_sentences=[], label_probs=[0,0,1])

                if 'predicted_sentences' in prediction:
                    predicted_sentences = [list(metadata[i]) for i in prediction['predicted_sentences']]

                cls = self.reverse_labels[int(np.argmax(prediction["label_probs"]))]

            yield {"predicted_label": cls, "predicted_evidence": predicted_sentences}

    def predict_single(self, item, cuda_device=-1):
        prediction = self.predict([item], cuda_device)
        return list(prediction)[0]
                            

