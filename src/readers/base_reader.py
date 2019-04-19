import json

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

class BaseReader(DatasetReader):
    def iter_data(self, path:str=None, data:list=None, duplicates:set=None):
        if data is None:
            def iter_file():
                with open(path) as f:
                    for line in f:
                        yield json.loads(line)
            data = iter_file()
            
        for instance in data:
            yield instance
        
