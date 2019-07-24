import shutil
import os

from allennlp.models import archival

config = 'model_params.json'

import sys

file_dir = sys.argv[1]
weights = sys.argv[2]

if archival.CONFIG_NAME == 'config.json' and os.path.exists(os.path.join(file_dir, config)):
    shutil.copyfile(os.path.join(file_dir, config), os.path.join(file_dir, 'config.json'))

archival.archive_model(file_dir, weights)
        
