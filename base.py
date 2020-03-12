import numpy
import pandas
import math
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import sklearn
from pathlib import Path
import re
import yaml
import io
import fastparquet
pandas.set_option('display.max_columns', 500)
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import seaborn
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow
config = tensorflow.ConfigProto(device_count={"CPU": 8})
#keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=config))
from keras_tqdm import TQDMNotebookCallback
from IPython.display import clear_output
from sklearn.metrics import classification_report

def transform_training_dataframe(dataframe:pandas.DataFrame):
    
    ### STAGE 1 - Generate auxiliary structures
    column_name = []
    column_type = []
    for k,v in dataframe.dtypes.items():
        column_name.append(k)
        column_type.append(str(v))
    
    # Assuming that the target column is the first one, we discard it
    column_name = column_name[1:]
    column_type = column_type[1:]
    column_values = dataframe.to_numpy().transpose()
    target_values = column_values[0]
    column_values = column_values[1:]
    
    ### STAGE 2 - Automated layer generation
    embedding_layers = []
    embedding_layers_names = []
    base_input_layers = []
    model_input_layers = []
    layer_transformer = [None]*(len(column_name))

    for i_idx in range(len(column_name)):
        c_name = column_name[i_idx]
        c_type = column_type[i_idx]
        c_values = column_values[i_idx]

        #print("Now exploring ", c_name)
        input_layer = keras.layers.Input(shape=(1,), name=c_name+"_"+c_type)

        if c_type == "category":
            #print("Category type detected. Applying Embedding")
            distinct_values = numpy.unique(c_values)
            distinct_values_len = len(distinct_values)
            #print("Detected", distinct_values_len, "distinct values")

            ## Generate encoder
            encoder = sklearn.preprocessing.OrdinalEncoder()
            resulting_array = encoder.fit_transform(c_values.reshape(-1, 1)).reshape(1,-1)[0]
            column_values[i_idx] = resulting_array
            layer_transformer[i_idx] = encoder

            ## Generate layer
            embedding_layer_name = c_name+"_emb_"+c_type
            embedding_layer = keras.layers.Embedding(distinct_values_len, 10, name=embedding_layer_name)(input_layer)
            embedding_layer = keras.layers.Reshape((10,))(embedding_layer)
            embedding_layers.append(embedding_layer)
            embedding_layers_names.append(embedding_layer_name)

            model_input_layers.append(embedding_layer)
        # if it is continous
        else:
            model_input_layers.append(input_layer)
            #continous_input_layers.append(input_layer)
            #print("Continous variable detected, ignoring embedding.")
            pass
        base_input_layers.append(input_layer)
        #print(" ---- END ---- \n")
        
    ### STAGE 3 - Generate auxiliary transformations
    def _aux_transformer(target:pandas.DataFrame):
        val_column_values = target.to_numpy().transpose()
        val_target_values = val_column_values[0]
        val_column_values = val_column_values[1:]

        for i_idx in range(len(column_name)):
            _c_name = column_name[i_idx]
            _c_type = column_type[i_idx]
            _c_values = val_column_values[i_idx]
            _c_transformer = layer_transformer[i_idx]
            if _c_type == "category":
                _resulting_array = _c_transformer.transform(_c_values.reshape(-1, 1)).reshape(1,-1)[0]
                val_column_values[i_idx] = _resulting_array
            else:
                pass
        return val_column_values, val_target_values
        
        
    ### STAGE 4 - Merge input layers
    merged_layers = None
    if len(model_input_layers) > 1:
        merged_layers = keras.layers.concatenate(model_input_layers)
    else:
        merged_layers = base_input_layers[0]
        
        
    return column_values, target_values, _aux_transformer, base_input_layers, merged_layers, layer_transformer,embedding_layers_names
