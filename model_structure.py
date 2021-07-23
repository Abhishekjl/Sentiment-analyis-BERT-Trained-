import tensorflow as tf
import transformers
from tensorflow.keras.layers import Input, Dense
import numpy as np
import os 

from transformers import TFBertModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(r'static\sentiment analyser bert\bert-tokenizer', local_files_only = True)
bert = TFBertModel.from_pretrained(r'static\sentiment analyser bert\bert-model', local_files_only = True)
    
print('function loaded')
def create_model():   
    
   

    max_len = 70
   
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    # embeddings = dbert_model(input_ids,attention_mask = input_mask)[0]


    embeddings = bert(input_ids,attention_mask = input_mask)[0] #(0 is the last hidden states,1 means pooler_output)
    out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    out = Dense(128, activation='relu')(out)
    out = tf.keras.layers.Dropout(0.1)(out)
    out = Dense(32,activation = 'relu')(out)

    y = Dense(6,activation = 'sigmoid')(out)
    
    model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
    model.layers[2].trainable = True
    return model, tokenizer




model, tokenizer = create_model() 

model.load_weights('static\sentiment analyser bert\sentiment_weights.h5')

def evaluation(x_val):
    validation = (model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100 )
    validation = np.round(validation[0],2)
    encoded_dict  = {'anger':validation[0],'fear':validation[1], 'joy':validation[2], 
        'love':validation[3], 'sadness':validation[4], 'surprise':validation[5]}

    return encoded_dict



def prediction(input_text):
    x_val = tokenizer(
        text=input_text,
        add_special_tokens=True,
        max_length=70,truncation=True,
        padding='max_length', 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,verbose = True)

    predicted = evaluation(x_val)
    return predicted




          




