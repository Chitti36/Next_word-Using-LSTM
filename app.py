import numpy as np
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences




##load the model
model=load_model('nex_word_lstm.h5')

##load the pickle
with open('tokenizer_pickle','rb') as handle:
    tokenizer=pickle.load(handle)

 
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Truncate if input is longer than expected (n-1)
    if len(token_list) > max_sequence_len - 1:
        token_list = token_list[-(max_sequence_len - 1):]

    # Pad the sequence to match the required length
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    # Predict the next word
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]

    # Convert the index back to the word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return None

##stramlit app

st.title("Next word Prediction with LSTM and Early stopping")
input_text=st.text_input("Enter the sequence of words","To be or not to be")
if st.button("Predict Next Word"):
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f'.... {next_word}')



