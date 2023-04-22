import streamlit as st
import pandas as pd
import numpy as np
import preprocess_text
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import pickle

# Load the machine learning models and tokenizer objects
tf_idf = pickle.load(open('./Models/tfidf_tokenizer.pkl', 'rb'))
rf_model = pickle.load(open('./Models/random_forest.pkl', 'rb'))
tokenizer = pickle.load(open('./Models/tf_tokenizer.pkl', 'rb'))
model = load_model('./Models/lstm.h5', compile=False)

# Define the classification function
def classify_text(text):
    new_text = preprocess_text.clean_text(text)
    vec = tf_idf.transform([new_text])
    ml_pred = rf_model.predict(vec)
    ml_pred = int(ml_pred[0])
    if ml_pred == 1:
        return "Message is a Terrorism ideation"
    elif ml_pred == 0:
        return "Message is not related to terrorism ideation"
    else:
        return "Sorry, can't figure out!"

# Define the Streamlit app
def app():
    st.set_page_config(page_title='Terrorism Ideation Detection')

    st.title('Terrorism Ideation Detection')

    st.markdown('Enter some text to check if it is related to terrorism ideation.')

    input_text = st.text_input('Enter text here', value='')

    if st.button('Classify'):
        result = classify_text(input_text)
        st.write(result)

if __name__ == '__main__':
    app()