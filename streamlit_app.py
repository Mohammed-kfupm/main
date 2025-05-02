
import os, glob, joblib, numpy as np, streamlit as st
from textblob import TextBlob
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title='Spam Detection App',
    layout='wide',
    initial_sidebar_state='expanded'
)

# --- Load Resources ---
@st.cache_resource
def load_vectorizer():
    if os.path.isfile('tfidf_vectorizer.pkl'):
        return joblib.load('tfidf_vectorizer.pkl')
    candidates = glob.glob('*vectorizer*.pkl')
    if not candidates:
        st.error("⚠️ No vectorizer found. Place 'tfidf_vectorizer.pkl' here.")
        return None
    return joblib.load(max(candidates, key=os.path.getmtime))

@st.cache_resource
def load_models():
    desired = {
        'naive_bayes_spam_model.pkl':       'Naive Bayes',
        'knn_spam_model.pkl':               'KNN',
        'decision_tree_spam_model.pkl':     'Decision Tree',
        'neural_net_spam_model.pkl':        'Neural Net',
        'weighted_ensemble_spam_model.pkl': 'Weighted Ensemble'
    }
    mdl_dict = {}
    for fname, display in desired.items():
        if os.path.isfile(fname):
            mdl_dict[display] = joblib.load(fname)
    if not mdl_dict:
        st.error("⚠️ Models not found. Place the five PKLs here.")
    return mdl_dict

vectorizer = load_vectorizer()
models     = load_models()

CLASS_MAP = {0: 'Spam', 1: 'Valid'}

def predict_email(text, mdl, threshold=0.5):
    vec = vectorizer.transform([text])
    if hasattr(mdl, 'predict_proba'):
        prob = mdl.predict_proba(vec)[0][1]
    else:
        df_val = mdl.decision_function(vec)
        prob = 1 / (1 + np.exp(-float(df_val)))
    label = CLASS_MAP[int(prob >= threshold)]
    return label, prob

# --- UI ---
st.title('📨 Spam Detection App Dashboard')
st.sidebar.header('⚙️ Settings')
show_sentiment = st.sidebar.checkbox('Show Sentiment Analysis', True)
threshold      = st.sidebar.slider('Spam Threshold', 0.0, 1.0, 0.5, 0.01)
user_input     = st.text_area('✉️ Message to classify', height=250)

if st.sidebar.button('🔍 Classify'):
    if not user_input:
        st.sidebar.warning('Enter text to classify.')
    elif vectorizer is None or not models:
        st.sidebar.error('Missing vectorizer or model files.')
    else:
        with st.spinner('Analyzing...'):
            if show_sentiment:
                pol = TextBlob(user_input).sentiment.polarity
                st.metric('🧠 Sentiment Polarity', f'{pol:.2f}')
            st.subheader('🛡️ Predictions')
            cols = st.columns(2)
            for i, (name, mdl) in enumerate(models.items()):
                lbl, prob = predict_email(user_input, mdl, threshold)
                cols[i % 2].metric(label=name, value=lbl, delta=f'{prob:.1%}')
        st.success('✅ Done!')
