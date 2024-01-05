import glob
import streamlit as st
import plotly.express as px
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('all')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

filepaths = sorted(glob.glob('diary/*.txt'))
analyzer = SentimentIntensityAnalyzer()

neg = []
pos = []

for filepath in filepaths:
    with open(filepath) as file:
        content = file.read()
        scores = analyzer.polarity_scores(content)
        pos.append(scores['pos'])
        neg.append(scores['neg'])

dates = [name.strip('.txt').strip('diary/') for name in filepaths]

st.title('Diary tone')
st.subheader('Positivity')
pos_figure = px.line(x=dates, y=pos, labels={'x': 'Date', 'y': 'Positivity'})
st.plotly_chart(pos_figure)

st.subheader('Negativity')
neg_figure = px.line(x=dates, y=neg, labels={'x': 'Date', 'y': 'Negativity'})

st.plotly_chart(neg_figure)
