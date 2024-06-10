import streamlit as st
from transformers import pipeline

pipe = pipeline(task="text-classification",model="yartyjung/Fake-Review-Detector")

st.title(":rainbow[Review-Detector]")
st.divider()
text = st.text_input("Your :red[suspicious] review here :sunglasses:",value="")

if text is not None:
    predictions = pipe(text)
    if predictions[0]['label'] == 'fake':
        for p in predictions:
            st.subheader(f":red[FAKE]   :blue[{ round(p['score'] * 100, 1)} %]")
    elif predictions[0]['label'] == 'real':
        for p in predictions:
            st.subheader(f":green[REAL]   :blue[{ round(p['score'] * 100, 1)} %]")
    st.divider()
    st.markdown(":red[***disclaimer***   This is a prediction by an _AI_, which might turn out incorrect.]")
    url = "https://huggingface.co/yartyjung/Fake-Review-Detector"
    st.markdown("check out this [link](%s)" % url)