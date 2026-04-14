import streamlit as st

st.title("Medical VQA System")
st.write("Alternative interface for demonstration purposes.")
uploaded_file = st.file_uploader("Upload Medical Image")
if uploaded_file is not None:
    st.image(uploaded_file)
    question = st.text_input("Ask a question about this image")
    if st.button("Predict"):
        st.write("Mock Output: Normal")
