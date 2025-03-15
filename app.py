import streamlit as st

st.title("AI Brand Search")
st.write("Welcome to the AI-powered brand search tool!")

query = st.text_input("Enter your brand query:")

if query:
    st.write(f"Searching for: {query}")

