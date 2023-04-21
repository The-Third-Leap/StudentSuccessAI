import streamlit as st

st.title("Streamlit Tutorial App")
st.write("This is my first Streamlit app")

testButton = st.button("Click me")

if testButton:
    st.write("You clicked me")

like = st.checkbox("Does this app look good?")

if like:
    st.write("Yay! I worked hard to get here")
else:
    st.write("I'm not there yet")

