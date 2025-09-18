import streamlit as st
from streamlit_scroll_to_top import scroll_to_here

if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False

if "scroll_to_header" not in st.session_state:
    st.session_state.scroll_to_header = False

if st.session_state.scroll_to_top:
    scroll_to_here(0, key="top")  # Scroll to the top of the page
    st.session_state.scroll_to_top = False  # Reset the state after scrolling


def scroll():
    st.session_state.scroll_to_top = True


def scrollheader():
    st.session_state.scroll_to_header = True


st.title("Dummy Content")
st.write("Scroll down to see the 'Scroll to Top' button.")
for i in range(50):  # Generate dummy content
    if i == 25:
        if st.session_state.scroll_to_header:
            scroll_to_here(0, key="header")  # Scroll to the top of the page
            st.session_state.scroll_to_header = False  # Reset the state after scrolling
        st.header("Or scroll here")
    st.text(f"Line {i + 1}: This is some dummy content.")

st.button("Scroll to Top", on_click=scroll)
if st.button("Scroll to Top 2"):
    st.session_state.scroll_to_top = True
    st.rerun()

st.button("Scroll to Header", on_click=scrollheader)
if st.button("Scroll to Header 2"):
    st.session_state.scroll_to_header = True
    st.rerun()
