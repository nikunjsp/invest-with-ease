import app1
import app2
import app3
import app4
import streamlit as st
PAGES = {
    "42 Stocks allocation on daily basis": app1,
    "Particular stock price prediction": app2,
    "List of all 42 stocks": app3,
    "How Deep RL works": app4
}
st.sidebar.title('Navigation')
selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
