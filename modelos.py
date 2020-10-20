import streamlit as st
import siqr
import sird

st.beta_set_page_config(
    page_title="Modelos Biomatemáticos",
 	layout="centered",
 	initial_sidebar_state="expanded",
)

model = st.sidebar.selectbox('Seleccionar modelo', ['SIR-D','SIQR'])

if model == 'SIQR':
    siqr.main()

if model == 'SIR-D':
    sird.main()