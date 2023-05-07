import streamlit as st
import json
import requests

st.title("Kidney Stones Machine Learning Algorithm 🗿")

st.write("Select the patient's urine details with the sliders below:")
gravity = st.slider("Gravity (Density)", 1.008, 1.025, 1.008, 0.001, format="%.3f")
ph = st.slider("pH", 5., 7., 5., 0.01)
osmo = st.slider("Osmo (mOsm)", 250, 1000, 250)
cond = st.slider("Conductivity (mMoh)", 9., 35., 9., 0.1, format="%.1f")
urea = st.slider("Urea (mmol/L)", 75, 600, 75)
calc = st.slider("Calcium (mmol/L)", 0.2, 12., 0.2, 0.01)


# converting the inputs into a json format
inputs = {"gravity": gravity,
          "ph": ph,
          "osmo": osmo,
          "cond": cond,
          "urea": urea,
          "calc": calc}

# when the user clicks on button it will fetch the API
if st.button('Click to kiss your cat 👸🏼'):
    response = requests.post(url="http://127.0.0.1:8000/inference",
                        data=json.dumps(inputs))
    st.subheader(response.text[1:-1])
