import streamlit as st
import json
import requests

st.title("Kidney Stones Machine Learning Algorithm 🗿")
st.write("#")

st.write("Select the patient's urine details with the sliders below:")
st.write("#")

# Min and max sliders values are chosen to the naked eye with the training dataset
gravity = st.slider("**Gravity (Density of the urine)**", 1.008, 1.025, 1.008, 0.001, format="%.3f")
st.write()
ph = st.slider("**pH**", 5., 7., 5., 0.01)
st.write()
osmo = st.slider("**Osmolarity (mOsm)**", 250, 1000, 250)
st.write()
cond = st.slider("**Conductivity (mMoh)**", 9., 35., 9., 0.1, format="%.1f")
st.write()
urea = st.slider("**Urea (mmol/L)**", 75, 600, 75)
st.write()
calc = st.slider("**Calcium (mmol/L)**", 0.2, 12., 0.2, 0.01)
st.write("#")

# converting the inputs into a json format
inputs = {"gravity": gravity,
          "ph": ph,
          "osmo": osmo,
          "cond": cond,
          "urea": urea,
          "calc": calc}

# when the user clicks on button it will fetch the API
if st.button('Click to diagnose 👨‍⚕️'):
    response = requests.post(url="http://127.0.0.1:8000/myapp",
                             data=json.dumps(inputs))
    if response.status_code == 200:
        st.subheader(response.text[1:-1])
    else:
        st.subheader(response.text)
