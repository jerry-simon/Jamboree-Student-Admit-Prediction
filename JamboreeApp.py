import streamlit as st
import pickle
import pandas as pd
import sklearn
import numpy as np

Jamboree_df = pd.read_csv("./Jamboree_Admission.csv")

st.title(
    """
    Jamboree prediction of Admission chance
    """
)

st.dataframe(Jamboree_df.head())

col1, col2, col3 = st.columns(3)

GRE_Score = col1.text_input("GRE" , placeholder="Enter the GRE score here")

TOEFL_Score = col1.text_input("TOEFL" , placeholder="Enter the TOEFL score here")

University_rating = col2.selectbox("Select the university rating", ["1", "2", "3", "4", "5"])

SOP_rating = col2.slider("Select the SOP ratings", 1.0, 5.0, step=0.5)

LOR_rating = col3.slider("Select the LOR ratings", 1.0, 5.0, step=0.5)

CGPA_Score = col3.text_input("CGPA" , placeholder="Enter the CGPA score here")

Research = st.selectbox("Have you completed the research or not",["Yes","No"])


encode_dict = {
    "University_rating": {'1':1,'2':2,'3':3,'4':4,'5':5},
    "Research": {'Yes':1,'No':0}
}

def model_predict(GRE_Score, TOEFL_Score, encoded_University_rating, SOP_rating, LOR_rating, CGPA_Score, encoded_Research):
    GRE_Score = float(GRE_Score)
    TOEFL_Score = float(TOEFL_Score)
    SOP_rating = float(SOP_rating)
    LOR_rating = float(LOR_rating)
    CGPA_Score = float(CGPA_Score)
    encoded_University_rating = int(encoded_University_rating)
    encoded_Research = int(encoded_Research)
    loaded_model = pickle.load(open("./jamboree.sav", "rb"))
    input_features = np.array([[GRE_Score, TOEFL_Score, encoded_University_rating, SOP_rating, LOR_rating, CGPA_Score, encoded_Research]])
    return loaded_model.predict(input_features)

if st.button("Predict"):
    encoded_University_rating = encode_dict["University_rating"][University_rating]
    encoded_Research = encode_dict["Research"][Research]


    Admission_Chance = model_predict(GRE_Score, TOEFL_Score, encoded_University_rating, SOP_rating, LOR_rating, CGPA_Score, encoded_Research)

    st.write(f"Predicted Admission Chance for the candidate: {Admission_Chance}")

