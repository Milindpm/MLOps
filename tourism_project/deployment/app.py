
"""
Streamlit Application
Wellness Tourism Package Prediction
"""

import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Wellness Tourism Prediction",
    layout="centered"
)

st.title("Wellness Tourism Package Prediction")

st.write(
    "Enter the customer details below to predict whether "
    "the customer is likely to purchase the tourism package."
)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

MODEL_PATH = "tourism_project/deployment/best_model.pkl"

model = joblib.load(MODEL_PATH)

# ----------------------------------------------------
# User Inputs
# ----------------------------------------------------

Age = st.number_input("Age", 18, 100, 30)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry","Company Invited"]
)

CityTier = st.selectbox(
    "City Tier",
    [1,2,3]
)

Occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Small Business",
        "Large Business",
        "Free Lancer"
    ]
)

Gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number Of Persons Visiting",
    1,
    10,
    2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3,4,5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced"
    ]
)

NumberOfTrips = st.number_input(
    "Number Of Trips",
    0,
    20,
    2
)

Passport = st.selectbox(
    "Passport",
    [0,1]
)

OwnCar = st.selectbox(
    "Own Car",
    [0,1]
)

NumberOfChildrenVisiting = st.number_input(
    "Children Visiting",
    0,
    6,
    0
)

Designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    1000,
    500000,
    30000
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Standard",
        "Deluxe",
        "Super Deluxe",
        "King"
    ]
)

NumberOfFollowups = st.number_input(
    "Number Of Followups",
    0,
    10,
    2
)

DurationOfPitch = st.number_input(
    "Duration Of Pitch",
    0,
    120,
    30
)

# ----------------------------------------------------
# Create DataFrame
# ----------------------------------------------------

input_df = pd.DataFrame({

    "Age":[Age],

    "TypeofContact":[TypeofContact],

    "CityTier":[CityTier],

    "Occupation":[Occupation],

    "Gender":[Gender],

    "NumberOfPersonVisiting":[NumberOfPersonVisiting],

    "PreferredPropertyStar":[PreferredPropertyStar],

    "MaritalStatus":[MaritalStatus],

    "NumberOfTrips":[NumberOfTrips],

    "Passport":[Passport],

    "OwnCar":[OwnCar],

    "NumberOfChildrenVisiting":[NumberOfChildrenVisiting],

    "Designation":[Designation],

    "MonthlyIncome":[MonthlyIncome],

    "PitchSatisfactionScore":[PitchSatisfactionScore],

    "ProductPitched":[ProductPitched],

    "NumberOfFollowups":[NumberOfFollowups],

    "DurationOfPitch":[DurationOfPitch]

})

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

if st.button("Predict"):

    prediction = model.predict(input_df)

    if prediction[0] == 1:

        st.success(
            "Customer is likely to purchase the package."
        )

    else:

        st.error(
            "Customer is unlikely to purchase the package."
        )
