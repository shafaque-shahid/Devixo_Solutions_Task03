import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Used Vehicle Price Predictor",
    page_icon="🚘",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("used_vehicle_price_model.joblib")

model = load_model()

st.title("Used Vehicle Price Predictor")
st.write(
    "Estimate a used vehicle's price using a machine learning model. "
    "The result is an estimate, not a professional valuation."
)

st.divider()
st.subheader("Enter Vehicle Details")

# Basic vehicle details
brand = st.text_input("Brand", value="Toyota")
car_model = st.text_input("Model", value="Camry")
model_year = st.number_input(
    "Model Year", min_value=1974, max_value=2026,
    value=2020, step=1
)
mileage = st.number_input(
    "Mileage", min_value=0, max_value=1000000,
    value=50000, step=1000
)

# Additional features used during model training
fuel_type = st.selectbox(
    "Fuel Type",
    ["Gasoline", "Hybrid", "Diesel", "E85 Flex Fuel",
     "Plug-In Hybrid", "Electric", "Other", "Missing"]
)

transmission = st.text_input(
    "Transmission", value="Automatic"
)
exterior_color = st.text_input(
    "Exterior Color", value="Black"
)
interior_color = st.text_input(
    "Interior Color", value="Black"
)
accident = st.selectbox(
    "Accident History",
    ["None reported", "At least 1 accident or damage", "Missing"]
)
clean_title = st.selectbox(
    "Clean Title",
    ["Yes", "No", "Missing"]
)

st.subheader("Engine Details")
engine_liters = st.number_input(
    "Engine Size (liters)", min_value=0.0,
    max_value=15.0, value=2.5, step=0.1
)
horsepower = st.number_input(
    "Horsepower (HP)", min_value=0.0,
    max_value=2000.0, value=200.0, step=10.0
)
cylinders = st.number_input(
    "Number of Cylinders", min_value=0,
    max_value=16, value=4, step=1
)

if st.button("Predict Vehicle Price", type="primary"):
    # Match the exact feature names used during model training
    input_data = pd.DataFrame([{
        "brand": brand,
        "model": car_model,
        "car_age": max(0, 2026 - int(model_year)),
        "milage": mileage,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "ext_col": exterior_color,
        "int_col": interior_color,
        "accident": (
            "None reported" if accident == "None reported"
            else "At least 1 accident or damage"
            if accident == "At least 1 accident or damage"
            else "Missing"
        ),
        "clean_title": (
            "Yes" if clean_title == "Yes"
            else "No" if clean_title == "No"
            else "Missing"
        ),
        "engine_liters": engine_liters if engine_liters > 0 else None,
        "horsepower": horsepower if horsepower > 0 else None,
        "cylinders": cylinders if cylinders > 0 else None
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Vehicle Price: ${prediction:,.2f}")
        st.caption(
            "This estimate depends on the dataset and model limitations. "
            "It should not be treated as a guaranteed market price."
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that the saved model matches the features used by this app."
        )
