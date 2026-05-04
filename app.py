import streamlit as st
import pandas as pd
from patsy import dmatrix
import pickle

# Load models
with open('decision_tree_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('naive_bayes_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

st.title("In-Vehicle Coupon Recommendation System")

st.write("Predict whether a driver will accept a coupon based on various features.")

# Input features
destination = st.selectbox("Destination", ['Home', 'No Urgent Place', 'Work'])
passanger = st.selectbox("Passenger", ['Alone', 'Friend(s)', 'Kid(s)', 'Partner'])
weather = st.selectbox("Weather", ['Rainy', 'Snowy', 'Sunny'])
temperature = st.selectbox("Temperature", [30, 55, 80])
time = st.selectbox("Time", ['10AM', '10PM', '2PM', '6PM', '7AM'])
coupon = st.selectbox("Coupon", ['Bar', 'Carry out & Take away', 'Coffee House', 'Restaurant(20-50)', 'Restaurant(<20)'])
expiration = st.selectbox("Expiration", ['1d', '2h'])
gender = st.selectbox("Gender", ['Female', 'Male'])
age = st.selectbox("Age", ['21', '26', '31', '36', '41', '46', '50plus', 'below21'])
maritalStatus = st.selectbox("Marital Status", ['Divorced', 'Married partner', 'Single', 'Unmarried partner', 'Widowed'])
has_children = st.selectbox("Has Children", [0, 1])
education = st.selectbox("Education", ['Associates degree', 'Bachelors degree', 'Graduate degree (Masters or Doctorate)', 'High School Graduate', 'Some High School', 'Some college - no degree'])
occupation = st.selectbox("Occupation", ['Architecture & Engineering', 'Arts Design Entertainment Sports & Media', 'Building & Grounds Cleaning & Maintenance', 'Business & Financial', 'Community & Social Services', 'Computer & Mathematical', 'Construction & Extraction', 'Education&Training&Library', 'Farming Fishing & Forestry', 'Food Preparation & Serving Related', 'Healthcare Practitioners & Technical', 'Healthcare Support', 'Installation Maintenance & Repair', 'Legal', 'Life Physical Social Science', 'Management', 'Office & Administrative Support', 'Personal Care & Service', 'Production Occupations', 'Protective Service', 'Retired', 'Sales & Related', 'Student', 'Transportation & Material Moving', 'Unemployed'])
income = st.selectbox("Income", ['$100000 or More', '$12500 - $24999', '$25000 - $37499', '$37500 - $49999', '$50000 - $62499', '$62500 - $74999', '$75000 - $87499', '$87500 - $99999', 'Less than $12500'])
Bar = st.selectbox("Bar Visits", ['1~3', '4~8', 'gt8', 'less1', 'never'])
CoffeeHouse = st.selectbox("Coffee House Visits", ['1~3', '4~8', 'gt8', 'less1', 'never'])
CarryAway = st.selectbox("Carry Away Visits", ['1~3', '4~8', 'gt8', 'less1', 'never'])
RestaurantLessThan20 = st.selectbox("Restaurant <20 Visits", ['1~3', '4~8', 'gt8', 'less1', 'never'])
Restaurant20To50 = st.selectbox("Restaurant 20-50 Visits", ['1~3', '4~8', 'gt8', 'less1', 'never'])
toCoupon_GEQ5min = 1  # Always 1
toCoupon_GEQ15min = st.selectbox("To Coupon >=15 min", [0, 1])
toCoupon_GEQ25min = st.selectbox("To Coupon >=25 min", [0, 1])
direction_same = st.selectbox("Direction Same", [0, 1])
direction_opp = st.selectbox("Direction Opposite", [0, 1])

# Create DataFrame
data = {
    'destination': [destination],
    'passanger': [passanger],
    'weather': [weather],
    'temperature': [temperature],
    'time': [time],
    'coupon': [coupon],
    'expiration': [expiration],
    'gender': [gender],
    'age': [age],
    'maritalStatus': [maritalStatus],
    'has_children': [has_children],
    'education': [education],
    'occupation': [occupation],
    'income': [income],
    'Bar': [Bar],
    'CoffeeHouse': [CoffeeHouse],
    'CarryAway': [CarryAway],
    'RestaurantLessThan20': [RestaurantLessThan20],
    'Restaurant20To50': [Restaurant20To50],
    'toCoupon_GEQ5min': [toCoupon_GEQ5min],
    'toCoupon_GEQ15min': [toCoupon_GEQ15min],
    'toCoupon_GEQ25min': [toCoupon_GEQ25min],
    'direction_same': [direction_same],
    'direction_opp': [direction_opp]
}

df_input = pd.DataFrame(data)

# Create design matrix
X_input = dmatrix('0 + destination + passanger + weather + temperature + time + coupon + \
 expiration + gender + age + maritalStatus + has_children + education + \
 occupation + income + Bar + CoffeeHouse + CarryAway + \
 RestaurantLessThan20 + Restaurant20To50 + toCoupon_GEQ5min + \
 toCoupon_GEQ15min + toCoupon_GEQ25min + direction_same + direction_opp', df_input, return_type='dataframe')

# Ensure columns match
X_input = X_input.reindex(columns=feature_names, fill_value=0)

# Predictions
dt_pred = dt_model.predict(X_input)[0]
rf_pred = rf_model.predict(X_input)[0]
nb_pred = nb_model.predict(X_input)[0]

st.subheader("Predictions")
st.write(f"Decision Tree: {'Accept' if dt_pred == 1 else 'Reject'}")
st.write(f"Random Forest: {'Accept' if rf_pred == 1 else 'Reject'}")
st.write(f"Naive Bayes: {'Accept' if nb_pred == 1 else 'Reject'}")

st.write("Note: These are predictions based on trained models. Actual behavior may vary.")