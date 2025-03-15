import streamlit as st
import pandas as pd
import random
from datetime import date, datetime, timedelta
import joblib
import xgboost
import numpy as np
from pymongo import MongoClient
import time
import os

# Set page configuration
st.set_page_config(
    page_title="Luxury Hotel Booking",
    page_icon="🏨",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        border: none;
    }
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    .status-box {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize MongoDB connection
@st.cache_resource
def init_mongodb():
    return MongoClient("mongodb+srv://saisharanyasriramoju05:Sharanya032005@cluster0.7fmgr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

client = init_mongodb()

def load_models():
    encoder = joblib.load('encoder.pkl')
    xgb_model = joblib.load('xgb_model_dining.pkl')
    
    # Updated to use the feature_names.txt file correctly
    if os.path.exists('feature_names.txt'):
        with open('feature_names.txt', 'r') as f:
            features = [line.strip() for line in f.readlines()]
    elif os.path.exists('feature_structure.pkl'):
        # Fallback to the feature structure pickle
        feature_structure = pd.read_pickle('feature_structure.pkl')
        features = feature_structure.columns.tolist()
    else:
        # Last resort - try to read from Excel
        features_df = pd.read_excel('features.xlsx')
        if 'Unnamed: 0' in features_df.columns:
            features = features_df['Unnamed: 0'].tolist()
        else:
            features = features_df.columns.tolist()
    
    # Load dish classes
    dish_classes = np.load('dish_classes.npy', allow_pickle=True)
    
    return encoder, xgb_model, features, dish_classes

# Main title with animation
def display_title():
    st.markdown("""
        <h1 style='text-align: center; color: #1e3d59; animation: fadeIn 2s;'>
            🌟 Luxury Hotel Booking Experience 🌟
        </h1>
        <p style='text-align: center; color: #666; font-style: italic;'>
            Where Comfort Meets Elegance
        </p>
    """, unsafe_allow_html=True)

def main():
    display_title()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📝 Booking Details", "🏷️ Special Requests", "💫 Preferences"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            has_customer_id = st.radio("Do you have a Customer ID?", ("Yes", "No"))
            
            if has_customer_id == "Yes":
                customer_id = st.text_input("Enter your Customer ID", "")
            else:
                customer_id = random.randint(10001, 99999)
                st.info(f"Your generated Customer ID: {customer_id}")
            
            name = st.text_input("Full Name", placeholder="Enter your full name")
            email = st.text_input("Email Address", placeholder="your.email@example.com")
            phone = st.text_input("Phone Number", placeholder="+1234567890")
            age = st.number_input("Age", min_value=18, max_value=120, value=25)

        with col2:
            st.subheader("Stay Details")
            today = date.today()
            checkin_date = st.date_input("Check-in Date", 
                                       min_value=today,
                                       value=today + timedelta(days=1))
            checkout_date = st.date_input("Check-out Date", 
                                        min_value=checkin_date,
                                        value=checkin_date + timedelta(days=1))
            
            stayers = st.number_input("Number of Guests", min_value=1, max_value=4, value=2)
            room_type = st.selectbox("Room Type", 
                                   ["Standard", "Deluxe", "Suite", "Executive Suite"])
            
    with tab2:
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Dining Preferences")
            cuisine_options = ["South Indian", "North Indian", "Multi"]
            preferred_cuisine = st.selectbox("Preferred Cuisine", cuisine_options)
            dietary_restrictions = st.multiselect("Dietary Restrictions",
                                                ["Vegetarian", "Vegan", "Gluten-Free", 
                                                 "Dairy-Free", "Nut-Free"])
            meal_plan = st.selectbox("Meal Plan", 
                                   ["Room Only", "Breakfast Included", 
                                    "Half Board", "Full Board"])
        
        with col4:
            st.subheader("Additional Services")
            airport_pickup = st.checkbox("Airport Pickup Service")
            if airport_pickup:
                flight_number = st.text_input("Flight Number")
                arrival_time = st.time_input("Arrival Time")
            
            spa_service = st.checkbox("Spa Services")
            if spa_service:
                preferred_time = st.selectbox("Preferred Time", 
                                            ["Morning", "Afternoon", "Evening"])

    with tab3:
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("Room Preferences")
            bed_type = st.selectbox("Bed Type", ["King", "Queen", "Twin"])
            floor_preference = st.selectbox("Floor Preference", 
                                          ["Lower Floor", "Middle Floor", "High Floor"])
            view_preference = st.selectbox("View Preference", 
                                         ["City View", "Pool View", "Garden View"])
        
        with col6:
            st.subheader("Payment Details")
            payment_method = st.selectbox("Payment Method", 
                                        ["Credit Card", "Debit Card", "Points"])
            if payment_method in ["Credit Card", "Debit Card"]:
                card_number = st.text_input("Card Number", type="password")
                expiry = st.text_input("Expiry Date (MM/YY)")
                cvv = st.text_input("CVV", type="password", max_chars=3)
            
            preferred_booking = "Yes" if payment_method == "Points" else "No"

    # Special Requests
    st.subheader("Special Requests")
    special_requests = st.text_area("Any additional requests or notes?", 
                                  placeholder="Enter any special requirements...")

    # Terms and Conditions
    st.markdown("---")
    terms = st.checkbox("I agree to the terms and conditions")
    
    # Submit Button with loading animation
    if st.button("Confirm Booking", disabled=not terms):
        with st.spinner("Processing your booking..."):
            try:
                # Convert date objects to datetime objects for MongoDB compatibility
                checkin_datetime = datetime.combine(checkin_date, datetime.min.time())
                checkout_datetime = datetime.combine(checkout_date, datetime.min.time())
                
                # Create booking data dictionary with datetime objects instead of date objects
                new_data = {
                    'customer_id': int(customer_id),
                    'name': name,
                    'email': email,
                    'phone': phone,
                    'Preferred Cusine': preferred_cuisine,
                    'age': age,
                    'check_in_date': checkin_datetime,  # Using datetime object
                    'check_out_date': checkout_datetime,  # Using datetime object
                    'booked_through_points': preferred_booking,
                    'number_of_stayers': stayers,
                    'room_type': room_type,
                    'meal_plan': meal_plan,
                    'special_requests': special_requests
                }
                
                # For prediction, we'll use the original date objects since pandas can handle them
                prediction_data = {
                    'customer_id': int(customer_id),
                    'name': name,
                    'email': email,
                    'phone': phone,
                    'Preferred Cusine': preferred_cuisine,
                    'age': age,
                    'check_in_date': checkin_date,
                    'check_out_date': checkout_date,
                    'booked_through_points': preferred_booking,
                    'number_of_stayers': stayers,
                    'room_type': room_type,
                    'meal_plan': meal_plan,
                    'special_requests': special_requests
                }
                
                # Create DataFrame and process for prediction
                new_df = pd.DataFrame([prediction_data])
                new_df['booked_through_points'] = new_df['booked_through_points'].apply(
                    lambda x: 1 if x=='Yes' else 0)
                new_df['check_in_date'] = pd.to_datetime(new_df['check_in_date'])
                new_df['check_out_date'] = pd.to_datetime(new_df['check_out_date'])
                
                # Calculate additional features
                new_df['check_in_day'] = new_df['check_in_date'].dt.dayofweek
                new_df['check_out_day'] = new_df['check_out_date'].dt.dayofweek
                new_df['check_in_month'] = new_df['check_in_date'].dt.month
                new_df['check_out_month'] = new_df['check_out_date'].dt.month
                new_df['stay_duration'] = (new_df['check_out_date'] - 
                                         new_df['check_in_date']).dt.days
                
                # Load necessary data and models
                customer_features = pd.read_excel('customer_features.xlsx')
                customer_dish = pd.read_excel('customer_dish.xlsx')
                cuisine_features = pd.read_excel('cuisine_features.xlsx')
                cuisine_dish = pd.read_excel('cuisine_dish.xlsx')
                
                # Fix: Ensure customer_id is integer type
                customer_features['customer_id'] = customer_features['customer_id'].astype(int)
                customer_dish['customer_id'] = customer_dish['customer_id'].astype(int)
                new_df['customer_id'] = new_df['customer_id'].astype(int)
                
                # Merge features
                new_df = new_df.merge(customer_features, on='customer_id', how='left')
                new_df = new_df.merge(customer_dish, on='customer_id', how='left')
                new_df = new_df.merge(cuisine_features, on='Preferred Cusine', how='left')
                new_df = new_df.merge(cuisine_dish, on='Preferred Cusine', how='left')
                
                # Drop unnecessary columns
                new_df = new_df.drop(['check_in_date', 'check_out_date', 'name', 
                                    'email', 'phone', 'room_type', 'meal_plan', 
                                    'special_requests'], axis=1, errors='ignore')
                
                # Process for prediction
                encoder, model, features, dish_classes = load_models()
                categorical_cols = ['Preferred Cusine', 'most_frequent_dish', 'cuisine_popular_dish']
                
                # Handle missing values for categorical columns
                for col in categorical_cols:
                    if col in new_df.columns and new_df[col].isnull().any():
                        new_df[col].fillna('nan', inplace=True)
                        
                # Generate dish recommendations
                encoded_test = encoder.transform(new_df[categorical_cols])
                encoded_test_df = pd.DataFrame(
                    encoded_test, 
                    columns=encoder.get_feature_names_out(categorical_cols)
                )
                
                new_df = pd.concat([new_df.drop(columns=categorical_cols), 
                                  encoded_test_df], axis=1)
                
                # Fix: Ensure all required features are present
                for feature in features:
                    if feature not in new_df.columns:
                        new_df[feature] = 0
                
                # Make sure to select only the features the model knows about,
                # and in the right order
                new_df = new_df[features]
                
                # Get predictions
                y_pred_prob = model.predict_proba(new_df)
                top_3_indices = np.argsort(-y_pred_prob[0])[:3]
                top_3_dishes = [dish_classes[i] for i in top_3_indices]
                
                # Save to MongoDB
                db = client["hotel_guests"]
                new_bookings_collection = db["new_bookings"]
                result = new_bookings_collection.insert_one(new_data)
                
                # Success message with animation
                st.balloons()
                st.success("✨ Booking Confirmed Successfully! ✨")
                
                # Display booking summary in an organized way
                st.markdown("### 📋 Booking Summary")
                col7, col8 = st.columns(2)
                
                with col7:
                    st.markdown(f"""
                        **Guest Details:**
                        - Name: {name}
                        - Customer ID: {customer_id}
                        - Email: {email}
                        - Phone: {phone}
                    """)
                
                with col8:
                    st.markdown(f"""
                        **Stay Details:**
                        - Check-in: {checkin_date}
                        - Check-out: {checkout_date}
                        - Room Type: {room_type}
                        - Number of Guests: {stayers}
                    """)
                
                # Display recommended dishes and offers
                st.markdown("### 🍽️ Personalized Dining Recommendations")
                for i, dish in enumerate(top_3_dishes, 1):
                    discount = "20% off!" if "thali" in str(dish).lower() else "15% off!"
                    st.markdown(f"**{i}. {dish}** - *Get {discount}*")
                
                # Confirmation email message
                st.info("📧 A confirmation email has been sent to your email address "
                       "with booking details and discount coupons.")
                
            except Exception as e:
                st.error(f"An error occurred during booking: {str(e)}")
                st.warning("Please try again or contact support.")
                
                # For debugging
                st.error(f"Error details: {type(e).__name__}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()