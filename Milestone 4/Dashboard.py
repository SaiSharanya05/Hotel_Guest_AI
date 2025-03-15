import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import calendar
from wordcloud import WordCloud
from io import BytesIO

# Setup
st.set_page_config(page_title="Hotel Analytics", layout="wide")

# MongoDB connection
@st.cache_resource
def get_mongodb():
    return MongoClient("mongodb+srv://saisharanyasriramoju05:Sharanya032005@cluster0.7fmgr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# Data loading functions
@st.cache_data
def load_bookings(start_date, end_date):
    client = get_mongodb()
    collection = client["hotel_guests"]["new_bookings"]
    query = {"check_in_date": {"$gte": start_date, "$lte": end_date}}
    data = list(collection.find(query))
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    if not df.empty:
        df['stay_duration'] = (df['check_out_date'] - df['check_in_date']).dt.days
    return df

@st.cache_data
def load_dining(start_date, end_date):
    client = get_mongodb()
    collection = client["hotel_guests"]["dining_info"]
    data = list(collection.find())
    df = pd.DataFrame(data)
    if not df.empty:
        df['order_time'] = pd.to_datetime(df['order_time'])
        df = df[(df['order_time'] >= start_date) & (df['order_time'] <= end_date)]
        df['revenue'] = df['price_for_1'] * df.get('Qty', 1)
    return df

@st.cache_data
def get_review_data():
    # Sample review data
    dates = pd.date_range('2023-01-01', '2023-12-31', periods=100)
    ratings = np.random.choice([1, 2, 3, 4, 5], 100, p=[0.05, 0.1, 0.2, 0.35, 0.3])
    sentiments = ['Positive' if r >= 4 else 'Neutral' if r == 3 else 'Negative' for r in ratings]
    
    return pd.DataFrame({
        'review_date': dates,
        'Rating': ratings,
        'Sentiment': sentiments,
        'Review': [f"{'Great' if s=='Positive' else 'Average' if s=='Neutral' else 'Poor'} hotel experience" for s in sentiments]
    })

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    img = BytesIO()
    wc.to_image().save(img, format='PNG')
    return img.getvalue()

# Sidebar
st.sidebar.title("Filters")
start_date = pd.to_datetime(st.sidebar.date_input("From", datetime.now() - timedelta(days=90)))
end_date = pd.to_datetime(st.sidebar.date_input("To", datetime.now()))

# Main tabs
tab1, tab2, tab3 = st.tabs(["Bookings", "Dining", "Reviews"])

# Tab 1: Bookings
with tab1:
    bookings = load_bookings(start_date, end_date)
    
    if bookings.empty:
        st.warning("No booking data available for selected dates")
    else:
        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Bookings", len(bookings))
        c2.metric("Avg Stay", f"{bookings['stay_duration'].mean():.1f} days")
        c3.metric("Popular Room", bookings.get('room_type', pd.Series(['N/A'])).value_counts().index[0])
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Booking trends
            interval = st.radio("Trend interval:", ["Daily", "Weekly", "Monthly"], horizontal=True)
            freq = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[interval]
            trend_data = bookings.groupby(pd.Grouper(key='check_in_date', freq=freq)).size().reset_index(name='count')
            fig = px.line(trend_data, x='check_in_date', y='count', title='Booking Trends')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Room distribution
            if 'room_type' in bookings.columns:
                room_data = bookings['room_type'].value_counts().reset_index()
                room_data.columns = ['Room Type', 'Count']
                fig = px.pie(room_data, values='Count', names='Room Type', title='Room Distribution', hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

# Tab 2: Dining
with tab2:
    dining = load_dining(start_date, end_date)
    
    if dining.empty:
        st.warning("No dining data available for selected dates")
    else:
        # Cuisine filter
        if 'Preferred Cusine' in dining.columns:
            cuisines = dining['Preferred Cusine'].unique()
            selected = st.multiselect("Filter by cuisine:", cuisines)
            if selected:
                dining = dining[dining['Preferred Cusine'].isin(selected)]
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Cuisine costs
            if 'Preferred Cusine' in dining.columns and 'price_for_1' in dining.columns:
                cuisine_costs = dining.groupby('Preferred Cusine')['price_for_1'].mean().reset_index()
                fig = px.pie(cuisine_costs, values='price_for_1', names='Preferred Cusine', 
                             title='Average Cost by Cuisine')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue trends
            if 'revenue' in dining.columns:
                period = st.radio("Revenue interval:", ["Daily", "Weekly", "Monthly"], horizontal=True)
                freq = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[period]
                dining['period'] = dining['order_time'].dt.to_period(freq)
                revenue_data = dining.groupby('period')['revenue'].sum().reset_index()
                revenue_data['period'] = revenue_data['period'].astype(str)
                fig = px.line(revenue_data, x='period', y='revenue', title=f'{period} Revenue', markers=True)
                st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Popular dishes
            if 'dish' in dining.columns:
                dish_data = dining.groupby('dish').size().reset_index(name='count')
                dish_data = dish_data.sort_values('count', ascending=False).head(10)
                fig = px.bar(dish_data, x='dish', y='count', title='Top 10 Popular Dishes')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Hourly distribution
            if 'order_time' in dining.columns:
                dining['hour'] = dining['order_time'].dt.hour
                hourly_data = dining.groupby('hour').size().reset_index(name='count')
                fig = px.bar(hourly_data, x='hour', y='count', title='Orders by Hour')
                fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
                st.plotly_chart(fig, use_container_width=True)

# Tab 3: Reviews
with tab3:
    reviews = get_review_data()
    filtered_reviews = reviews[
        (reviews['review_date'] >= start_date) & 
        (reviews['review_date'] <= end_date)
    ]
    
    # Filter by rating
    rating_range = st.slider("Rating", 1, 5, (1, 5), 1)
    filtered_reviews = filtered_reviews[
        (filtered_reviews['Rating'] >= rating_range[0]) & 
        (filtered_reviews['Rating'] <= rating_range[1])
    ]
    
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reviews", len(filtered_reviews))
    c2.metric("Avg Rating", f"{filtered_reviews['Rating'].mean():.1f}/5")
    c3.metric("Positive %", f"{100 * (filtered_reviews['Sentiment'] == 'Positive').mean():.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        sentiment_data = filtered_reviews['Sentiment'].value_counts().reset_index()
        sentiment_data.columns = ['Sentiment', 'Count']
        colors = {'Positive': '#2ca02c', 'Neutral': '#1f77b4', 'Negative': '#d62728'}
        fig = px.pie(sentiment_data, values='Count', names='Sentiment', color='Sentiment', 
                    color_discrete_map=colors, title='Sentiment Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rating distribution
        fig = px.histogram(filtered_reviews, x='Rating', nbins=5, title='Rating Distribution')
        fig.update_layout(bargap=0.1)
        fig.update_xaxes(tickvals=[1, 2, 3, 4, 5])
        st.plotly_chart(fig, use_container_width=True)
    
    # Word cloud
    if not filtered_reviews.empty:
        st.subheader("Review Word Cloud")
        wordcloud = generate_wordcloud(" ".join(filtered_reviews['Review']))
        st.image(wordcloud)
    
    # Data table
    st.subheader("Reviews Data")
    st.dataframe(filtered_reviews[['review_date', 'Rating', 'Sentiment', 'Review']], 
                use_container_width=True, height=300)