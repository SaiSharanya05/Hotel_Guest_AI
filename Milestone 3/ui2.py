import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from langchain_together import TogetherEmbeddings
from together import Together
from pinecone import Pinecone, ServerlessSpec
import tempfile

# Set page config
st.set_page_config(page_title="Hotel Customer Sentiment Analysis", layout="wide")

# App title and description
st.title("Hotel Customer Sentiment Analysis")
st.markdown("Upload your hotel reviews data and analyze customer sentiment across different dimensions.")

# Sidebar for API keys and configuration
with st.sidebar:
    st.header("API Configuration")
    together_api_key = st.text_input("Together API Key", type="password", 
                                    value="cb6bcac5583a1f6ce6bbbf1eb1af471d5bbeaf1c4ee6551e1d830a2b0cd00964")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password", 
                                    value="pcsk_4SKEDS_QZVnzA5gEJ6c58zYvFRbQLJTTXasgwVHC9hX428L4ToFtxWpyTGJ9UmEqhzvWEc")
    
    st.header("Vector Database")
    pinecone_index_name = st.text_input("Pinecone Index Name", value="hotel-reviews")
    pinecone_host = st.text_input("Pinecone Host", value="hotel-reviews-dcl6jm7.svc.aped-4627-b74a.pinecone.io")
    
    embedding_model = st.selectbox(
        "Embedding Model",
        ["togethercomputer/m2-bert-80M-8k-retrieval"],
        index=0
    )
    
    llm_model = st.selectbox(
        "LLM Model",
        ["meta-llama/Llama-Vision-Free"],
        index=0
    )

# Main content
tab1, tab2, tab3 = st.tabs(["Upload & Process", "Query Reviews", "Sentiment Dashboard"])

# Tab 1: Upload & Process
with tab1:
    st.header("Upload Reviews Data")
    
    uploaded_file = st.file_uploader("Upload your hotel reviews Excel file", type=["xlsx", "csv"])
    
    if uploaded_file:
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name
        
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(tmp_filepath)
            else:
                df = pd.read_excel(tmp_filepath)
            
            st.success("File uploaded successfully!")
            
            # Display dataframe preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Check for required columns
            required_columns = ["customer_id", "review_id", "Rating", "Review"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                # Data processing options
                st.subheader("Process Data")
                
                with st.form("processing_form"):
                    process_data = st.checkbox("Process and index data", value=False)
                    batch_size = st.slider("Batch Size for Processing", min_value=32, max_value=256, value=128, step=32)
                    
                    # Date handling
                    if "review_date" in df.columns:
                        st.info("Found review_date column. Will use for filtering.")
                    else:
                        st.warning("No review_date column found. Please ensure dates are in the right format.")
                        
                    submitted = st.form_submit_button("Process Data")
                
                if submitted and process_data:
                    # Set API keys
                    os.environ["TOGETHER_API_KEY"] = together_api_key
                    
                    # Initialize embedding model
                    with st.spinner("Initializing embedding model..."):
                        embeddings = TogetherEmbeddings(model=embedding_model)
                    
                    # Check for review_date_numeric
                    if "review_date_numeric" not in df.columns and "review_date" in df.columns:
                        try:
                            # Try to convert to numeric date format (YYYYMMDD)
                            df["review_date_numeric"] = pd.to_datetime(df["review_date"]).dt.strftime("%Y%m%d").astype(int)
                        except:
                            st.error("Could not convert review_date to numeric format. Using default values.")
                            df["review_date_numeric"] = 20240101  # Default value
                    
                    # Process embeddings
                    reviews = df["Review"].tolist()
                    embedding_list = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(0, len(reviews), batch_size):
                        batch = reviews[i:i + batch_size]
                        status_text.text(f"Generating embeddings for reviews {i} to {i + len(batch)} of {len(reviews)}")
                        batch_embeddings = embeddings.embed_documents(batch)
                        embedding_list.extend(batch_embeddings)
                        progress_bar.progress((i + len(batch)) / len(reviews))
                    
                    status_text.text("Generating metadata...")
                    
                    # Create metadata
                    metadata_list = df.apply(lambda row: {
                        "customer_id": int(row["customer_id"]) if "customer_id" in row else 0,
                        "review_date": int(row["review_date_numeric"]) if "review_date_numeric" in row else 20240101,
                        "Rating": int(row["Rating"]) if "Rating" in row else 5,
                        "review_id": int(row["review_id"]) if "review_id" in row else i
                    }, axis=1).tolist()
                    
                    # Initialize Pinecone
                    status_text.text("Initializing Pinecone...")
                    pc = Pinecone(api_key=pinecone_api_key)
                    
                    # Check if index exists, create if it doesn't
                    index_list = pc.list_indexes()
                    
                    if pinecone_index_name not in [idx.name for idx in index_list]:
                        status_text.text(f"Creating new Pinecone index: {pinecone_index_name}")
                        pc.create_index(
                            name=pinecone_index_name,
                            dimension=len(embedding_list[0]),  # Dimension of the embedding vectors
                            metric='cosine',
                            spec=ServerlessSpec(
                                cloud='aws',
                                region='us-east-1'
                            )
                        )
                    
                    # Connect to index
                    index = pc.Index(host=pinecone_host)
                    
                    # Insert vectors in batches
                    status_text.text("Uploading vectors to Pinecone...")
                    upload_progress = st.progress(0)
                    
                    for i in range(0, len(embedding_list), batch_size):
                        batch_vectors = [
                            (str(i + j), embedding_list[i + j], metadata_list[i + j])
                            for j in range(min(batch_size, len(embedding_list) - i))
                        ]
                        index.upsert(vectors=batch_vectors)
                        upload_progress.progress((i + len(batch_vectors)) / len(embedding_list))
                    
                    status_text.text("Processing complete! You can now query your data.")
                    st.session_state.df = df
                    st.session_state.processed = True
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        
        # Clean up temporary file
        os.unlink(tmp_filepath)

# Tab 2: Query Reviews
with tab2:
    st.header("Query Reviews")
    
    # Check if data has been processed
    if 'processed' in st.session_state and st.session_state.processed:
        # Date range filter
        st.subheader("Date Range Filter")
        
        # Get min and max dates from dataframe if available
        if 'df' in st.session_state and 'review_date' in st.session_state.df.columns:
            try:
                min_date = pd.to_datetime(st.session_state.df['review_date']).min().date()
                max_date = pd.to_datetime(st.session_state.df['review_date']).max().date()
            except:
                min_date = datetime.now().date() - timedelta(days=30)
                max_date = datetime.now().date()
        else:
            min_date = datetime.now().date() - timedelta(days=30)
            max_date = datetime.now().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date)
        with col2:
            end_date = st.date_input("End Date", max_date)
        
        # Rating filter
        st.subheader("Rating Filter")
        min_rating, max_rating = st.slider("Select Rating Range", 1, 5, (1, 5))
        
        # Query input
        st.subheader("Enter Your Query")
        query = st.text_input("What would you like to know about the reviews?", 
                             "What are reviews mentioning about food and restaurant service?")
        
        # Number of results
        top_k = st.slider("Number of Results", 1, 50, 5)
        
        if st.button("Search Reviews"):
            try:
                # Set API keys
                os.environ["TOGETHER_API_KEY"] = together_api_key
                
                # Initialize components
                embeddings = TogetherEmbeddings(model=embedding_model)
                client = Together(api_key=together_api_key)
                pc = Pinecone(api_key=pinecone_api_key)
                index = pc.Index(host=pinecone_host)
                
                # Convert dates to numeric format for filtering
                start_date_numeric = int(start_date.strftime("%Y%m%d"))
                end_date_numeric = int(end_date.strftime("%Y%m%d"))
                
                # Generate query embedding
                with st.spinner("Generating query embedding..."):
                    query_embedding = embeddings.embed_query(query)
                
                # Query the index
                with st.spinner("Searching for relevant reviews..."):
                    results = index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True,
                        filter={
                            "Rating": {"$gte": min_rating, "$lte": max_rating},
                            "review_date": {"$gte": start_date_numeric, "$lte": end_date_numeric}
                        }
                    )
                
                if 'matches' in results and results['matches']:
                    matches = results["matches"]
                    
                    # Extract review_ids from matches
                    matched_ids = [int(match["metadata"]["review_id"]) for match in matches]
                    
                    # Get reviews from dataframe
                    req_df = st.session_state.df[st.session_state.df["review_id"].isin(matched_ids)]
                    
                    # Display matched reviews
                    st.subheader("Matching Reviews")
                    for i, (_, row) in enumerate(req_df.iterrows()):
                        with st.expander(f"Review #{i+1} - Rating: {row['Rating']}"):
                            st.write(row['Review'])
                            st.caption(f"Review ID: {row['review_id']} | Customer ID: {row['customer_id']}")
                    
                    # Generate sentiment summary
                    st.subheader("AI-Generated Sentiment Summary")
                    
                    concatenated_reviews = " ".join(req_df["Review"].tolist())
                    
                    with st.spinner("Analyzing sentiment..."):
                        response = client.chat.completions.create(
                            model=llm_model,
                            messages=[{
                                "role": "user", 
                                "content": f"""Briefly summarize the overall sentiment of customers about {query.split('about')[-1].strip()} based on these reviews: 
                                {concatenated_reviews}. 
                                Don't mention the name of the hotel. Provide insights on both positive and negative aspects. 
                                Format your response in markdown with clear sections."""
                            }]
                        )
                    
                    summary = response.choices[0].message.content
                    st.markdown(summary)
                else:
                    st.info("No matching reviews found. Try adjusting your filters or query.")
            
            except Exception as e:
                st.error(f"Error querying reviews: {str(e)}")
    else:
        st.info("Please upload and process your data in the 'Upload & Process' tab first.")

# Tab 3: Sentiment Dashboard
with tab3:
    st.header("Sentiment Dashboard")
    
    if 'processed' in st.session_state and st.session_state.processed:
        df = st.session_state.df
        
        # Create dashboard with visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            st.subheader("Rating Distribution")
            rating_counts = df["Rating"].value_counts().sort_index()
            fig = px.bar(x=rating_counts.index, y=rating_counts.values, 
                        labels={"x": "Rating", "y": "Count"})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rating over time (if date column exists)
            if "review_date" in df.columns:
                st.subheader("Ratings Over Time")
                try:
                    df_time = df.copy()
                    df_time["review_date"] = pd.to_datetime(df_time["review_date"])
                    df_time.set_index("review_date", inplace=True)
                    df_time = df_time.resample('M').mean(numeric_only=True)
                    
                    fig = px.line(df_time, y="Rating", labels={"index": "Date", "value": "Avg. Rating"})
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.error("Could not process time series data for ratings.")
        
        # Topic-based sentiment analysis
        st.subheader("Topic Analysis")
        
        topics = ["Food & Restaurant", "Room Cleanliness", "Staff & Service", "Amenities", "Value for Money"]
        selected_topic = st.selectbox("Select Topic to Analyze", topics)
        
        if st.button(f"Analyze {selected_topic} Sentiment"):
            # Map topics to queries
            topic_queries = {
                "Food & Restaurant": "restaurant food dining breakfast lunch dinner menu cuisine",
                "Room Cleanliness": "cleanliness clean dirty room housekeeping hygiene bathroom",
                "Staff & Service": "staff service reception desk manager concierge friendly helpful",
                "Amenities": "pool gym spa facilities business center fitness internet wifi",
                "Value for Money": "price value expensive cheap worth cost money pricing"
            }
            
            try:
                # Set API keys
                os.environ["TOGETHER_API_KEY"] = together_api_key
                
                # Initialize components
                embeddings = TogetherEmbeddings(model=embedding_model)
                client = Together(api_key=together_api_key)
                pc = Pinecone(api_key=pinecone_api_key)
                index = pc.Index(host=pinecone_host)
                
                # Generate query embedding
                with st.spinner(f"Finding reviews about {selected_topic}..."):
                    query_embedding = embeddings.embed_query(f"Reviews about {topic_queries[selected_topic]}")
                
                # Query the index
                results = index.query(
                    vector=query_embedding,
                    top_k=20,
                    include_metadata=True
                )
                
                if 'matches' in results and results['matches']:
                    matches = results["matches"]
                    
                    # Extract review_ids from matches
                    matched_ids = [int(match["metadata"]["review_id"]) for match in matches]
                    
                    # Get reviews from dataframe
                    req_df = st.session_state.df[st.session_state.df["review_id"].isin(matched_ids)]
                    
                    # Generate sentiment summary
                    with st.spinner(f"Analyzing {selected_topic} sentiment..."):
                        concatenated_reviews = " ".join(req_df["Review"].tolist())
                        
                        response = client.chat.completions.create(
                            model=llm_model,
                            messages=[{
                                "role": "user", 
                                "content": f"""Analyze the sentiment about {selected_topic} in these hotel reviews:
                                {concatenated_reviews}
                                
                                Provide:
                                1. An overall sentiment score from 1-10
                                2. Key positive points
                                3. Key negative points
                                4. Recommendations for improvement
                                
                                Format your response in markdown. Don't mention the hotel name."""
                            }]
                        )
                    
                    summary = response.choices[0].message.content
                    st.markdown(summary)
                    
                    # Show matching reviews
                    with st.expander("View Relevant Reviews"):
                        for i, row in req_df.iterrows():
                            st.markdown(f"**Rating: {row['Rating']}**")
                            st.write(row['Review'])
                            st.divider()
                else:
                    st.info(f"No reviews found about {selected_topic}. Try another topic.")
            
            except Exception as e:
                st.error(f"Error analyzing topic: {str(e)}")
    else:
        st.info("Please upload and process your data in the 'Upload & Process' tab first.")