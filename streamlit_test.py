import streamlit as st
import pandas as pd
import psycopg2
import requests
from transformers import AutoTokenizer
from ingest import TextEmbedder  # Assuming your TextEmbedder class is in text_embedder.py

# Streamlit App
def main():
    st.title("Text Embedding Hybrid Search")
    st.markdown(
        """
        A simple interface for hybrid search using vector embeddings and keyword matching.
        """
    )
    
    # Configuration
    API_URL = "http://localhost:8080/embed"
    DB_CONFIG = {
        "database": "vectordb",
        "user": "postgres",
        "password": "password",
        "host": "127.0.0.1",
        "port": 5432
    }
    TABLE_NAME = "text_embeddings"
    embedder = TextEmbedder(API_URL, DB_CONFIG)

    # User Inputs
    input_text = st.text_input("Enter query text:")
    keyword = st.text_input("Enter optional keyword filter (leave blank for vector-only search):")
    top_k = st.number_input("Number of results (Top-K):", min_value=1, max_value=50, value=10)

    if st.button("Search"):
        if input_text:
            try:
                with st.spinner("Searching..."):
                    results = embedder.query_hybrid_search(input_text, TABLE_NAME, top_k=int(top_k), keyword=keyword)

                if results:
                    st.success("Search completed!")
                    results_df = pd.DataFrame(results, columns=["Title", "Similarity"])
                    st.write("### Results:")
                    st.dataframe(results_df)
                else:
                    st.warning("No results found.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter query text.")

    st.markdown("---")
    st.write("### Data Ingestion")
    uploaded_file = st.file_uploader("Upload a CSV file to ingest:", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Ingest Data"):
            try:
                vector_dim = st.number_input("Enter vector dimension for embeddings:", min_value=1, value=1024)
                with st.spinner("Ingesting data into the database..."):
                    embedder.ingest_to_db(df, TABLE_NAME, vector_dim)
                st.success("Data ingested successfully!")
            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")


if __name__ == "__main__":
    main()
