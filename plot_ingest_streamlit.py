import streamlit as st
import pandas as pd
import psycopg2
import requests
from typing import Dict, Any

class SingletonTokenizerStreamlit:
    _instance = None

    @staticmethod
    def get_instance(model_name="bert-base-uncased"):
        if SingletonTokenizerStreamlit._instance is None:
            from transformers import AutoTokenizer
            SingletonTokenizerStreamlit._instance = AutoTokenizer.from_pretrained(model_name)
        return SingletonTokenizerStreamlit._instance

class TextEmbedderStreamlit:
    def __init__(self, api_url: str, db_config: Dict[str, Any]):
        self.api_url = api_url
        self.conn = psycopg2.connect(**db_config)

    def get_embedding(self, text: str):
        try:
            response = requests.post(
                self.api_url,
                json={"inputs": [text]},
                headers={"Content-Type": "application/json"},
                timeout=30  
            )
            response.raise_for_status()
            
            chunk_embedding = response.json()
            if chunk_embedding:
                return chunk_embedding[0]
            else:
                raise ValueError("Empty embedding response")
        except Exception as e:
            st.error(f"Error embedding text: {e}")
            return None

    def query_most_similar(self, input_text: str, table_name: str, top_k: int = 5):
        embedding = self.get_embedding(input_text)
        
        if embedding is None:
            return []
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
            SELECT text, embedding <-> %s::VECTOR AS similarity
            FROM {table_name}
            ORDER BY similarity ASC
            LIMIT %s
            """, (embedding, top_k))
            results = cur.fetchall()
        
        return results

    def close(self):
        if self.conn:
            self.conn.close()

def main():
    st.title("Semantic Search Application")

    # Configuration
    API_URL = "http://localhost:8080/embed"
    DB_CONFIG = {
        "database": "vectordb",
        "user": "postgres",
        "password": "password",
        "host": "127.0.0.2",
        "port": 5432
    }
    TABLE_NAME = "plot_embeddings"

    # Initialize embedder
    embedder = TextEmbedderStreamlit(API_URL, DB_CONFIG)

    # User input
    input_text = st.text_area("Enter text to find similar entries:", height=150)
    top_k = st.slider("Number of similar entries to retrieve:", min_value=1, max_value=10, value=5)

    if st.button("Search"):
        if input_text:
            try:
                # Perform semantic search
                results = embedder.query_most_similar(input_text, TABLE_NAME, top_k)

                st.subheader("Similar Entries:")
                for i, (text, similarity) in enumerate(results, 1):
                    st.markdown(f"**Entry {i}**")
                    st.text(text)
                    st.text(f"Similarity Score: {similarity}")
                    st.divider()

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to search.")

    # Close the database connection when done
    st.sidebar.button("Close Database Connection", on_click=embedder.close)

if __name__ == "__main__":
    main()