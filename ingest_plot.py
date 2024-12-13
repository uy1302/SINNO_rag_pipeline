import psycopg2
import requests
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from typing import List, Dict, Any, Union
import re
from sklearn.metrics.pairwise import cosine_similarity

class SingletonTokenizer:
    _instance = None

    @staticmethod
    def get_instance(model_name="bert-base-uncased"):
        if SingletonTokenizer._instance is None:
            SingletonTokenizer._instance = AutoTokenizer.from_pretrained(model_name)
        return SingletonTokenizer._instance


class TextEmbedder:
    def __init__(self, api_url: str, db_config: Dict[str, Any], model_name: str = "bert-base-uncased"):
        self.api_url = api_url
        self.conn = psycopg2.connect(**db_config)
        self.tokenizer = SingletonTokenizer.get_instance(model_name)

    def split_para_to_sentence(self, para):
        sentences = re.split(r'(?<=[.])\s+',para)
        return sentences

    def create_semantic_chunks(self, para):
        # Use regex to split sentences more robustly
        sentences = self.split_para_to_sentence(para)
        
        semantic_chunks = []
        current_chunk = []
        
        for i, sentence in enumerate(sentences):
            if not current_chunk:
                current_chunk.append(sentence)
            else:
                last_sentence = current_chunk[-1]
                
                current_sen_embedding = np.array(self.get_embedding(sentence)).reshape(1, -1)
                last_sen_embedding = np.array(self.get_embedding(last_sentence)).reshape(1, -1)
                
                similarity = cosine_similarity(current_sen_embedding, last_sen_embedding)[0][0]
                
                if similarity > 0.55:
                    current_chunk.append(sentence)
                else:
                    semantic_chunks.append(current_chunk)
                    current_chunk = [sentence]
        
        if current_chunk:
            semantic_chunks.append(current_chunk)
        
        return semantic_chunks

    def get_embedding(self, text: str) -> List[float]:
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
            print(f"Error embedding text: {e}")
            raise

    def ingest_long_text_to_db(
        self, 
        texts: Union[List[str], pd.Series], 
        table_name: str, 
        vector_dimension: int, 
        text_column: str = 'text'
    ):
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                embedding VECTOR({vector_dimension})
            )
            """)
            self.conn.commit()
        
        with self.conn.cursor() as cur:
            chunk = self.create_semantic_chunks(texts)
            for text in chunk:
                text = ' '.join(text)
                try:
                    embedding = self.get_embedding(text)
                    
                    cur.execute(
                        f"INSERT INTO {table_name} (text, embedding) VALUES (%s, %s)",
                        (text, embedding)
                    )
                except Exception as e:
                    print(f"Error processing text: {e}")
                    continue
            
            self.conn.commit()
        
        print(f"Data ingested successfully into table {table_name}.")

    def query_most_similar(
        self, 
        input_text: str, 
        table_name: str, 
        top_k: int = 1
    ) -> List[tuple]:

        embedding = self.get_embedding(input_text)
        
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
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    API_URL = "http://localhost:8080/embed"
    DB_CONFIG = {
        "database": "vectordb",
        "user": "postgres",
        "password": "password",
        "host": "127.0.0.2",
        "port": 5432
    }
    TABLE_NAME = "plot_embeddings"
    VECTOR_DIMENSION = 1024  

    embedder = TextEmbedder(API_URL, DB_CONFIG)

    df = pd.read_csv("viblo_data.csv")
    df = df.drop(columns=['URL','Title'])

    try:
        # for x in range(2,5):
        #     long_texts = (df['Plot'][x])
            
        #     embedder.ingest_long_text_to_db(
        #         long_texts, 
        #         TABLE_NAME, 
        #         VECTOR_DIMENSION
        #     )
        input_text = input("Enter a question: ")
        results = embedder.query_most_similar(input_text, TABLE_NAME, top_k=5)

        # for result in results:
        #     print(f"Text: {result[0]}\nSimilarity: {result[1]}")
        #     print()
        url = "https://db30-34-82-190-189.ngrok-free.app/predict"
        payload = {"input_text": "answer the following question" + input_text + "when combine with this information" + results[0][0]}
        response = requests.post(url, json=payload)
        print(response.json())
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        embedder.close()