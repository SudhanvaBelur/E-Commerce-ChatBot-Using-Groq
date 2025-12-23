import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

groq_client = Groq()
chroma_client = chromadb.Client()
collection_name_faqs = "faqs"
ef=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

faqs_path = Path(__file__).parent/"resources"/"faq_data.csv"

def ingest_faq_data(path):
    if collection_name_faqs not in [c.name for c in chroma_client.list_collections()]:
        print("Ingesting FAQ data into Chromadb...")
        collection=chroma_client.get_or_create_collection(
            name=collection_name_faqs,
            embedding_function=ef
        )
        df=pd.read_csv(path)
        docs = df['question'].to_list()
        metadata = [{"answer":ans} for ans in df['answer'].to_list()]
        ids = [f"faq_{i}" for i in range(len(docs))]

        collection.add(
            documents = docs,
            metadatas = metadata,
            ids = ids
        )
        print(f"FAQ Data successfully ingested into Chroma collection: {collection_name_faqs}")
    else:
        print(f"Collection: {collection_name_faqs} already exist")

def get_relevant_query(query):
    collection=chroma_client.get_collection(name=collection_name_faqs, embedding_function=ef)
    results = collection.query(
        query_texts = [query],
        n_results = 2
    )
    return results

def faq_chain(query):
    result = get_relevant_query(query)
    context = ''.join([r.get('answer') for r in result['metadatas'][0]])
    print(f"Context: {context}")
    answer = generate_answer(query, context)
    return answer

def generate_answer(query, context):

    prompt = f'''Given the following context and question, generate answer based on this context only.
    If the answer is not found in the context, kindly state "I don't know". Don't try to make up an answer.
    
    CONTEXT: {context}
    
    QUESTION: {query}
    '''
    completion = groq_client.chat.completions.create(
        model=os.environ["GROQ_MODEL"],
        messages=[
        {
            "role": "user",
            "content": prompt
        }
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    ingest_faq_data(faqs_path)
    query = "Do you take cash as a payment option?"
    #result = get_relevant_query(query)
    answer = faq_chain(query)
    print(answer)