# document_processing.py
import os
import json
from typing import List, Dict, Any
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import time

load_dotenv()

class DocumentProcessor:
    def __init__(
        self, 
        data_dir: str = "data", 
        db_dir: str = "chroma_db",
        collection_name: str = "medical_docs",
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        embedding_model: str = "abhinand/MedEmbed-small-v0.1"
    ):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

        # Initialize the sentence transformer model using the API key
        self.embedding_model = SentenceTransformer(embedding_model, use_auth_token=hf_api_key)

        os.makedirs(db_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_dir)
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Using existing ChromaDB collection '{collection_name}'")
        except Exception as e:
            print(f"Creating new ChromaDB collection '{collection_name}'")
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def load_documents(self, filename: str = "medical_documents.json") -> List[Dict[str, Any]]:
        """Load documents from JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'r') as f:
                documents = json.load(f)
            print(f"Loaded {len(documents)} documents from {filepath}")
            return documents
        except FileNotFoundError:
            print(f"File {filepath} not found. Returning empty list.")
            return []
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {filepath}. Returning empty list.")
            return []
    
    def save_documents(self, documents: List[Dict[str, Any]], filename: str = "medical_documents.json"):
        """Save documents to JSON file"""
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(documents, f, indent=2)
        print(f"Saved {len(documents)} documents to {filepath}")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        if len(text) <= self.chunk_size:
            chunks.append(text)
        else:
            start = 0
            while start < len(text):
                # Find the end of the chunk
                end = start + self.chunk_size
                
                # If we're not at the end of the text, try to break at a sentence
                if end < len(text):
                    # Look for a period, question mark, or exclamation point followed by a space
                    punctuation_match = re.search(r'[.!?]\s', text[end-30:end+30])
                    if punctuation_match:
                        # Adjust end to be after the punctuation
                        end = end - 30 + punctuation_match.end()
                
                # Add the chunk to our list
                chunks.append(text[start:end])
                
                # Move the start position, accounting for overlap
                start = end - self.chunk_overlap
                
                # Don't create tiny chunks at the end
                if len(text) - start < self.chunk_size // 2:
                    chunks.append(text[start:])
                    break
                    
        return chunks
    
    def fetch_content_from_url(self, url: str) -> str:
        """Fetch and extract main content from a URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
            return ""
    
    def process_medical_websites(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Process multiple medical websites and return structured documents"""
        documents = []
        
        for url in urls:
            print(f"Processing {url}...")
            try:
                # Extract domain as source
                domain = url.split("//")[-1].split("/")[0]
                
                # Fetch content
                content = self.fetch_content_from_url(url)
                if not content:
                    print(f"No content extracted from {url}")
                    continue
                
                # Create document
                document = {
                    "title": f"Medical content from {domain}",
                    "content": content,
                    "source": domain,
                    "url": url
                }
                
                documents.append(document)
                
                # Be nice to the websites
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
        
        return documents
    
    def process_documents(self, documents: List[Dict[str, Any]]):
        """Process documents into chunks and add to vector store"""
        for doc_idx, doc in enumerate(documents):
            print(f"Processing document {doc_idx+1}/{len(documents)}: {doc.get('title', 'Untitled')}")
            
            # Extract the document content
            content = doc.get('content', '')
            if not content:
                print(f"Empty content for document {doc_idx+1}, skipping")
                continue
            
            # Chunk the content
            chunks = self.chunk_text(content)
            
            # Add each chunk to the vector database
            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID for the chunk
                chunk_id = f"doc{doc_idx+1}_chunk{chunk_idx+1}"
                
                # Create embedding using the model
                embedding = self.embedding_model.encode(chunk).tolist()
                
                # Process and add to ChromaDB
                try:
                    self.collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[{
                            "title": doc.get("title", "Untitled"),
                            "source": doc.get("source", "Unknown source"),
                            "url": doc.get("url", "#"),
                            "chunk_idx": chunk_idx,
                            "doc_idx": doc_idx,
                            "total_chunks": len(chunks)
                        }],
                        ids=[chunk_id]
                    )
                except Exception as e:
                    print(f"Error adding chunk to ChromaDB: {e}")
                
        print(f"Added {self.collection.count()} chunks to the vector database")
    
    def run_pipeline(self, json_file=None, urls=None):
        """Run the full document processing pipeline"""
        documents = []
        
        # Load documents from JSON if specified
        if json_file:
            documents.extend(self.load_documents(json_file))
        
        # Process URLs if specified
        if urls and len(urls) > 0:
            web_documents = self.process_medical_websites(urls)
            documents.extend(web_documents)
            
            # Save the newly fetched documents
            self.save_documents(web_documents, "web_documents.json")
        
        if not documents:
            print("No documents to process. Please provide a JSON file or URLs.")
            return 0
        
        # Process and add to vector store
        self.process_documents(documents)
        
        return self.collection.count()

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    csv_file_path = "medical_websites.csv"  # Update path if needed
    try:
        df = pd.read_csv(csv_file_path)
        medical_urls = df["Website URL"].tolist()
        print(f"Loaded {len(medical_urls)} websites from CSV.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        medical_urls = []
    
    count = processor.run_pipeline(urls=medical_urls)
    print(f"Successfully processed documents into {count} chunks in the vector database")