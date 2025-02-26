import os
import groq
import chromadb
import requests
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
from urllib.parse import urlparse
import json
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class MedicalRAG:
    def __init__(self, 
                 db_path="./chroma_db/",
                 collection_name="medical_docs",
                 embedding_model_name="abhinand/MedEmbed-small-v0.1",
                 llm_model="llama3-70b-8192",
                 cache_expiry_days=7):
        
        # Setup embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Connect to existing ChromaDB or create a new one
        try:
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Connected to existing ChromaDB collection: {collection_name}")
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            print("Creating a new collection instead")
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
        # Setup cache collection
        try:
            self.cache_collection = self.chroma_client.get_collection(name=f"{collection_name}_cache")
            print(f"Connected to existing cache collection: {collection_name}_cache")
        except Exception:
            print(f"Creating a new cache collection")
            self.cache_collection = self.chroma_client.create_collection(
                name=f"{collection_name}_cache",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Setup LLM client
        self.groq_client = groq.Client(api_key=GROQ_API_KEY)
        self.llm_model = llm_model
        self.cache_expiry_days = cache_expiry_days
    
    def google_search(self, query, max_results=5):
        """Run Google search and return results"""
        url = "https://customsearch.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'cx': GOOGLE_CSE_ID,
            'key': GOOGLE_API_KEY,
            'num': max_results,
            'safe': 'active'
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            
            data = resp.json()
            if 'error' in data:
                print(f"API Error: {data['error']['message']}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"Error during search request: {e}")
            return []
        except ValueError as e:
            print(f"Error parsing JSON response: {e}")
            return []

        results = []
        if "items" in data:
            for item in data["items"]:
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet", ""),
                    "domain": urlparse(item.get("link", "")).netloc
                })
        return results
    
    def search_chroma_db(self, query, top_k=3, collection=None):
        """Find relevant documents in ChromaDB"""
        if collection is None:
            collection = self.collection
            
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Search the database
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            if results["documents"] and len(results["documents"]) > 0:
                # Format results
                formatted_results = []
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if "metadatas" in results and i < len(results["metadatas"][0]) else {}
                    source = metadata.get("source", "Unknown source")
                    title = metadata.get("title", "Untitled document")
                    url = metadata.get("url", "#")
                    
                    formatted_results.append({
                        "content": doc,
                        "title": title,
                        "source": source,
                        "url": url
                    })
                
                return formatted_results, results["distances"][0] if "distances" in results else None
            
            return [], None  # No results found
        
        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            return [], None
    
    def check_query_cache(self, query):
        """Check if we already have a similar query cached"""
        cached_results, distances = self.search_chroma_db(query, top_k=1, collection=self.cache_collection)
        
        # Check for similar queries
        if cached_results and distances and distances[0] < 0.05:  # Similarity threshold
            cached_data = json.loads(cached_results[0]["content"])
            timestamp = cached_data.get("timestamp", 0)
            
            # Check if cache is still valid
            if (time.time() - timestamp) < (self.cache_expiry_days * 86400):
                print(f"Using cached response (similarity: {distances[0]:.4f})")
                return cached_data
        
        return None
    
    def store_in_cache(self, query, response_data):
        """Save query and response to cache"""
        try:
            # Add timestamp
            response_data["timestamp"] = time.time()
            json_data = json.dumps(response_data)
            
            # Create unique ID
            cache_id = f"cache_{int(time.time())}_{hash(query) % 10000}"
            
            # Create embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Save to cache
            self.cache_collection.add(
                ids=[cache_id],
                embeddings=[query_embedding.tolist()],
                documents=[json_data],
                metadatas=[{
                    "source": "query_cache",
                    "title": f"Cached response for: {query[:50]}...",
                    "url": "#",
                    "query": query,
                    "cache_date": datetime.now().isoformat()
                }]
            )
            print(f"Stored response in cache with ID: {cache_id}")
            
        except Exception as e:
            print(f"Error storing in cache: {e}")
    
    def generate_answer(self, question, context_docs, source_type="Unknown"):
        """Generate answer using LLM based on context"""
        
        # Format context
        context = ""
        for i, doc in enumerate(context_docs):
            if isinstance(doc, dict) and "content" in doc:
                # Format ChromaDB results
                title = doc.get("title", "Document")
                source = doc.get("source", "Unknown")
                url = doc.get("url", "#")
                content = doc.get("content", "")
                
                context += f"[{i+1}] {title}\nSource: {source} ({url})\n{content}\n\n"
            elif isinstance(doc, dict) and "snippet" in doc:
                # Format Google results
                title = doc.get("title", "")
                source = doc.get("domain", "")
                url = doc.get("link", "")
                snippet = doc.get("snippet", "")
                
                context += f"[{i+1}] {title}\nSource: {source} ({url})\n{snippet}\n\n"
            else:
                # Plain text
                context += f"[{i+1}] {doc}\n\n"
        
        prompt = f"""You are medicos, a medical AI assistant that provides accurate, well-referenced medical information.

Answer the following medical question using ONLY the provided context. Follow these rules:

1. Only use information from the provided sources
2. If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question accurately."
3. Do not make up information or cite sources not provided
4. Include numbered references to the specific sources used
5. Include a medical disclaimer at the end
6. Suggest 2-3 follow-up questions related to the topic

Question: {question}

Context:
{context}

Source type: {source_type}

Your answer should be structured as follows:
1. A concise answer to the question
2. References section with numbered sources
3. 2-3 follow-up questions
4. Medical disclaimer
"""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are medicos, a medical AI assistant that provides accurate, well-referenced information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Low temperature for factual responses
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer with Groq API: {e}")
            return "I encountered an error while generating the answer. Please try again later."
    
    def format_sources(self, sources):
        """Format sources for the response"""
        if not sources:
            return []
            
        if isinstance(sources[0], dict) and "content" in sources[0]:
            # Format ChromaDB results
            return [
                {
                    "title": source.get("title", "Unknown document"),
                    "source": source.get("source", "Unknown source"),
                    "url": source.get("url", "#")
                }
                for source in sources
            ]
        elif isinstance(sources[0], dict) and "snippet" in sources[0]:
            # Format Google results
            return [
                {
                    "title": result.get("title", ""),
                    "source": result.get("domain", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", "")
                }
                for result in sources
            ]
        else:
            # Format plain text
            return [{"title": f"Result {i+1}", "source": str(source), "url": "#"} for i, source in enumerate(sources)]
    
    def validate_database_response(self, question, chroma_results, similarity_threshold=0.3):
        """Check if database results are actually relevant"""
        if not chroma_results:
            return False
            
        # Check similarity scores
        _, distances = self.search_chroma_db(question)
        
        if distances and distances[0] > similarity_threshold:
            print(f"Database results found but relevance too low: {distances[0]}")
            return False
            
        # Use LLM to verify relevance
        validation_prompt = f"""You are evaluating if a database response is relevant to a user query.

Query: {question}

Database Response: {chroma_results[0].get('content', '')}

Task: Determine if this database response actually answers the user's query.
Answer only 'YES' if it is relevant and informative for the query, or 'NO' if it doesn't properly address the query.
"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a validation assistant determining if search results are relevant to a query."},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1
            )
            
            validation = response.choices[0].message.content
            is_valid = "YES" in validation.upper()
            print(f"LLM validation result: {validation} (Valid: {is_valid})")
            return is_valid
            
        except Exception as e:
            print(f"Error in validation: {e}")
            # Default to accepting results if validation fails
            return True
    
    def process_medical_query(self, question, use_google_fallback=True, top_k=5):
        """Main method to handle medical queries"""
        
        # Check cache first
        cached_response = self.check_query_cache(question)
        if cached_response:
            print("Using cached response")
            return cached_response
        
        # Try ChromaDB first
        chroma_results, _ = self.search_chroma_db(question, top_k)
        
        # Validate results
        if chroma_results:
            is_valid = self.validate_database_response(question, chroma_results)
            if is_valid:
                context = chroma_results
                source_type = "ChromaDB (Pre-stored Medical Docs)"
                formatted_sources = self.format_sources(chroma_results)
            else:
                # Results not relevant, use fallback
                chroma_results = []
        
        # Use Google if needed
        if not chroma_results and use_google_fallback:
            google_results = self.google_search(question, max_results=top_k)
            
            if google_results:
                context = google_results
                source_type = "Google API (Live Search)"
                formatted_sources = self.format_sources(google_results)
                
                # Store Google results for future use
                try:
                    for i, result in enumerate(google_results):
                        doc_id = f"google_{int(time.time())}_{i}"
                        snippet = result.get("snippet", "")
                        
                        # Skip short snippets
                        if len(snippet) < 50:
                            continue
                            
                        # Store in database
                        snippet_embedding = self.embedding_model.encode(snippet)
                        
                        self.collection.add(
                            ids=[doc_id],
                            embeddings=[snippet_embedding.tolist()],
                            documents=[snippet],
                            metadatas=[{
                                "source": result.get("domain", "Google Search"),
                                "title": result.get("title", "Search Result"),
                                "url": result.get("link", "#"),
                                "query": question,
                                "date_added": datetime.now().isoformat()
                            }]
                        )
                    print(f"Added {len(google_results)} Google results to the database")
                except Exception as e:
                    print(f"Error adding Google results to database: {e}")
            else:
                return {
                    "question": question,
                    "answer": "I couldn't find relevant information to answer this question accurately.",
                    "context_source": "None",
                    "sources": []
                }
        elif not chroma_results:
            return {
                "question": question,
                "answer": "I don't have enough information in my knowledge base to answer this question accurately.",
                "context_source": "None",
                "sources": []
            }
        
        # Generate answer
        answer = self.generate_answer(question, context, source_type)

        # Prepare response
        response = {
            "question": question,
            "answer": answer,
            "context_source": source_type,
            "sources": formatted_sources
        }
        
        # Cache for future use
        self.store_in_cache(question, response)

        return response

def main():
    rag = MedicalRAG()
    
    print("Welcome to medicos!")
    print("This system now validates database responses and caches results.")
    
    while True:
        user_query = input("Enter a medical query (or type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            print("Exiting Medical RAG System. Goodbye!")
            break

        print("Processing query...")
        response = rag.process_medical_query(user_query)
        print("\n" + "="*50 + "\nRESULTS\n" + "="*50)
        print(json.dumps(response, indent=2))
        print("="*50)

if __name__ == "__main__":
    main()