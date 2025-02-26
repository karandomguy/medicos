import streamlit as st
import requests
import json
import time

# Set page configuration
st.set_page_config(
    page_title="medicos - Medical Question Answering",
    page_icon="🩺",
    layout="wide"
)

# Define API endpoint
API_URL = "http://localhost:8000/api/query"

def query_medical_rag(question, use_google_fallback=True, top_k=5):
    """Make API call to the medical RAG backend"""
    try:
        payload = {
            "question": question,
            "use_google_fallback": use_google_fallback,
            "top_k": top_k
        }
        
        with st.spinner("Searching medical knowledge base..."):
            response = requests.post(API_URL, json=payload)
            
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

st.title("🩺 medicos - Medical Question Answering")
st.subheader("Ask medical questions and get evidence-based answers")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    use_google = st.checkbox("Use Google Search Fallback", value=True, 
                            help="If enabled, the system will search Google when its knowledge base doesn't have relevant information")
    
    top_k = st.slider("Number of sources to retrieve", min_value=1, max_value=10, value=5, 
                     help="How many documents to retrieve from the knowledge base")
    
    st.divider()
    st.markdown("### About Medicos")
    st.info("""
    Medicos is a Retrieval-Augmented Generation (RAG) system for medical questions. 
    It uses a combination of pre-stored medical knowledge and (optionally) live Google search results.
    
    **Note:** This is for informational purposes only and not a substitute for professional medical advice.
    """)

# Main interaction area
query = st.text_area("Enter your medical question:", height=100, 
                    placeholder="e.g., What are the symptoms of type 2 diabetes?")

col1, col2 = st.columns([1, 5])
with col1:
    submit_button = st.button("Submit", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("Clear Results", use_container_width=False)

# Process the query when the button is clicked
if submit_button and query:
    # Store the query and response in session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    result = query_medical_rag(query, use_google_fallback=use_google, top_k=top_k)
    
    if result:
        # Add to history
        st.session_state.history.append(result)
        
        # Display the answer
        st.markdown("## Answer")
        st.markdown(result["answer"])
        
        # Display sources
        st.markdown("## Sources")
        source_type = result["context_source"]
        st.info(f"Source Type: {source_type}")
        
        # Create tabs for each source
        if result["sources"]:
            tabs = st.tabs([f"Source {i+1}: {s['title'][:30]}..." for i, s in enumerate(result["sources"])])
            
            for i, (tab, source) in enumerate(zip(tabs, result["sources"])):
                with tab:
                    st.markdown(f"**Title:** {source['title']}")
                    st.markdown(f"**Source:** {source['source']}")
                    st.markdown(f"**URL:** [{source['url']}]({source['url']})")
                    if "snippet" in source and source["snippet"]:
                        st.markdown("**Snippet:**")
                        st.markdown(f"> {source['snippet']}")
        else:
            st.warning("No sources were found for this query.")

# clear results 
if clear_button:
    if 'history' in st.session_state:
        st.session_state.history = []
    st.rerun()

# show history
if 'history' in st.session_state and st.session_state.history:
    st.divider()
    with st.expander("Search History", expanded=False):
        for i, item in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**Query {len(st.session_state.history)-i}:** {item['question']}")
            if st.button(f"View Answer {len(st.session_state.history)-i}", key=f"history_{i}"):
                st.session_state.selected_history = item
                st.rerun()

# Show selected history item
if 'selected_history' in st.session_state:
    st.divider()
    st.markdown("## Previous Result")
    st.markdown(f"**Question:** {st.session_state.selected_history['question']}")
    st.markdown(st.session_state.selected_history["answer"])
    
    # Clear selected history when done
    if st.button("Close Previous Result"):
        del st.session_state.selected_history
        st.rerun()
