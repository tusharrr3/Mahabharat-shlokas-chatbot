
import streamlit as st
import sys
import os
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from main import MahabharatChatbot

# Page configuration
st.set_page_config(
    page_title="Mahabharat RAG Chatbot",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .verse-card {
        background-color: #f8f9fa;
        border-left: 4px solid #FF6B35;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .verse-number {
        color: #FF6B35;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .shlok-text {
        font-size: 1.3rem;
        color: #2c3e50;
        font-family: 'Noto Sans Devanagari', serif;
        margin: 1rem 0;
        line-height: 1.8;
    }
    .translation-text {
        color: #34495e;
        font-size: 1.1rem;
        font-style: italic;
        margin: 1rem 0;
    }
    .meaning-text {
        color: #555;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-top: 1rem;
    }
    .confidence-badge {
        display: inline-block;
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .not-found {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        color: #856404;
    }
    .stat-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #0066cc;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    """Load and cache the chatbot instance."""
    with st.spinner("🔄 Initializing chatbot... This may take a moment on first run..."):
        return MahabharatChatbot()


def display_verse_response(response):
    """Display verse response in a beautiful card format."""
    st.markdown(f"""
    <div class="verse-card">
        <div class="verse-number">📖 Verse {response['verse_number']}</div>
        <div style="margin-top: 1rem;">
            <span class="confidence-badge">Confidence: {response['confidence_score']*100:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show LLM response if available
    if "llm_response" in response and response["llm_response"]:
        st.subheader("💡 Analysis & Insights")
        st.markdown(response["llm_response"])
    
    # Show full content in expander
    with st.expander("📚 View Complete Verse Details"):
        st.markdown(f'<div class="meaning-text">{response["content"]}</div>', unsafe_allow_html=True)


def display_not_found():
    """Display not found message."""
    st.markdown("""
    <div class="not-found">
        <strong>⚠️ Answer Not Found</strong><br>
        Your query doesn't match any verses in the knowledge base. 
        Please try rephrasing or ask about topics covered in the Bhagavad Gita.
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">📖 Mahabharat RAG Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about the Bhagavad Gita and receive verses with context</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/book.png", width=80)
        st.title("About")
        st.markdown("""
        This is a **Retrieval Augmented Generation (RAG)** chatbot that answers 
        questions about the Bhagavad Gita using semantic search and AI insights.
        
        **Features:**
        - Semantic search powered by FAISS
        - LLM-enhanced responses with Gemini
        - 701 verses from the Bhagavad Gita
        """)
        
        st.divider()
        
        # Statistics
        try:
            chatbot = load_chatbot()
            stats = chatbot.get_stats()
            
            st.subheader("📊 System Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{stats['num_documents']}</div>
                    <div class="stat-label">Total Verses</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{stats['top_k']}</div>
                    <div class="stat-label">Top Results</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"**Similarity Threshold:** {stats['similarity_threshold']}")
            st.markdown(f"**Vector Store:** {stats['vector_store_type']}")
            
        except Exception as e:
            st.error(f"Error loading statistics: {e}")
    
    # Main content
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Query input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input(
            "Ask your question:",
            placeholder="e.g., What does Krishna say about dharma?",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Process query
    if search_button and query:
        try:
            # Load chatbot
            chatbot = load_chatbot()
            
            # Get response
            with st.spinner("🔍 Searching for relevant verses..."):
                response = chatbot.query(query)
            
            # Add to chat history
            st.session_state.chat_history.insert(0, {
                'query': query,
                'response': response,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.markdown("---")
            st.subheader("📝 Query Result")
            
            if "answer" in response:
                display_not_found()
            else:
                display_verse_response(response)
                
                # Download button for response
                st.download_button(
                    label="📄 Download Response (JSON)",
                    data=json.dumps(response, indent=2, ensure_ascii=False),
                    file_name=f"verse_{response['verse_number']}.json",
                    mime="application/json"
                )
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📜 Query History")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
        
        for i, item in enumerate(st.session_state.chat_history[:5]):  # Show last 5 queries
            with st.expander(f"🔍 {item['query']} - {item['timestamp']}", expanded=(i==0)):
                if "answer" in item['response']:
                    display_not_found()
                else:
                    display_verse_response(item['response'])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
       
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
