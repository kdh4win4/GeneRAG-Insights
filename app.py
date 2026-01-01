import streamlit as st
import os
from dotenv import load_dotenv
from src.processor import VariantProcessor
from src.retriever import BioRetriever
from src.generator import BioGenerator

# Load environment variables (API Keys)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    st.set_page_config(page_title="GeneRAG-Insights", page_icon="🧬", layout="wide")
    
    st.title("🧬 GeneRAG-Insights")
    st.markdown("### Automated Clinical Interpretation of Genetic Variants")
    st.divider()

    # Sidebar for Configuration
    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("Enter OpenAI API Key", value=OPENAI_API_KEY, type="password")
    uploaded_file = st.sidebar.file_uploader("Upload Reference Literature (PDF)", type=["pdf"])

    # Initialize Components
    if api_key:
        processor = VariantProcessor()
        retriever = BioRetriever(api_key)
        generator = BioGenerator(api_key)

        # 1. Document Ingestion
        if uploaded_file:
            with st.spinner("Indexing document..."):
                # Save uploaded file temporarily
                with open("temp_paper.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                status = retriever.ingest_documents("temp_paper.pdf")
                st.sidebar.success(status)

        # 2. User Input
        variant_input = st.text_input("Enter Genetic Variant (e.g., EGFR T790M):", placeholder="BRAF V600E")

        if st.button("Generate Clinical Report"):
            if variant_input:
                with st.spinner("Analyzing variant and searching literature..."):
                    # Process Input
                    parsed_data = processor.parse_variant_string(variant_input)
                    query = processor.format_query_for_rag(parsed_data)
                    
                    # Retrieve Context
                    context_docs = retriever.search_relevant_context(query)
                    
                    # Generate Report
                    report = generator.generate_clinical_report(variant_input, context_docs)
                    
                    # Display Results
                    st.subheader("📋 Clinical Interpretation Report")
                    st.markdown(report)
                    
                    with st.expander("View Retrieved Evidence"):
                        for i, doc in enumerate(context_docs):
                            st.write(f"**Source {i+1}:** {doc.page_content[:500]}...")
            else:
                st.warning("Please enter a variant to analyze.")
    else:
        st.info("Please enter your OpenAI API Key in the sidebar to start.")

if __name__ == "__main__":
    main()
