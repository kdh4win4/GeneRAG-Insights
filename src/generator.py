from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class BioGenerator:
    """
    Synthesizes clinical reports using LLM based on retrieved biological context.
    """
    def __init__(self, openai_api_key):
        # We use GPT-4o for high-quality clinical reasoning
        # Temperature is set to 0 to ensure factual consistency
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            openai_api_key=openai_api_key,
            temperature=0
        )

    def generate_clinical_report(self, variant_query, context_docs):
        """
        Creates a structured clinical interpretation report.
        """
        # 1. Combine content from all retrieved snippets
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        # 2. Craft a professional system prompt
        template = """
        SYSTEM: You are a world-class Clinical Geneticist and Bioinformatician.
        Your task is to interpret the following genetic variant based ONLY on the provided research context.
        If the context is insufficient, state that more clinical evidence is needed.
        
        CONTEXT:
        {context}
        
        VARIANT QUERY:
        {variant_query}
        
        INSTRUCTIONS:
        - Provide a summary of the variant's clinical significance.
        - Mention potential therapeutic implications if available.
        - Maintain a professional, medical-grade tone.
        - Structure the output using Markdown.
        
        ANSWER:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 3. Define the LCEL (LangChain Expression Language) Chain
        chain = (
            {"context": lambda x: context_text, "variant_query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # 4. Invoke the chain to get the result
        return chain.invoke(variant_query)
