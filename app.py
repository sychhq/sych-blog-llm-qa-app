import streamlit as st
from PyPDF2 import PdfReader
import pickle
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title("📝 LLM Q/A App")
    st.markdown('''
    ## About
    This app is an LLM powered chat bot that can take files as input.
    - [Read full article](https://sych.io/blog/how-to-augment-chatgpt-with-your-own-data)
    - [View the source code](https://github.com/sychhq/sych-blog-llm-qa-app)
    ''')

def submit (uploaded_pdf, query, api_key):

    if uploaded_pdf:

        #Pdf Text Extraction
        pdf_reader = PdfReader(uploaded_pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        #Text Splittting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        #Compute Embeddings and Vector Store 
        store_name = uploaded_pdf.name[:4]
        if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    vector_store = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
        
        if query:
            #Accept User Queries
            docs = vector_store.similarity_search(query=query, k=2)

            #Generate Responses Using LLM
            llm = ChatOpenAI(openai_api_key=api_key, temperature=0.9, verbose=True)
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            #Callback and Query Information
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                st.header("AI Response")
                st.write(response)
                st.info(f'''
                    #### Query Information
                    Successful Requests: {cb.successful_requests}\n
                    Total Cost (USD): {cb.total_cost}\n
                    Tokens Used: {cb.total_tokens}\n
                    - Prompt Tokens: {cb.prompt_tokens}\n
                    - Completion Tokens: {cb.completion_tokens}\n 
                ''')

def main():
    st.header("LLM Q/A App")

    form = st.form(key='my_form')
    form.text_input("Your Open AI API Key", key="open_ai_api_key", type="password")
    uploaded_pdf = form.file_uploader("Upload your pdf file", type=("pdf"))
    query = form.text_area(
                "Ask something about the file",
                placeholder="Can you give me a short summary?",
                key="question"
            )
    form.form_submit_button("Run", on_click=submit(uploaded_pdf=uploaded_pdf, query=query, api_key=st.session_state.open_ai_api_key))

if __name__ == '__main__':
    main()
