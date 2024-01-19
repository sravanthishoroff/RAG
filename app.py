import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from htmlTemplates import css, bot_template, user_template
import traceback

# Include the source PDF name in the text chunks
def get_text_chunks_with_source(pdf_docs):
    chunks = []
    source_map = {}
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_name = pdf.name
        for page_number, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                chunk_id = f"{pdf_name}_page_{page_number}"
                chunks.append(text)
                source_map[chunk_id] = pdf_name
    return chunks, source_map

def  get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = load_qa_chain(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Error handling function
def handle_error():
    error_message = traceback.format_exc()
    st.error("An error occurred: \n" + error_message)

def handle_userinput(user_question, source_map):
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            display_message = message.content
            if i % 2 != 0 and message.source in source_map:  # Bot response with source
                source_pdf = source_map[message.source]
                display_message += f" ([Source PDF]({source_pdf}))"

            if i % 2 == 0:  # User message
                st.write(user_template.replace("{{MSG}}", display_message), unsafe_allow_html=True)
            else:  # Bot message
                st.write(bot_template.replace("{{MSG}}", display_message), unsafe_allow_html=True)
    except Exception:
        handle_error()

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    submit_question = st.button("Submit Question")

    source_map = {}  # Initialize source_map here

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        process_docs = st.button("Process Documents")

        if process_docs:
            with st.spinner("Processing..."):
                try:
                    text_chunks, source_map = get_text_chunks_with_source(pdf_docs)  # Update source_map
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                except Exception:
                    handle_error()

    if submit_question and user_question and st.session_state.conversation:
        handle_userinput(user_question, source_map)


if __name__ == '__main__':
    main()
