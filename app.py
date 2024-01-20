import os
import streamlit as st
from openai import OpenAI
import PyPDF2

client = OpenAI()

# Set your OpenAI API key
OpenAI.api_key = os.environ["OPENAI_API_KEY"]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to get the answer using OpenAI API
def get_openai_answer(question, context):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Question: {question}\nContext: {context}\nAnswer:",
        temperature=0.5,
        max_tokens=100,
        n=1,
        stop=None,
    )
    return response.choices[0].text.strip()

# Streamlit app
def main():
    st.title("Document QA System")

    # User input
    user_question = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        # Select PDF folder
        pdf_folder = "C:\\Users\\Sravanthi\\Desktop\\Use-Cases\\GenAI\\RAG\\pdfs"

        # Get a list of PDF files in the folder
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

        # Initialize variables to store the best answer and its source PDF
        best_answer = ""
        best_pdf_path = ""

        # Iterate through each PDF in the folder
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)

            # Extract text from the PDF
            pdf_text = extract_text_from_pdf(pdf_path)

            # Get the answer from OpenAI
            answer = get_openai_answer(user_question, pdf_text)

            # If the answer is better than the current best, update the variables
            if len(answer) > len(best_answer):
                best_answer = answer
                best_pdf_path = pdf_path

        # Display the best answer and its source PDF
        st.subheader("Best Answer:")
        st.write(best_answer)
        st.subheader("Source PDF:")
        st.write(best_pdf_path)

if __name__ == "__main__":
    main()
