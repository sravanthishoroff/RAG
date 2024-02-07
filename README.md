# Document QA System with OpenAI 

This project demonstrates a Document Question-Answering System using OpenAI GPT-3.5 Turbo and PyPDF2. It allows users to ask questions, and the system finds the best answer from a collection of PDF documents.

## Prerequisites

- Python 3.x
- OpenAI API key
- Streamlit
- PyPDF2

## Installation

- Clone the repository:

   ```bash
   git clone https://github.com/sravanthi.shoroff/RAG.git
   cd RAG
  ```
- Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
- Set up OpenAI API key:
    Set your OpenAI API key as an environment variable. You can export it in your terminal or set it directly in your script.
    ```bash
    export OPENAI_API_KEY=your_openai_api_key_here
    ```
## Usage
- Run the Streamlit app
    ```bash
    streamlit run app.py
    ```
## Configuration 
- Adjust the code in app.py as needed, such as updating the default PDF folder path or modifying the UI elements.

## Acknowledgments
- OpenAI for providing the GPT-3.5 Turbo model
- Streamlit and PyPDF2 communities for their useful libraries
