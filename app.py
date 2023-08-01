import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.pinecone import Pinecone
#from langchain import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os

def db(texts,text_splitter,api):
    
    chunks = text_splitter.split_text(texts)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_texts(chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm=OpenAI(temperature=0), retriever=retriever, chain_type="refine",memory=memory)
    return qa

def ai(prompt):
    file = open("save.txt","r")
    system_prompt = str(file.read()) 
    print(system_prompt)
    file.close()
    #prompt=system_prompt+str(": question is :")+prompt
    result = qa({"question": prompt,  "chat_history": chat_history})
    return result["answer"]
def set_environment_variable(api_key):
    # Set the environment variable for the OpenAI API key
    os.environ["OPENAI_API_KEY"] = api_key
def main():
    global qa, chat_history
    placeholder=st.empty()
    placeholder.empty()
    placeholder.title("your openai api key")
    global api
    api=st.text_input("enter here")
    if st.button("Load API Key"):
        # Set the environment variable with the provided API key
        set_environment_variable(api)
        st.success("API key loaded successfully!")
    placeholder.title("Upload or Chat PDF")
    #st.header("PDF/URL QA")
    #global system_prompt
    names = ['Upload', 'Random talk']
    page = st.radio('Format', names)

    if page == 'Upload':
        pdf = st.file_uploader("Upload your PDF", type="pdf")
        if pdf is not None:
            #print(pdf)
            pdf_reader = PdfReader(pdf)
            texts = ""
            for page in pdf_reader.pages:
                texts += page.extract_text()
            text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size = 1000,
            chunk_overlap = 0
        )
            qa=db(texts,text_splitter,api)
            chat_history = []
            st.header("PDF/URL QA")
            query = st.text_input("Ask a question in PDF")
            if query:
                output = ai(query)
                chat_history=chat_history.append(query)
                st.write(output)

    elif page == 'Random talk':
            chat_history=[]
            st.header("Start Chatting")
            message=st.text_input("Your message")
            if st.button('reply'):
                prompt = "\"Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations. " + message + "\""
                # Call the OpenAI Api to process our prompt
                openai_response = openai.Completion.create(model="text-davinci-003", prompt=prompt,max_tokens=4000)
                print("openai response:", openai_response)
                # Parse the response to get the response text for our prompt
                response_text = openai_response.choices[0].text
                st.write( response_text)


if __name__ == "__main__":
    main()