import os
import time
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


# os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_API_KEY']=st.secrets['LANGCHAIN_API_KEY']

os.environ['LANGCHAIN_TRACING_V2']='true'

# os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
os.environ['HF_TOKEN']=st.secrets['HF_TOKEN']

os.environ['LANGCHAIN_PROJECT']="PDF CHATBOT WITH HISTORY"

# GROQ_API_KEY=os.getenv('GROQ_API_KEY')
GROQ_API_KEY=st.secrets['GROQ_API_KEY']


st.sidebar.title('Settings')
engine=st.sidebar.selectbox('Select Engine',['Gemma2-9b-It','Gemma-7b-It','Llama3-70b-8192','Llama3-8b-8192','Mixtral-8x7b-32768',
                                             'Llama-3.1-70b-Versatile','Llama-3.2-90b-Text-Preview'])
Temperature=st.sidebar.slider('Temperature',min_value=0.1,max_value=1.0,step=0.10,value=0.7)
max_token=st.sidebar.slider('Maximum Token',min_value=10,max_value=1000,step=10,value=50)


llm_model=ChatGroq(model=engine,api_key=GROQ_API_KEY,temperature=Temperature,max_tokens=max_token)
st.title('AI PDF CHATBO WITH HISTORY')
session_id=st.text_input('Session_ID',value='default_session')
uploaded_files=st.file_uploader('Upload the PDF file',type='pdf',accept_multiple_files=True)

if 'store' not in st.session_state:
    st.session_state.store={}


if uploaded_files:
    documents=[]
    for upladed_file in uploaded_files:
        tempfile=f'./temp.pdf'
        with open(tempfile,'wb') as file:
            file.write(upladed_file.getvalue())
            file_name=upladed_file.name
            loader=PyPDFLoader(tempfile)
            docs=loader.load()
            documents.extend(docs)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
    splits=text_splitter.split_documents(documents)
    vectorstore=FAISS.from_documents(documents=splits,embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))
    FAISS_ret=vectorstore.as_retriever()
    
    context_qna_system_prompt=(
            "given a chat history and the latest user question"
            'which might reference context in the chat history,'
            'formulate a standalone question which can be understood'
            'without the chat history. Do not answer the question,'
            'just reformulate it if needed and otherwise return it as is.'
        )
                    
    context_qna_prompt=ChatPromptTemplate.from_messages([
        ('system',context_qna_system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human','{input}')
    ])

    history_aware_retriever=create_history_aware_retriever(llm=llm_model,retriever=FAISS_ret,prompt=context_qna_prompt)

    qna_system_prompt=(
        'you are an good assistant for question-answering tasks'
        'use the following pieces of retreived context to answer'
        'the question. if you dont know the answer then say you dont know'
        'use maximum 3 sentences, keep answer concise'
        '\n\n'
        '{context}'
        )
    qna_prompt=ChatPromptTemplate.from_messages([
        ('system',qna_system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human','{input}')
    ])

    question_answer_chain=create_stuff_documents_chain(llm=llm_model,prompt=qna_prompt)
    rag_chain=create_retrieval_chain(retriever=history_aware_retriever,combine_docs_chain=question_answer_chain)
    
    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain=RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_session_history,
        input_messages_key='input',
        output_messages_key='answer',
        history_messages_key='chat_history'
    )
    user_input=st.text_input('Enter your query ')
    if user_input:
        session_history=get_session_history(session_id)
        response=conversational_rag_chain.invoke(
            {'input':user_input},config={'configurable':{'session_id':session_id}}
        )
        st.write(st.session_state.store)
        st.write("Assistant: \n",response['answer'])
        st.write('Chat History: \n',session_history.messages)










