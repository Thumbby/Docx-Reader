import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader, UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.globals import set_verbose, set_debug
from chain import chain_init
import re
import os
import torch


torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# set_debug(True)
def main():
    # App title
    st.title("📄 需求文档分析")

    # Sidebar for instructions and settings
    with st.sidebar:
        st.header("使用步骤")
        st.markdown("""
        1. 上传需求文档(以docx格式).
        2. 询问与文档相关的问题.
        3. 系统会根据文档尝试回答您的问题.
        """)

        st.header("设置")
        st.markdown("""
        - **Embedding Model**: HuggingFace
        - **Retriever Type**: Similarity Search
        - **LLM**: DeepSeek R1 1.5b(Ollama)
        """)

        # Main file uploader section
        st.header("📁 上传您的需求文档")
        uploaded_file = st.file_uploader("此处上传您的需求文档", type="docx")

    if uploaded_file is not None:
        st.success("需求文档上传成功！正在解析...")

        with open("temp.docx", "wb") as f:
            f.write(uploaded_file.getvalue())
            
        loader = Docx2txtLoader('temp.docx')
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=256)
        documents = text_splitter.split_documents(docs)    
        
        # Initialize chain
        retrieval_chain = chain_init(documents=documents)    

        # Question input and response display
        st.header("❓ 开始询问与文档相关的问题")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "你好,有什么我可以帮你的吗？"}]

         
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # React to user input
        if prompt := st.chat_input("请输入您的问题:"):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})    
                            
            response = retrieval_chain.invoke({"input": prompt})
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                                
    else:
        st.info("请先上传文件")
    
if __name__ == "__main__":
    main()