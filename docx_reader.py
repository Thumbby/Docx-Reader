import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader, UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.globals import set_verbose, set_debug
from chain import retrieval_chain_init, eval_chain_init
from util import remove_think_chain
import os
import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

set_debug(True)
def app():
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
        if "retrieval_chain" not in st.session_state:
            retrieval_chain = retrieval_chain_init(documents=documents)
            st.session_state.retrieval_chain = retrieval_chain
        else:
            retrieval_chain = st.session_state.retrieval_chain
        
        if "eval_chain" not in st.session_state:
            eval_chain = eval_chain_init()
            st.session_state.eval_chain = eval_chain
        else:
            eval_chain = st.session_state.eval_chain                 

        # Question input and response display
        st.header("❓ 开始询问与文档相关的问题")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "你好,有什么我可以帮你的吗？"}]
            st.session_state.greeting = True

        # For each message in session state
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        last_message = st.session_state.messages[-1]
        
        if last_message["role"] == "assistant":
            # React to user input
            if prompt := st.chat_input("请输入您的问题:"):
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.spinner("思考中。。。"):    
                    response = retrieval_chain.invoke({"input": prompt})                        
                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        st.markdown(response['answer'])
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                    
            # Show the evaluation button
            
            if st.session_state.greeting:
                st.session_state.greeting = False
            else:    
                if st.button("评价"):
                    st.session_state.show_eval = True
                
            if st.session_state.get("show_eval", False):
                with st.form(key="eval_form"):
                    evaluate_input = st.text_area("请输入预期回答:")
                    submitted = st.form_submit_button("提交评价")
                    if submitted:
                        # Get the final bot's reponse
                        ai_response = st.session_state.messages[-1]["content"]
                        with st.spinner("评价中..."):
                            score = eval_chain.invoke(input={
                                "ai_response": remove_think_chain(ai_response),
                                "human_response": evaluate_input
                            })
                            st.success(f"评分结果: {score}")
                        # Reset the evaluation button state
                        st.session_state.show_eval = False
                                
    else:
        st.info("请先上传文件")
    
if __name__ == "__main__":
    app()
