import streamlit as st
from langchain.document_loaders import Docx2txtLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

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

        # Split the document into chunks
        text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        documents = text_splitter.split_documents(docs)
        
        # Instantiate the embedding model
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create vector store and retriever
        vector = FAISS.from_documents(documents, embedder)
        
        # Define the LLM and the prompt
        llm = Ollama(model="deepseek-r1:8b")
        prompt = """
        1. 参考下列内容并在最后回答问题.
        2. 如果你不知道答案,请回答文档中未提及,请不要自己编造答案.
        3. 尽可能以下列内容中原文来回答问题
        内容: {context}
        问题: {question}
        回答:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

        # Define the document and combination chains
        llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="Context:\ncontent:{page_content}\nsource:{source}",
        )
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
            verbose=True
        )
        
        # Question input and response display
        st.header("❓ 开始询问与文档相关的问题")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
         
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # React to user input
        if prompt := st.chat_input("请输入您的问题:"):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
                
            retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 1})    
            retriever.invoke(prompt)
            qa = RetrievalQA(
                combine_documents_chain=combine_documents_chain,
                retriever=retriever,
                verbose=True,
                return_source_documents=True
            )
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display assistant response in chat message container
            response = qa(prompt)["result"]
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
                                
    else:
        st.info("请先上传文件")
    
if __name__ == "__main__":
    main()