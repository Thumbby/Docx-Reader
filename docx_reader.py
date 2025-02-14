import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain, create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, SequentialChain
from langchain.globals import set_verbose, set_debug
import os
import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# set_debug(True)

def chain_init():
    loader = Docx2txtLoader('temp.docx')
    docs = loader.load()

    # Split the document into chunks
    # text_splitter = SemanticChunker(OllamaEmbeddings(
    #     model="bge-m3",
    #     num_gpu=0,
    #     num_thread=4
    # ))
    # text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=100, separator="\n")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=256)
    documents = text_splitter.split_documents(docs)
    
    with open('chunk_output.txt', 'w', encoding='utf-8') as f:
    # print chunks
        f.write('\n'.join(map(str,documents)))
            
    # Instantiate the embedding model
    # embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedder = OllamaEmbeddings(
        model="bge-m3",
        # num_gpu=0,
        num_thread=4
    )

    # Create vector store and retriever
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            
    # Define the LLM and the prompt
    llm = OllamaLLM(
        model="deepseek-r1:1.5b",
        # num_gpu=0,
        num_thread=4,
        temperature=0.1,
    )
    prompt = """
    1. 参考下列内容并在最后回答问题.
    2. 如果中不包含回答,,请回答文档中未提及,请不要自己编造答案.
    3. 尽可能以下列内容中原文来回答问题
    4. 内容中可能存在不相关信息,请你筛选出相似度最高的信息来回答问题,其中若内容和问题中存在大量相同字符串,可认为是相似高.
        例如询问"会员状态接口信息是什么"而内容中包含"会员状态",则尽可能以该内容回答问题
    内容: {context}
    问题: {input}
    回答:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # Define the document and combination chains
    llm_chain = QA_CHAIN_PROMPT | llm | StrOutputParser()
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )
    combine_documents_chain = create_stuff_documents_chain(
        llm=llm,
        prompt= QA_CHAIN_PROMPT,
        document_variable_name="context",
        document_prompt=document_prompt,
    )
    
    retrieval_chain = create_retrieval_chain(
        retriever = retriever,
        combine_docs_chain = combine_documents_chain,    
    )
    
    return retrieval_chain
        
    

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
        
        retrieval_chain = chain_init()    

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