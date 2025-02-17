from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, SequentialChain
from langchain_core.documents import Document
from typing import List
import streamlit as st
from langchain_core.output_parsers import StrOutputParser

def chain_init(documents:List[Document]):
    # Instantiate the embedding model
    # embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedder = OllamaEmbeddings(
        model="bge-m3",
        # num_gpu=0,
        num_thread=8
    )

    # Create vector store and retriever
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            
    # Define the LLM and the prompt
    llm = OllamaLLM(
        model="deepseek-r1:1.5b",
        # num_gpu=0,
        num_thread=8,
        temperature=0.1,
    )
    prompt = """
    1. 参考下列内容并在最后回答问题.
    2. 如果中不包含回答,,请回答文档中未提及,请不要自己编造答案.
    3. 尽可能以下列内容中原文来回答问题,并保证你的回答尽可能全面,可以尝试回答内容中所有相似信息
    4. 内容中可能存在不相关信息,请你筛选出相似度最高的信息来回答问题,其中若内容和问题中存在大量相同字符串,可认为是相似高.
        例如询问"会员状态接口信息是什么"而内容中包含"会员状态",则尽可能以该内容回答问题
    5. 如果内容中不存在完全相同的信息,请筛选其中相似度最高的信息进行回答    
    内容: {context}
    问题: {input}
    回答:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # Define the document and combination chains
    llm_chain = QA_CHAIN_PROMPT | llm | StrOutputParser()
    document_prompt = PromptTemplate(
        prompt= """
            除了{page_content},请把{source}也纳入相似度的检索范围
        """,
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