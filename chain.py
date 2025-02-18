from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, SequentialChain
from langchain_core.documents import Document
from typing import List
import streamlit as st
from langchain_core.output_parsers import StrOutputParser

def retrieval_chain_init(documents:List[Document]):
    # Instantiate the embedding model
    # embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedder = OllamaEmbeddings(
        model="bge-m3",
        num_thread=8
    )

    # Create vector store and retriever
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            
    # Define the LLM and the prompt
    llm = OllamaLLM(
        model="deepseek-r1:1.5b",
        num_thread=8,
        temperature=0.1,
        top_p=0.5,
        top_k=10
    )
    prompt = """
        #角色
        你是一个智能文档分析助手，拥有以下能力：
        1. 精准判断问题是否需要基于用户上传的文档回答
        2. 当需要文档时严格依据检索内容回答
        3. 处理通用问题时展现自然对话能力
        #判断规则
        按以下逻辑处理问题：
        1. 收到用户问题
        2. 判断问题是否需要通过文档内容回答
        3. 若需要,则进行文档回答;若不需要,则进行通用回答
        #文档回答要求
        1. 禁止编造文档不存在的内容
        2. 若用户问题中存在"以文档原文内容回答"或相关描述,请直接回答文档内容
        #通用回答要求
        1. 使用口语化表达
        2. 禁止出现"根据文档"等误导性表述    
    文档内容: {context}
    用户问题: {input}"""
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

def eval_chain_init():
    
    prompt_template = """
    #角色
    你是一个大模型回答评论机器,你会根据AI回答和人类回答进行评分
    #回答标准
    1.以人类回答为标准给大模型打分
    2.分数区间为0-5分,AI回答越贴近人类回答则越接近5分,越不符合人类回答则越接近0分,接近的标准为:AI回答是否包含人类回答中信息原文或者含义相近内容
    3.最终在0-5分区间内允许小数,但最多只能包含两位小数,如:4.51
    4.你的回答只需要包含一个数字即可,**不允许**包含除了分数以外的任何内容
    AI回答:{ai_response}
    人类回答:{human_response}
    """
    EVAL_CHAIN_PROMPT = PromptTemplate(
        input_variables=["ai_response", "human_response"],
        template = prompt_template
    )
    
    llm = OllamaLLM(
        model="deepseek-r1:8b",
        num_thread=8,
        temperature=0.1,
        top_k=10,
        top_p=0.5
    )
    
    eval_chain = (EVAL_CHAIN_PROMPT | llm | StrOutputParser())
    
    return eval_chain
    