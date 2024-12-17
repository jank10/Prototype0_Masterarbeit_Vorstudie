#this code has been inspired by https://github.com/JohannesJolkkonen/funktio-ai-samples/blob/main/knowledge-graph-demo/main.py

import streamlit as st
from streamlit_chat import message
from timeit import default_timer as timer

from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain

import dotenv
import os

dotenv.load_dotenv()

# OpenAI API configuration

llm = ChatOpenAI(
    model="gpt-4o-mini",                
    temperature=0,                
    max_tokens=1500,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


#Neo4j configuration
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")


CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is empty, say that you don't know the answer.
Only use the information in the provided context that directly contributes to precisely answering the question.

Final answer should be easily readable and structured.
Information:
{context}

Question: {question}
Helpful Answer:"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)



embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
vector_index_skill = Neo4jVector.from_existing_index(
    embedding = embeddings_model,
    index_name = "skill_index_vector",
    keyword_index_name = "skill_index_fulltext",
    search_type = "hybrid"
)

vector_index_project = Neo4jVector.from_existing_index(
    embedding = embeddings_model,
    index_name = "project_index_vector",
    keyword_index_name = "project_index_fulltext",
    search_type = "hybrid"
)




def query_graph(user_input):
    # Perform vector search to find related terms or entities based on the input
    vector_results_skill = vector_index_skill.similarity_search(user_input, k=1)  
    vector_related_terms_skill = [el.page_content for el in vector_results_skill]  

    vector_results_project = vector_index_project.similarity_search(user_input, k=1)  
    vector_related_terms_project = [el.page_content for el in vector_results_project]  

    # Combine results from both searches
    combined_vector_related_terms = vector_related_terms_skill + vector_related_terms_project
    vector_context = ", ".join(combined_vector_related_terms) if combined_vector_related_terms else "None"


    
    cypher_generation_template_with_vector = """
    You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided and a vector similarity search, following the instructions below:
    0. Your main focus is to write queries that identify people within the knowledge graph
    1. Generate Cypher query compatible ONLY for Neo4j Version 5
    2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
    3. Use only Nodes and relationships mentioned in the schema
    4. Only search for skills or projects contained in this list: {vector_context}. Build the terms into your search query.
    5. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Client, use `toLower(client.id) contains 'neo4j'`. 
        To search for skills, use 'toLower(skill.name) contains 'neo4j'`. 
        To search for a project, use `toLower(project.name) contains 'sales training'.
    6. Never use relationships that are not mentioned in the given schema
    7. Add to every query the syntax to extract the names of related skill and projects nodes. For example:
        MATCH (person:Person)-[:HAS_SKILL]->(skill:Skill)
        WHERE toLower(skill.name) CONTAINS 'budget' OR toLower(skill.name) CONTAINS 'budgeting'
        MATCH (person)-[r1]-(sk:Skill)
        MATCH (person)-[r2]-(pr:Project)
        RETURN person.name, collect(DISTINCT sk.name) AS skill_names, collect(DISTINCT pr.name) AS project_names;
    8. For every Person node that you extract, also extract the property called profile_summary. For example:
        MATCH (person:Person)-[:HAS_SKILL]->(skill:Skill)
        WHERE toLower(skill.name) CONTAINS 'account management'
        MATCH (person)-[r1]-(sk:Skill)
        MATCH (person)-[r2]-(pr:Project)
        RETURN person.name, person.profile_summary, collect(DISTINCT sk.name) AS skill_names, collect(DISTINCT pr.name) AS project_names;   
    

    schema: {schema}

    

    Question: {query}
    """

    # Create the prompt template with vector context as an input variable
    cypher_prompt_with_vector = PromptTemplate(
        template=cypher_generation_template_with_vector,
        input_variables=["schema", "query", "vector_context"]
    )

    graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_password)
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        cypher_prompt=cypher_prompt_with_vector,
        qa_prompt=qa_prompt,
        allow_dangerous_requests=True
    )

    #result = chain(user_input)
    result = chain({
    "vector_context": vector_context,
    "query": user_input
    })
    
    return result
    

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    Conversation:
    {chat_history}
    Follow-up question: {question}
    Standalone question:"""
)


st.set_page_config(layout="wide")

if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "context" not in st.session_state:
    st.session_state["context"] = None

if "input" not in st.session_state:
    st.session_state["input"] = ""



def handle_user_input(user_input):
    if not st.session_state["chat_history"]:
        # No chat history, perform a graph search
        result = query_graph(user_input)
        answer = result["result"]
        intermediate_steps = result["intermediate_steps"]
        cypher_query = intermediate_steps[0]["query"]
        database_results = intermediate_steps[1]["context"]
        context = database_results
        st.session_state["context"] = context
        if answer != "I don't know the answer." or database_results == []:
            st.session_state["chat_history"].append((user_input, answer))
        print(answer)
        return answer, cypher_query, database_results
    else:
        # Use chat history to rewrite follow-up question
        formatted_history = "\n".join(
            [f"Human: {q}\nAI: {a}" for q, a in st.session_state["chat_history"]]
        )
        print("Formatted History:", formatted_history)


        standalone_prompt = CONDENSE_QUESTION_PROMPT.format(
            chat_history=formatted_history,
            question=user_input
        )
        standalone_question = llm.invoke(standalone_prompt).content
        print("Standalone Question", standalone_question)
        
        # Use RAG chain for follow-up question
        answer_prompt = qa_prompt.format(
            context=st.session_state["context"],
            question=standalone_question
        )
        answer = llm.invoke(answer_prompt).content
        st.session_state["chat_history"].append((user_input, answer))
        print("Chat History:", st.session_state["chat_history"])
        
        cypher_query=None
        database_results=None
        return answer, cypher_query, database_results


title_col, empty_col, img_col = st.columns([2, 1, 2])    

with title_col:
    st.markdown("# People Search\n### Prototype0\n")

    debug_mode = st.checkbox("Debug Mode", value=False)



with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("Enter your question", key="input")
    
    col_submit, col_reset, empty_col = st.columns([1, 1, 10])
    with col_submit:
        submit = st.form_submit_button("Submit")
    with col_reset:
        reset = st.form_submit_button("Reset")
    
if reset:
    st.session_state.user_msgs = []
    st.session_state.system_msgs = []
    st.session_state.chat_history = []
    st.session_state.context = None
    st.success("Chat has been reset!")

if submit:
    st.session_state.user_msgs.append(user_input)
    
    
    with st.spinner("Working on your question..."):
        
        #start = timer()

        cypher_query = None
        database_results = None

        try:
            answer, cypher_query, database_results = handle_user_input(user_input)
            st.session_state.system_msgs.append(answer)

        except Exception as e:
            st.write("Failed to process question. Please try again.")
            print(e)
            print(f"Generated Cypher Query: {cypher_query}")
               
    
    
    

    # Display the chat history
    if debug_mode:
        # 3 columns needed if we want to show what's going on under the hood of the code
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.session_state["system_msgs"]:
                for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
                    message(st.session_state["system_msgs"][i], key = str(i) + "_assistant")
                    message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")
        
        with col2:
            if cypher_query:
                st.text_area("Last Cypher Query", cypher_query, key="_cypher", height=240)
            
        with col3:
            if database_results:
                st.text_area("Last Database Results", database_results, key="_database", height=240)
    else:
        col1,_ = st.columns([5,1])

        # Display the chat history
        with col1:
            if st.session_state["system_msgs"]:
                for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
                    message(st.session_state["system_msgs"][i], key = str(i) + "_assistant")
                    message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")


        
    