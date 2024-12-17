# Prototype0  
### *Context-aware People Search: Leveraging LLMs for Enhanced User Experiences*

Welcome to **Prototype0**, the initial Proof-of-Concept (PoC) for my master thesis, *“Context-aware people search: leveraging LLMs for enhanced user experiences”*.  

Below, you'll find all the necessary background information and steps to set up and test the prototype on your local machine.

---

## Table of Contents

1. [Preliminary Tasks](#preliminary-tasks)
2. [Setup Neo4j](#setup-neo4j)
3. [Enrich Knowledge Graph with Insights](#enrich-knowledge-graph-with-insights)
4. [Test the Prototype](#test-the-prototype)
5. [References](#references)

---

## Preliminary Tasks

Before starting, ensure you complete the following steps:

1. **Fork** this repository and clone it to a local directory of your choice.
2. **Create a virtual environment**:
   - For example using Anaconda, run:  
     ```bash
     conda create --name myenv python=3.11
     conda activate myenv
     ```
3. **Install dependencies**:  
   Run the following command to install the required packages:  
   ```bash
   pip install -r requirements.txt
4. **Ensure docker is running** on your local machine.
5. Open the cloned repository in an IDE of your choice (e.g., **Visual Studio Code**).
6. Create a ```.env``` file in the root directory with the following content:
   ```
   OPENAI_API_KEY="your_openai_api_key"
   NEO4J_URI="bolt://localhost:7687"
   NEO4J_USERNAME="neo4j"
   NEO4J_PASSWORD="your_password" //leave it just like this
   ```
---

## Setup Neo4j

1. **Start Neo4j Database**:
   Execute the ```docker-compose.yaml``` file to start a containerized Neo4j instance:
   ```docker-compose up```
3. **Access the Neo4j browser at**:
```http://localhost:7474/browser```. Use the login details from your .env file to log in to Neo4j.
The database will initially be empty.
4. **Populate the Knowledge Graph**:
Open and run the Jupyter Notebook: ```0_knowledge_graph_construction.ipynb```.
This notebook executes Cypher statements from the cypher_statements folder to build a graph representing:
Employees, their relationships to colleagues, teams, and superiors.
Emails, chats, and documents they produce.
5. **Test the Graph**:
Run the following Cypher queries in the Neo4j browser:
  - See the entire graph:
  ```
  MATCH (n) 
  RETURN n;
  ```
  - Inspect relationships for a specific person:
  ```
  MATCH (p:Person {name: "Daniel Roberts"})-[r]-(n)
  RETURN p, r, n;
  ```

---

## Enrich Knowledge Graph with Insights
To enhance the graph with insights:

1. Run the following Jupyter Notebooks:
  - ```1_profile_extraction.ipynb```
  - ```2_feature_extraction.ipynb```
  - There is a known error which has occured on a test device:
    ```
    TransientError: {code: Neo.TransientError.General.MemoryPoolOutOfMemoryError} {message: The allocation of an extra 2.0 MiB would use more than the limit 688.8 MiB. Currently using 687.3 MiB. dbms.memory.transaction.total.max threshold reached}
    ```
    It could be resolved by deleting resetting the neo4j instance. This means deleting the docker image as well as the folder called "data" which is created in the working directory when running the ```docker-compose up``` command for the first time.
2. These notebooks perform the following tasks:
  - Profile Extraction: An LLM generates a profile summary for each person based on their connected nodes and saves it as a property of the person node.
  - Feature Extraction: Skills and project involvement are extracted and stored as new nodes (```:Skill``` and ```:Project```) connected to the person nodes via ```HAS_SKILL``` and ```WORKS_ON``` relationships.

---

## Test the Prototype
1. **Launch the Streamlit Application**:
   - Run the following command:
   ```streamlit run main.py```
   - Access the application at: ```http://localhost:8501```.
   - Enable ```Debug Mode``` in the UI to observe internal processes.
3. **Interact with the Prototype**:
   Use the search bar to explore the fictional company.

   -**Important note**: If you want to ask a follow-up question to your current query you can simply enter a new question. If you want to ask a new question on the other hand, you have to click the ```Reset``` button first.

   Example queries:
   - Who can help me with budgeting?
   - Who can help me onboard a new customer?
   - Who does George Harris report to and which team does he belong to?
   - Who has experience with planning events?

   Example follow-up questions:
   - Who is his/her boss?
   - What projects does Tom work on?
   
5. **Stop the Application**:
   - Close the terminal or press ```Ctrl+C``` to stop the Streamlit service.
6. Stop the Neo4j Container:
   - Run the following command:
     ```docker-compose down```
   - A data folder will be created in your project directory. This ensures your graph data persists. Restart Neo4j later with:
     ```docker-compose up```

---

## References

Prototype0 has taken inspiration from three GitHub repositories.

- Docker setup of neo4j: [GraphRAG-with-Llama-3.1](https://github.com/Coding-Crashkurse/GraphRAG-with-Llama-3.1)
- Conceptual idea of using a dummy company and the basis of the streamlit web app: [Knowledge-Graph-Demo](https://github.com/JohannesJolkkonen/funktio-ai-samples/tree/main/knowledge-graph-demo)
- Conceptual idea of using the hybrid search in Neo4j and rephrasing follow-up questions: [Intro-to-GraphRAG](https://github.com/ms-johnalex/intro-to-graphrag/tree/main)
