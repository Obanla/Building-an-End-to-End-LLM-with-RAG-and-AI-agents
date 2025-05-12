# Building-an-End-to-End-LLM-with-RAG-and-AI-agents

## Table of Contents
* Background
* Key Components of the Project
* Discussion and Future Work

 ## Background
 This project focuses on developing a comprehensive end-to-end solution using Large Language Models (LLMs) by integrating Retrieval-Augmented Generation (RAG), web search capabilities via Tavily, 
 and collaborative AI agents through CrewAI. The primary objective is to mitigate LLM hallucinations by employing RAG, which facilitates real-time information retrieval from external sources to ensure the delivery of 
 accurate and current responses. Furthermore, the project advances its scope by implementing a multi-agent system where AI agents collaborate to research user queries, draft detailed responses, critique the provided answers, and compile a final report.

 ## Key Components of the Project
 
## 1. Environment Setup
* **Programming Languages:** Python 3.9
* **OpenAI API:** Used for integrating OpenAI's AI models like GPT-4 for natural language processing and data analysis tasks.
* **Groq API:** Powers fast LLM inference with low latency for tasks like text generation in the project using the "mistral-saba-24b" model.
* **Tavily API:** Provides real-time search results for AI agents to fetch accurate, current data on AI in healthcare via TavilySearchResults tool.

Ensure you are keeping your APi keys safe google colab offers colab secrets if trying to run on colab as seen in the RAG_LLM_Analysis.ipynb

* **langchain** – Framework for LLM apps
* **tavily-python** – Real-time search API
* **groq** – For inferencing LLM models running on groq cloud, so you don't have to run the LLM models on your local system
* **streamlit** – A Python library for creating interactive web applications, used to build the user interface of the AI in Healthcare Summarization App for inputting queries and displaying results.
* **langgraph** – A component of LangChain for building complex workflows and agentic systems, facilitating the orchestration of multiple AI agents in the project.
* **markdown_pdf** – A tool for converting Markdown content into PDF format, utilized to generate downloadable reports from the summarized content.
* **dotenv** – A Python library for loading environment variables from a .env file, ensuring secure management of API keys and configuration settings.
* **FAISS Retriever (custom tool)** – A custom retrieval tool integrated into the project, designed to fetch relevant documents from a local knowledge base on AI in healthcare for enhancing response accuracy through RAG.
* **OpenAI API:** Used for integrating OpenAI's AI models like GPT-4 for natural language processing and data analysis tasks.
* **Groq API:** Powers fast LLM inference with low latency for tasks like text generation in the project using the "mistral-saba-24b" model.
* **Tavily API:** Provides real-time search results for AI agents to fetch accurate, current data on AI in healthcare via TavilySearchResults tool.

## 2. Data Loading and Preprocessing
Documents are loaded from the crew_data folder. Pip Install PyPDF to be able to load the PDF files. Once loaded the PDF files are split into chunks using the RecursiveCharacterTextSplitter for efficient indexing. I chose a chunk size of 500 and an overlap of 50. Set this as per your requirements.
Once split and chunked, create an embeddings vector store and ensure you save it locally, this way you dont need to keep making API calls. You can create either a FAISS vector store or a Chroma DB vector store of the embeddings, I used FAISS in this project. 

## 3. Retrieval-Augmented QA
The FAISS vectorstore is used to retrieve documents relevant to a user’s query. The query and context are passed to the LLM to generate a meaningful response. After this is done, similarity_search() is used to 
see what documents match a query. Embeddings are visualized using PCA and each chunk is converted into an embedding vector using OpenAI Embeddings.

The embeddings are indexed with FAISS, a high-performance similarity search library. After which new documents are added to crew_data and the FAISS FAISS index is rebuilt and new queries are tested.

Also, I conducted a similarity search on the documents to unveil what document is similar to a query and then proceeded to use elbow method to find optimal K for document clusters while using PCA to visualize the document clusters in 2D as is seen in the plots below

![image](https://github.com/user-attachments/assets/6b764f6a-4ef5-4250-8586-c200cde3c975)

![image](https://github.com/user-attachments/assets/12bb26ae-9b9e-46d7-be3d-93a838cdf693)


## 4. 🌐 Tavily Search Integration
Tavily lets you add real-time web search capability to your AI agents and get real time information. The trend agent was created using LangChain's create_react_agent function with ChatGroq with "mistral-saba-24b" model
and tools like the FAISS Retriever and TavilySearchResults. It was then wrapped in an AgentExecutor to manage its execution and interactions,
with a custom prompt pulled from "hwchase17/react" to guide its research tasks on AI in healthcare trends Using Tavily tool. It's goal was to summarize the latest news on AI in healthcare

## 5. 🤖 CrewAI Agent Creation
In this project, CrewAI was used to orchestrate multiple specialized AI agents working collaboratively. Each agent is defined with a clear role, a specific goal, and a custom toolset. 
Here's a breakdown of the agents implemented in the notebook:

**1. Researcher Agent**
**Role:** Researcher
**Goal:** Conduct deep research on a given query by retrieving information from both the local vectorstore and live web sources.
**Tools:** FAISS retriever (for custom document knowledge base) and Tavily search (for real-time web data)

**2. Writer Agent**
**Role:** Content Writer
**Goal:** Generate detailed and structured responses or reports based on the information retrieved by the Researcher Agent.
**Tools:** LLM for generation, with context provided by the Researcher.

**3. Critic Agent**
**Role:** Reviewer
**Goal:** Evaluate the response generated by the Writer Agent, check for factual accuracy, coherence, and completeness, and suggest improvements.
**Tools:** LLM for analysis and refinement tasks.

Agents are created using LangChain's create_react_agent function from langgraph, implementing ReAct prompting with a specified LLM like ChatGroq ("mistral-saba-24b"), 
tools (TavilySearchResults, FAISS Retriever), and a prompt template from "hwchase17/react". Each agent (researcher, writer, critic, report generator) is customized for its role via specific tools and prompts. 
Once created, agents are managed by AgentExecutor 
for task execution and interaction, with real-time feedback via StreamlitCallbackHandler in the Streamlit app.

## Discussion and Future Work
This project will suit diverse business needs ranging from:

**1. Healthcare:** This project directly targets AI in healthcare summarization, offering improved decision-making for clinicians by retrieving patient-specific data, medical literature, and treatment guidelines, thus enhancing patient care and operational efficiency. It also supports pharmaceutical companies by accelerating drug development through instant access to clinical trial results and compliance reporting.

**Customer Support:** By integrating RAG and multi-agent systems, this project can be tailored and tweaked to assist businesses enhance AI-driven customer service with accurate, personalized responses to queries, reducing resolution times and increasing customer satisfaction across industries like retail and technology. This leads to higher loyalty and reduced operational costs.

**Financial Services:** The project’s ability to analyze data in real-time and generate detailed reports aids in fraud detection, risk assessment, and regulatory compliance, enabling banks and financial institutions to make informed decisions quickly and streamline processes. This improves security and operational efficiency.

**Manufacturing and Industrial Operations:** Multi-agent RAG systems can monitor equipment health, predict failures, and automate maintenance reports, minimizing downtime and boosting productivity for manufacturers through efficient data retrieval and analysis. This reduces costs associated with unplanned disruptions.

**E-Commerce:** The project supports personalized shopping experiences by pulling relevant product details and customer preferences, allowing platforms to tailor recommendations and content, which enhances user engagement and drives sales. This scalability adapts to dynamic market demands effectively.

**Future Work will include:**
1. Incoporating a chat function that provides continous answers to queries incase the user wants to continue from the last query while keeping answers short
2. Allowing direct PDF and document upload on the frontend




 
