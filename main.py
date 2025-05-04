import sys
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.agents import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_tavily import TavilySearch
from markdown_pdf import MarkdownPdf, Section
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt
import numpy as np





# Loading the environment variables
if not load_dotenv():
    raise ValueError("Failed to load .env file. Ensure it exists in the project root.")
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables.")


#defining the LLM
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables.")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found in environment variables.")


#setting up the directories
data_dir = 'crew_data'
new_docs_dir = os.path.join(data_dir, "new_docs")
faiss_index_path = "faiss_index"


#Loading and splitting original documents
documents = []
for filename in os.listdir(data_dir):
    if filename.endswith('.pdf'):
        loader = PyPDFLoader(os.path.join(data_dir, filename))
        documents.extend(loader.load())
if not documents:
    raise ValueError(f"No PDF files found in '{data_dir}'.")

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Create embeddings and store in a vector database uncomment if it's the first time running
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# vector_store = FAISS.from_documents(texts, embeddings)
#
# vector_store.save_local("faiss_index")
# print("FAISS vector store saved locally.")

#Loading the created FAISS vector store since it has been created and saved already
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
print(f"Successfully loaded and processed {len(documents)} documents.")



query = "What is the role of AI in healthcare today?"
docs = vector_store.similarity_search(query)
print("\nSample query results:")
for doc in docs:
    print(f"Task 1 Completed, these are the retrieved Documents: {doc.page_content[:200]}... (Source: {doc.metadata.get('source', 'Unknown')})")


#
#Expanding the knowledge base
all_new_chunks = []
if os.path.exists(new_docs_dir):
    for filename in os.listdir(new_docs_dir):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(new_docs_dir, filename))
            new_docs = loader.load()
            new_chunks = text_splitter.split_documents(new_docs)
            all_new_chunks.extend(new_chunks)

    if all_new_chunks:
        vector_store.add_documents(all_new_chunks)
        vector_store.save_local("faiss_index")
        print(f"Task 2 completed, I have Added {len(all_new_chunks)} new chunks from {len(os.listdir(new_docs_dir))} documents.")
    else:
        print("No new documents found to add.")
#

#
#
#Setting up RetrievalQA chain
llm = ChatGroq(model_name="mistral-saba-24b", temperature=0, max_tokens=10000)
# Setting up retriever from vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# Creating the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Test QA chain with the same query
response = qa_chain.invoke(query)
print(f"Query: {query}")
print(f"Response: {response}")


#TASK 1(a)
# Test similarity search
# query = "Why is making primary care safe difficult?"
# docs = vector_store.similarity_search(query)
# for doc in docs:
#     print(f"Task 1 Retrieved Document: {doc.page_content[:200]}... (Source: {doc.metadata.get('source', 'Unknown')})")
# #
# #TASK 1(b)
# # Extract embeddings from the vector store
embedding_vectors = np.array([vector_store.index.reconstruct_n(i, 1)[0] for i in range(vector_store.index.ntotal)])
# Reduce dimensions to 2D using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embedding_vectors)

#Determining optimal k using elbow method
# wcss = []  # With cluster Sum of squared errors
# k_range = range(1, 11)  # Test k from 1 to 10
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
#     kmeans.fit(embedding_vectors)  # Fit on original embeddings, not reduced
#     wcss.append(kmeans.inertia_)  # inertia_ is the SSE for this k

#  Plotting the Elbow Curve
# plt.figure(figsize=(8, 6))
# plt.plot(k_range, wcss, marker='o')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Sum of Squared Distances (SSE)')
# plt.title('Elbow Method for Optimal k')
# plt.xticks(k_range)
# plt.grid(True)
# plt.show()

optimal_k = 6
#Cluster with the chosen k and visualize
kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init='auto')
labels = kmeans.fit_predict(embedding_vectors)
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.6)
# plt.title(f'PCA Visualization of Document Embeddings with {optimal_k} Clusters')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.colorbar(scatter, label='Cluster')
# plt.show()

#plot to determine optimum k using elbow method
# # Plot
# plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
# plt.title("PCA Visualization of Document Embeddings")
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.show()


# Trying to determine what cluster each chunk belongs to
# print("\nDocuments/Chunks per Cluster:")
# for cluster_id in range(optimal_k):
#     print(f"\nCluster {cluster_id}:")
#     cluster_indices = np.where(labels == cluster_id)[0]  # Getting indices of embeddings in this cluster
#     for idx in cluster_indices[:6]:
#         if idx < len(texts):
#             chunk = texts[idx]
#             source = chunk.metadata.get('source', 'Unknown')
#             content_snippet = chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
#             print(f"  - Chunk {idx}: Source: {source}, Content: {content_snippet}")
#         else:
#             print(f"  - Chunk {idx}: (Index out of range for original chunk list)")
#     if len(cluster_indices) > 5:
#         print(f"  ... and {len(cluster_indices) - 5} more chunks in Cluster {cluster_id}")

# TASK 3: Create a New AI Agent ("Trend") with Tavily Tool

print("\n=== Starting Task 3: Creating 'Trend' AI Agent for AI in Healthcare News ===\n")

tavily_tool = TavilySearch(
    max_results=2,
    topic="news",  # Focus on news for latest updates
    include_answer=True,
    include_raw_content=False
)

custom_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
    You are 'Trend', an expert AI agent specializing in summarizing the latest news and trends on AI in Healthcare.
    Use the provided tools to fetch up-to-date information from the web.
    Synthesize the information into a concise summary of 3-5 sentences, highlighting key trends and technologies.
    Avoid listing raw data; focus on a cohesive narrative.
    Query: {input}
    """
)

# Pulling a prebuilt ReAct prompt from the hub
react_prompt = hub.pull("hwchase17/react")

# Initializing the Trend Agent with the prebuilt prompt
trend_agent = create_react_agent(
    llm=llm,
    tools=[tavily_tool],
    prompt=react_prompt  # Prebuilt prompt with required variables
)

# Creating an executor to run the agent
trend_agent_executor = AgentExecutor(
    agent=trend_agent,
    tools=[tavily_tool],
    verbose=True
)

# Run the query
trend_query = "Summarize the latest news on AI in Healthcare."
trend_result = trend_agent_executor.invoke({"input": trend_query})
print(f"This is Trend Agent Response to Task 3: {trend_result['output']}")


retriever_tool = Tool.from_function(
    func=retriever.invoke,
    name="FAISS Retriever",
    description="Retrieve relevant documents from a local knowledge base on AI in healthcare."
)

# Researcher Agent Setup
researcher_prompt = hub.pull("hwchase17/react")
researcher_agent = create_react_agent(
    llm=llm,
    tools=[retriever_tool, tavily_tool],
    prompt=researcher_prompt
)
researcher_agent_executor = AgentExecutor(agent=researcher_agent, tools=[retriever_tool, tavily_tool], verbose=True)
research_query = "Summarize the latest news on AI in Healthcare, including detailed information on specific initiatives, events, and technologies mentioned in recent reports."
research_result = researcher_agent_executor.invoke({"input": research_query})
print(f"This is what the research agent has researched: {research_result['output']}")

# Writer Agent Setup
writer_prompt = hub.pull("hwchase17/react")
writer_agent = create_react_agent(
    llm=llm,
    tools=[],
    prompt=writer_prompt
)
writer_agent_executor = AgentExecutor(agent=writer_agent, tools=[], verbose=True, handle_parsing_errors=True)
write_result = writer_agent_executor.invoke(
    {"input": f"Generate a detailed summary based on this data: {research_result['output']}"})
print(f"This is what the writer agent has written regarding the summary: {write_result['output']}")

# Critic Agent Setup
critic_prompt = hub.pull("hwchase17/react")
critic_agent = create_react_agent(
    llm=llm,
    tools=[tavily_tool],
    prompt=critic_prompt
)
critic_agent_executor = AgentExecutor(
    agent=critic_agent,
    tools=[tavily_tool],
    verbose=True,
    handle_parsing_errors=True
)
critique_result = critic_agent_executor.invoke({
                                                   "input": f"Review this summary for accuracy and completeness, suggesting specific improvements: {write_result['output']}"})
critique_text = critique_result.get('output', '')
print("Critique Suggestions:", critique_text)

# Feedback Loop for Revisions
max_revision_attempts = 2  # Limit the number of revision cycles
revision_count = 0
current_summary = write_result['output']
current_critique = critique_text

while revision_count < max_revision_attempts:
    if any(keyword in current_critique.lower() for keyword in
           ["improve", "add", "more detail", "specific", "lacking", "missing"]):
        print(
            f"Revision {revision_count + 1}/{max_revision_attempts}: Critic feedback suggests improvements. Sending back to writer agent.")
        revision_query = f"Revise this summary based on the following critique: Summary: {current_summary} | Critique: {current_critique}"
        revised_write_result = writer_agent_executor.invoke({"input": revision_query})
        current_summary = revised_write_result['output']
        print(f"Revised Summary (Iteration {revision_count + 1}): {current_summary}")

        critique_result = critic_agent_executor.invoke({
                                                           "input": f"Review this revised summary for accuracy and completeness, suggesting specific improvements: {current_summary}"})
        current_critique = critique_result.get('output', '')
        print(f"Updated Critique (Iteration {revision_count + 1}): {current_critique}")

        revision_count += 1
    else:
        print("Critic feedback accepted. No further revisions needed.")
        break

if revision_count >= max_revision_attempts:
    print(f"Reached maximum revision attempts ({max_revision_attempts}). Proceeding with the latest summary.")

final_revised_summary = current_summary
print("Final Revised Summary after Feedback Loop:", final_revised_summary)


# Report Generation Tool
def generate_report_tool(content: str, output_md_path: str = "report2.md", output_pdf_path: str = "report2.pdf") -> str:
    original_summary = ""
    critique = ""
    revised_summary = ""

    if "Original Summary:" in content and "Critique:" in content and "Revised Summary:" in content:
        parts = content.split("Original Summary:")[1].split("Critique:")
        original_summary = parts[0].strip()
        remaining = parts[1].split("Revised Summary:")
        critique = remaining[0].strip()
        revised_summary = remaining[1].strip()
    else:
        revised_summary = content  # Fallback if parsing fails

    markdown_content = f"""
# AI in Healthcare Report

## Summary of Latest Trends and Insights (Original)
{original_summary}

## Critique and Suggested Improvements
{critique}

## Summary of Latest Trends and Insights (Revised)
{revised_summary}

## Conclusion
This report summarizes the latest developments in AI for healthcare based on recent data and news.
    """
    with open(output_md_path, 'w', encoding='utf-8') as md_file:
        md_file.write(markdown_content)

    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(markdown_content))
    pdf.meta['title'] = 'AI in Healthcare Report'
    pdf.meta['author'] = 'AI Agent Report Generator Created by Obanla Oluwaseun'
    pdf.save(output_pdf_path)

    return (f"The Report has been generated and saved as {output_md_path} (Markdown) and "
            f"{output_pdf_path} (PDF)")


# Report Tool and Agent Setup
report_tool = Tool.from_function(
    func=generate_report_tool,
    name="ReportGeneratorTool",
    description="Generate a structured report in Markdown and PDF format from provided content."
)
report_prompt = hub.pull("hwchase17/react")
report_agent = create_react_agent(
    llm=llm,
    tools=[report_tool],
    prompt=report_prompt
)
report_agent_executor = AgentExecutor(
    agent=report_agent,
    tools=[report_tool],
    verbose=True
)

# Invoke Report Generator with All Components
report_query = f"Generate a structured report with these components: Original Summary: {write_result['output']} | Critique: {critique_text} | Revised Summary: {final_revised_summary}"
report_result = report_agent_executor.invoke({"input": report_query})
print(f"Report Generation Result: {report_result['output']}")
