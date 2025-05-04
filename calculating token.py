import tiktoken
from langchain_community.document_loaders import DirectoryLoader

# Load documents
data_dir = 'crew_data'
loader = DirectoryLoader(data_dir, glob="**/*.pdf")
documents = loader.load()

# Count tokens using tiktoken
encoder = tiktoken.get_encoding("cl100k_base")  # Used by most OpenAI models
total_tokens = 0
for doc in documents:
    tokens = len(encoder.encode(doc.page_content))
    total_tokens += tokens
    print(f"Document {doc.metadata.get('source', 'Unknown')}: {tokens} tokens")
print(f"Total tokens across all documents: {total_tokens}")