from flask import Flask, request, jsonify
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    get_response_synthesizer
)
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from qdrant_client import QdrantClient
import logging
import time
import httpx
import json  # 导入json模块

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask application
app = Flask(__name__)

# Load the local data directory and chunk the data for further processing
docs = SimpleDirectoryReader(input_dir="./data", required_exts=[".pdf"]).load_data(show_progress=True)
text_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

text_chunks = []
doc_ids = []
nodes = []

# Create a local Qdrant vector store
logger.info("Initializing the vector store related objects")
client = QdrantClient(host="localhost", port=6333, timeout=600.0)
vector_store = QdrantVectorStore(client=client, collection_name="research_papers")

# Local vector embeddings model
logger.info("Initializing the OllamaEmbedding")
embed_model = OllamaEmbedding(model_name='shaw/dmeta-embedding-zh', base_url="http://localhost:11434")

logger.info("Initializing the global settings")
Settings.embed_model = embed_model
Settings.llm = Ollama(model="llama2-chinese:7b-chat-q6_K", base_url="http://localhost:11434")
Settings.transformations = [text_parser]

# Enumerating docs
logger.info("Enumerating docs")
for doc_idx, doc in enumerate(docs):
    curr_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(curr_text_chunks)
    doc_ids.extend([doc_idx] * len(curr_text_chunks))

# Enumerating text_chunks
logger.info("Enumerating text_chunks")
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(text=text_chunk)
    src_doc = docs[doc_ids[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)

# Enumerating nodes and processing in batches
logger.info("Enumerating nodes and processing in batches")
batch_size = 100  # 每批次处理的节点数量
for i in range(0, len(nodes), batch_size):
    batch_nodes = nodes[i:i + batch_size]
    for node in batch_nodes:
        content = node.get_content(metadata_mode=MetadataMode.ALL)
        print("Node content:", content)
        
        # Ensure get_text_embedding_batch is correctly called and process its returned embeddings
        try:
            node_embedding = embed_model.get_text_embedding_batch([content])
            print("Generated embedding for content:", node_embedding)
            
            if node_embedding and isinstance(node_embedding, list) and len(node_embedding) > 0:
                node.embedding = node_embedding[0]
                print("Embedding stored for node:", node.embedding)
            else:
                logger.error(f"Failed to get embeddings for node: {content}")
        except Exception as e:
            logger.error(f"Error generating embedding for node: {content}, error: {str(e)}")
    
    # 等待一段时间，以减轻Qdrant服务器的负载
    time.sleep(5)

# Initializing the storage context
logger.info("Initializing the storage context")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Indexing the nodes in VectorStoreIndex
logger.info("Indexing the nodes in VectorStoreIndex")
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    transformations=Settings.transformations,
)

# Initializing the VectorIndexRetriever with top_k as 5
logger.info("Initializing the VectorIndexRetriever with top_k as 5")
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
response_synthesizer = get_response_synthesizer()

# Creating the RetrieverQueryEngine instance
logger.info("Creating the RetrieverQueryEngine instance")
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
)

# Creating the HyDEQueryTransform instance
logger.info("Creating the HyDEQueryTransform instance")
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(vector_query_engine, hyde)

# Function to handle query with retry
def query_with_retry(query_engine, query_string, retries=3, delay=5):
    for attempt in range(retries):
        try:
            logger.info("Executing query, attempt %d", attempt + 1)
            response = query_engine.query(query_string)
            return response
        except (httpx.ReadTimeout, httpx.HTTPStatusError) as e:
            logger.error(f"Error during query execution: {str(e)}")
            if attempt < retries - 1:
                logger.info("Retrying in %d seconds...", delay)
                time.sleep(delay)
    raise Exception("Maximum retry attempts reached")

# Helper function to convert response to JSON serializable format
def convert_response_to_serializable(response):
    if isinstance(response, dict):
        return {k: convert_response_to_serializable(v) for k, v in response.items()}
    elif isinstance(response, list):
        return [convert_response_to_serializable(item) for item in response]
    elif hasattr(response, 'to_dict'):
        return response.to_dict()
    else:
        return str(response)

# Flask route to handle queries
@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        query_string = data.get('query', '')
        if not query_string:
            return jsonify({"error": "Query string is required"}), 400
        
        response = query_with_retry(hyde_query_engine, query_string)
        print(response)
        serializable_response = convert_response_to_serializable(response)
        return app.response_class(response=json.dumps({"response": serializable_response}, ensure_ascii=False), mimetype='application/json; charset=utf-8')
    except Exception as e:
        logger.error(f"Error during query execution: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
