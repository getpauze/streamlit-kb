# ===========================================================
# 🧠 GenAI Knowledgebase (Strands + Bedrock + ChromaDB)
# ===========================================================
# 📚 Classroom Objective:
# This application demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline:
#  1) Upload & process documents (PDF, Markdown, TXT)
#  2) Split text into chunks
#  3) Create vector embeddings (Amazon Titan via AWS Bedrock, using boto3)
#  4) Store vectors in ChromaDB (local persistent vector database)
#  5) Retrieve the most relevant chunks using vector similarity
#  6) Answer questions with an LLM (via Strands Agent + Bedrock), grounded in retrieved chunks
#
# 🔧 Tech choices:
#  - No LangChain: embedding and vector store are called directly
#  - Strands Agents: small agent framework to define LLM + tool(s)
#  - ChromaDB: local vector database with persistent storage
#  - Streamlit: simple web UI for uploads, indexing, debugging, and Q&A
#
# -----------------------------------------------------------
# REQUIREMENTS (install once):
#   pip install streamlit python-dotenv PyPDF2 boto3 chromadb strands-agents
#
# -----------------------------------------------------------
# AWS ENVIRONMENT:
# You must have AWS credentials that allow Bedrock model invocation.
# Minimum permission: bedrock:InvokeModel
#
# Optional .env file in the same directory:
#   AWS_REGION=us-west-2
#   EMBED_MODEL_ID=amazon.titan-embed-text-v1
#   LLM_MODEL_ID=us.anthropic.claude-3-5-haiku-20241022-v1:0
#
# -----------------------------------------------------------
# HOW TO RUN:
#   streamlit run app.py
#
# -----------------------------------------------------------
# TEACHING TIPS:
# - Start by uploading a small .txt or .md file (or a short PDF).
# - Click "Re-index Knowledgebase" so embeddings are created & stored.
# - Use "Test Retrieval" to verify that chunks are being found.
# - Then ask a question in "Ask via LLM" and inspect the cited chunks.
# - Emphasize that tools must not call Streamlit APIs (they can run off-thread).
# ===========================================================

import os
import json
import time
import tempfile
import threading
import csv
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
import pandas as pd

import boto3
from botocore.config import Config

import chromadb
from chromadb.config import Settings

# Strands Agents — a tiny agent framework for LLM + tools
from strands import Agent, tool
from strands.models import BedrockModel


# ===========================================================
# 🌎 Environment & App Configuration
# ===========================================================
st.set_page_config(page_title="🧠 Knowledgebase (Strands + Bedrock)", layout="wide")

# Load variables from .env if present (useful for AWS_REGION, etc.)
load_dotenv()

# Region & model IDs: you can override via .env or edit defaults here
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v1")  # Titan embeddings
DEFAULT_LLM_MODEL_ID = os.getenv(
    "LLM_MODEL_ID",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0"  # Bedrock Anthropic model (Claude Haiku as example)
)
DEFAULT_TEMPERATURE = 0.2

# Folder to store uploads & converted text
DATA_DIR = "data"
# Persistent ChromaDB storage in data folder
PERSIST_ROOT = os.path.join(DATA_DIR, "chroma_db")

# Ensure the data directory exists so uploads don’t fail
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------
# GLOBALS (module-level, because tools must not call streamlit)
# -----------------------------------------------------------
CHROMA_CLIENT = None       # Chroma persistent client (reopened on rerun)
COLLECTION = None          # Chroma collection handle (reopened on rerun)
COLLECTION_NAME = "kb_main"  # Must be 3–512 chars; allowed [a-zA-Z0-9._-]

# Retrieval settings (default top-K)
K_RETRIEVE = 3

# Thread-safe buffer to store last retrieved sources so the UI can render them
_LAST_SOURCES: List[Dict[str, Any]] = []
_LAST_SOURCES_LOCK = threading.Lock()


def _set_last_sources(items: List[Dict[str, Any]]) -> None:
    """
    Store last retrieved chunks in a thread-safe way.
    Tools should not call Streamlit APIs, so we use this buffer to pass data back to the UI.
    """
    global _LAST_SOURCES
    with _LAST_SOURCES_LOCK:
        _LAST_SOURCES = items


def _get_last_sources() -> List[Dict[str, Any]]:
    """Access the last retrieved chunks (read-only copy)."""
    with _LAST_SOURCES_LOCK:
        return list(_LAST_SOURCES)


# ===========================================================
# 🔌 AWS Bedrock — Client & Embedding Function
# ===========================================================
def bedrock_client():
    """
    Bedrock Runtime client — used for invoking embeddings or text models.
    The account/role you run under must have bedrock:InvokeModel permissions.
    """
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        config=Config(retries={"max_attempts": 5, "mode": "standard"})
    )


def titan_embed(text: str) -> List[float]:
    """
    Create a vector embedding using Amazon Titan (Text Embeddings) via Bedrock.
    - Embeddings are numeric vectors that represent text in a high-dimensional space.
    - Similar texts → similar vectors → small distance in vector space.

    Returns:
        list[float]: The embedding vector.
    """
    if not text or not text.strip():
        return []

    # Titan v1 expects {"inputText": "..."} as the body
    body = {"inputText": text}

    resp = bedrock_client().invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())
    return payload.get("embedding", [])


# ===========================================================
# 📄 Text Utilities
# ===========================================================


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Naive character-based text splitter with overlap.
    Teaching notes:
    - Overlap preserves context continuity across chunk boundaries.
    - More sophisticated splitters (by sentence/paragraph) can improve retrieval precision.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


# ===========================================================
# 💾 ChromaDB Helpers (Native Client, Persistent)
# ===========================================================
def new_chroma_client(persist_dir: str):
    """
    Create a Chroma persistent client rooted at `persist_dir`.
    Teaching note: Chroma can run in-memory or persist to disk; we choose persistence so
    state survives Streamlit reruns.
    """
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def reset_collection(client, name: str):
    """
    Create a clean collection with the given name, deleting any existing one.
    Teaching note: In production, you might want versioning instead of delete/recreate.
    """
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        client.delete_collection(name)
    return client.create_collection(name=name)


def load_collection_from_persist() -> bool:
    """
    Load the persistent Chroma client/collection from the data folder.
    """
    global CHROMA_CLIENT, COLLECTION
    
    if not os.path.exists(PERSIST_ROOT):
        return False
        
    try:
        CHROMA_CLIENT = new_chroma_client(PERSIST_ROOT)
        COLLECTION = CHROMA_CLIENT.get_or_create_collection(COLLECTION_NAME)
        # Update session state for consistency
        st.session_state["persist_dir"] = PERSIST_ROOT
        st.session_state["vectorstore_loaded"] = True
        return True
    except Exception as e:
        st.warning(f"Failed to load from persistent folder: {e}")
        return False


def pregenerate_index() -> bool:
    """
    Pre-generate the ChromaDB index using all .txt files from the index_source folder.
    This function can be called to create the initial index.
    """
    global CHROMA_CLIENT, COLLECTION

    # Ensure the data directory exists
    os.makedirs(PERSIST_ROOT, exist_ok=True)

    # Define the index source directory
    index_source_dir = os.path.join(DATA_DIR, "index_source")
    
    if not os.path.exists(index_source_dir):
        print(f"Error: Index source directory not found: {index_source_dir}")
        return False

    # 1) Gather file contents from all .txt files in index_source folder
    docs, ids, metadatas = [], [], []
    txt_files = [f for f in os.listdir(index_source_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"Error: No .txt files found in {index_source_dir}")
        return False
    
    print(f"Found {len(txt_files)} .txt files to index:")
    for fname in txt_files:
        print(f"  - {fname}")
    
    for fname in txt_files:
        path = os.path.join(index_source_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

        if not content.strip():
            print(f"Warning: {fname} is empty, skipping...")
            continue

        # 2) Split into chunks
        chunks = split_text(content, chunk_size=500, overlap=50)
        for i, ch in enumerate(chunks):
            docs.append(ch)
            ids.append(f"{fname}-{i}")                # Unique ID per chunk
            metadatas.append({"source": fname, "chunk": i})  # Store source & chunk index

    if not docs:
        print("Error: No valid chunks produced from source files.")
        return False

    # 3) Create Chroma client and collection
    CHROMA_CLIENT = new_chroma_client(PERSIST_ROOT)
    COLLECTION = reset_collection(CHROMA_CLIENT, COLLECTION_NAME)

    # 4) Embed and add to Chroma
    print(f"Creating embeddings for {len(docs)} chunks...")
    embeddings = []
    
    for i, doc in enumerate(docs):
        embeddings.append(titan_embed(doc))
        if (i + 1) % 10 == 0:
            print(f"Embedded {i + 1}/{len(docs)} chunks")

    # Add to Chroma
    COLLECTION.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)

    print(f"✅ Pre-generated index with {len(docs)} chunks from {len(set(m['source'] for m in metadatas))} files.")
    print(f"Index stored in: {PERSIST_ROOT}")
    return True


# ===========================================================
# 🛠️ Retrieval Tool (Strands) — NO Streamlit calls inside!
# ===========================================================
@tool
def tool_retrieve_chunks(question: str, search_type: str = "general") -> str:
    """
    Comprehensive retrieval tool for searching the knowledge base index.
    
    Args:
        question: Natural language query to search for
        search_type: Type of search - "general" for raw chunks, "product" for formatted product info
    
    Returns:
        Relevant information from the knowledge base, formatted based on search_type
    """
    global COLLECTION, K_RETRIEVE
    if COLLECTION is None:
        return "[Error] No knowledge base loaded. Please ensure the index is available."

    # 1) Embed the question
    q_vec = titan_embed(question)

    # 2) Query Chroma for similar chunks
    res = COLLECTION.query(
        query_embeddings=[q_vec],
        n_results=max(1, int(K_RETRIEVE)),
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    if not docs:
        if search_type == "product":
            return f"[Info] No product information found for query: '{question}'"
        else:
            return "[No matching context]"

    # 3) Store for UI (thread-safe) - always store for UI display
    packed = []
    for d, m, dist in zip(docs, metas, dists):
        packed.append(
            {
                "text": d or "",
                "meta": m or {},
                "distance": float(dist) if dist is not None else None,
            }
        )
    _set_last_sources(packed)

    # 4) Format output based on search type
    if search_type == "product":
        # Product-focused formatting
        output_lines = []
        output_lines.append(f"🛍️ Product Search Results for: '{question}'")
        output_lines.append("=" * 60)
        
        for rank, item in enumerate(packed, start=1):
            src = item["meta"].get("source", "unknown")
            dist = item.get("distance")
            dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "N/A"
            
            output_lines.append(f"\n📄 Result {rank} — {src} (relevance: {dist_str})")
            output_lines.append("-" * 40)
            output_lines.append(item["text"])
        
        return "\n".join(output_lines)
    else:
        # General formatting for agent consumption
        out_lines = []
        for rank, item in enumerate(packed, start=1):
            src = item["meta"].get("source", "unknown")
            dist = item.get("distance")
            dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "NA"
            out_lines.append(f"[Source {rank} — {src} — dist:{dist_str}]\n{item['text']}")
        return "\n\n".join(out_lines)


# ===========================================================
# 🛠️ Order Management Tools
# ===========================================================

@tool
def tool_product_search(search_term: str = None, category: str = None) -> str:
    """
    Search for product information in the knowledge base.
    
    Args:
        search_term: Search term to look for in product descriptions
        category: Product category to filter by (optional)
    
    Returns:
        Product information from the knowledge base
    """
    global COLLECTION, K_RETRIEVE
    if COLLECTION is None:
        return "[Error] No knowledge base loaded. Please ensure the index is available."

    # Build search query
    if search_term and category:
        query = f"{search_term} {category}"
    elif search_term:
        query = search_term
    elif category:
        query = category
    else:
        return "[Error] Please provide either search_term or category"

    # Use the existing retrieval tool with product search type
    return tool_retrieve_chunks(query, search_type="product")


@tool
def tool_order_lookup(order_id: str) -> str:
    """
    Look up a specific order by its ID and return detailed information.
    
    Args:
        order_id: The order ID to look up (e.g., "ORD-001")
    
    Returns:
        Detailed order information including customer and product details
    """
    try:
        # Load orders data
        orders_path = os.path.join(DATA_DIR, "oms", "orders.csv")
        customers_path = os.path.join(DATA_DIR, "oms", "customers.csv")
        products_path = os.path.join(DATA_DIR, "oms", "products.csv")
        
        if not all(os.path.exists(p) for p in [orders_path, customers_path, products_path]):
            return "[Error] Required CSV files not found"
        
        orders_df = pd.read_csv(orders_path)
        customers_df = pd.read_csv(customers_path)
        products_df = pd.read_csv(products_path)
        
        # Find the order
        order = orders_df[orders_df['order_id'] == order_id]
        if order.empty:
            available_orders = orders_df['order_id'].tolist()[:5]  # Show first 5 orders
            return f"[Error] Order '{order_id}' not found. Available orders: {', '.join(available_orders)}"
        
        order_row = order.iloc[0]
        
        # Get customer details
        customer = customers_df[customers_df['customer_id'] == order_row['customer_id']]
        customer_info = customer.iloc[0] if not customer.empty else None
        
        # Get product details
        product = products_df[products_df['product_id'] == order_row['product_id']]
        product_info = product.iloc[0] if not product.empty else None
        
        # Build response
        output_lines = []
        output_lines.append(f"📦 Order Details: {order_id}")
        output_lines.append("=" * 50)
        output_lines.append(f"Order ID: {order_row['order_id']}")
        output_lines.append(f"Order Date: {order_row['order_date']}")
        output_lines.append(f"Status: {order_row['status']}")
        output_lines.append(f"Quantity: {order_row['quantity']}")
        output_lines.append(f"Unit Price: ${order_row['unit_price']}")
        output_lines.append(f"Total Amount: ${order_row['total_amount']}")
        output_lines.append(f"Shipping Address: {order_row['shipping_address']}")
        output_lines.append("")
        
        if customer_info is not None:
            output_lines.append("👤 Customer Information:")
            output_lines.append(f"  Name: {customer_info['name']}")
            output_lines.append(f"  Email: {customer_info['email']}")
            output_lines.append(f"  Phone: {customer_info['phone']}")
            output_lines.append(f"  Status: {customer_info['status']}")
            output_lines.append("")
        
        if product_info is not None:
            output_lines.append("🛍️ Product Information:")
            output_lines.append(f"  Name: {product_info['name']}")
            output_lines.append(f"  Category: {product_info['category']}")
            output_lines.append(f"  Description: {product_info['description']}")
            output_lines.append(f"  Stock: {product_info['stock_quantity']} units")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"[Error] Failed to look up order: {str(e)}"


@tool
def tool_customer_profile(customer_id: str = None, email: str = None) -> str:
    """
    Get customer profile and order history.
    
    Args:
        customer_id: Customer ID to look up
        email: Customer email to look up (alternative to customer_id)
    
    Returns:
        Customer profile and order history
    """
    try:
        orders_path = os.path.join(DATA_DIR, "oms", "orders.csv")
        customers_path = os.path.join(DATA_DIR, "oms", "customers.csv")
        products_path = os.path.join(DATA_DIR, "oms", "products.csv")
        
        if not all(os.path.exists(p) for p in [orders_path, customers_path, products_path]):
            return "[Error] Required CSV files not found"
        
        customers_df = pd.read_csv(customers_path)
        orders_df = pd.read_csv(orders_path)
        products_df = pd.read_csv(products_path)
        
        # Find customer
        if customer_id:
            customer = customers_df[customers_df['customer_id'] == customer_id]
        elif email:
            customer = customers_df[customers_df['email'] == email]
        else:
            return "[Error] Please provide either customer_id or email"
        
        if customer.empty:
            return f"[Error] Customer not found"
        
        customer_info = customer.iloc[0]
        
        # Get customer's orders
        customer_orders = orders_df[orders_df['customer_id'] == customer_info['customer_id']]
        
        # Build response
        output_lines = []
        output_lines.append(f"👤 Customer Profile: {customer_info['name']}")
        output_lines.append("=" * 50)
        output_lines.append(f"Customer ID: {customer_info['customer_id']}")
        output_lines.append(f"Name: {customer_info['name']}")
        output_lines.append(f"Email: {customer_info['email']}")
        output_lines.append(f"Phone: {customer_info['phone']}")
        output_lines.append(f"Registration Date: {customer_info['registration_date']}")
        output_lines.append(f"Status: {customer_info['status']}")
        output_lines.append(f"Total Orders: {customer_info['total_orders']}")
        output_lines.append("")
        
        if not customer_orders.empty:
            output_lines.append("📦 Order History:")
            total_spent = customer_orders['total_amount'].sum()
            output_lines.append(f"Total Spent: ${total_spent:.2f}")
            output_lines.append("")
            
            for _, order in customer_orders.iterrows():
                product = products_df[products_df['product_id'] == order['product_id']]
                product_name = product.iloc[0]['name'] if not product.empty else "Unknown Product"
                output_lines.append(f"  • {order['order_id']} - {product_name} (${order['total_amount']}) - {order['status']} - {order['order_date']}")
        else:
            output_lines.append("📦 No orders found for this customer")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"[Error] Failed to get customer profile: {str(e)}"


@tool
def tool_order_status_summary() -> str:
    """
    Get a summary of all order statuses and key metrics.
    
    Returns:
        Order status summary and metrics
    """
    try:
        orders_path = os.path.join(DATA_DIR, "oms", "orders.csv")
        
        if not os.path.exists(orders_path):
            return "[Error] Orders CSV file not found"
        
        orders_df = pd.read_csv(orders_path)
        
        # Ensure status column is treated as string
        orders_df['status'] = orders_df['status'].astype(str)
        
        # Calculate metrics
        total_orders = len(orders_df)
        total_revenue = orders_df['total_amount'].sum()
        avg_order_value = orders_df['total_amount'].mean()
        
        # Status breakdown
        status_counts = orders_df['status'].value_counts()
        
        # Recent orders (last 5)
        recent_orders = orders_df.sort_values('order_date', ascending=False).head(5)
        
        # Build response
        output_lines = []
        output_lines.append("📊 Order Status Summary")
        output_lines.append("=" * 40)
        output_lines.append(f"Total Orders: {total_orders}")
        output_lines.append(f"Total Revenue: ${total_revenue:.2f}")
        output_lines.append(f"Average Order Value: ${avg_order_value:.2f}")
        output_lines.append("")
        
        output_lines.append("📈 Order Status Breakdown:")
        for status, count in status_counts.items():
            percentage = (count / total_orders) * 100
            output_lines.append(f"  {status}: {count} orders ({percentage:.1f}%)")
        
        output_lines.append("")
        output_lines.append("🕒 Recent Orders:")
        for _, order in recent_orders.iterrows():
            output_lines.append(f"  {order['order_id']} - ${order['total_amount']} - {order['status']} - {order['order_date']}")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"[Error] Failed to get order status summary: {str(e)}"


@tool
def tool_inventory_check(product_id: str = None, category: str = None, low_stock_threshold: int = 10) -> str:
    """
    Check inventory levels for products.
    
    Args:
        product_id: Specific product ID to check
        category: Check all products in a category
        low_stock_threshold: Threshold for low stock warning (default: 10)
    
    Returns:
        Inventory status information
    """
    try:
        products_path = os.path.join(DATA_DIR, "oms", "products.csv")
        
        if not os.path.exists(products_path):
            return "[Error] Products CSV file not found"
        
        products_df = pd.read_csv(products_path)
        
        # Apply filters
        if product_id:
            filtered_df = products_df[products_df['product_id'] == product_id]
            if filtered_df.empty:
                return f"[Error] Product '{product_id}' not found"
        elif category:
            filtered_df = products_df[products_df['category'].str.contains(category, case=False, na=False)]
        else:
            filtered_df = products_df
        
        # Check stock levels
        low_stock = filtered_df[filtered_df['stock_quantity'] <= low_stock_threshold]
        out_of_stock = filtered_df[filtered_df['stock_quantity'] == 0]
        in_stock = filtered_df[filtered_df['stock_quantity'] > low_stock_threshold]
        
        # Build response
        output_lines = []
        output_lines.append("📦 Inventory Status")
        output_lines.append("=" * 30)
        
        if product_id:
            product = filtered_df.iloc[0]
            output_lines.append(f"Product: {product['name']}")
            output_lines.append(f"Current Stock: {product['stock_quantity']} units")
            if product['stock_quantity'] == 0:
                output_lines.append("⚠️ OUT OF STOCK")
            elif product['stock_quantity'] <= low_stock_threshold:
                output_lines.append("⚠️ LOW STOCK")
            else:
                output_lines.append("✅ In Stock")
        else:
            output_lines.append(f"Total Products: {len(filtered_df)}")
            output_lines.append(f"In Stock: {len(in_stock)}")
            output_lines.append(f"Low Stock: {len(low_stock)}")
            output_lines.append(f"Out of Stock: {len(out_of_stock)}")
            output_lines.append("")
            
            if not out_of_stock.empty:
                output_lines.append("🚫 Out of Stock:")
                for _, product in out_of_stock.iterrows():
                    output_lines.append(f"  • {product['product_id']} - {product['name']}")
                output_lines.append("")
            
            if not low_stock.empty:
                output_lines.append("⚠️ Low Stock:")
                for _, product in low_stock.iterrows():
                    output_lines.append(f"  • {product['product_id']} - {product['name']} ({product['stock_quantity']} units)")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"[Error] Failed to check inventory: {str(e)}"
        


# ===========================================================
# 🖥️ Streamlit UI
# ===========================================================
st.title("🧠 AIkea")

# Track whether we've created an index during this session
if "vectorstore_loaded" not in st.session_state:
    st.session_state["vectorstore_loaded"] = False

# Initialize conversation history
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

# Try to pregenerate index first, then load existing persistent index on startup
if not st.session_state.get("vectorstore_loaded", False):
    # First, try to pregenerate the index
    if pregenerate_index():
        st.session_state["vectorstore_loaded"] = True
        st.success("✅ Pre-generated ChromaDB index from index_source folder")
    elif load_collection_from_persist():
        st.session_state["vectorstore_loaded"] = True
        st.success("✅ Loaded pre-generated ChromaDB index from data folder")
    else:
        st.error("❌ No index found. Please ensure the index exists in the data folder.")

# Sidebar navigation
st.sidebar.title("🧭 Navigation")

# Streamlit reruns can drop globals — rehydrate Chroma from disk if needed
if COLLECTION is None:
    load_collection_from_persist()

# Quick health check (count chunks)
if COLLECTION is not None:
    try:
        st.info(f"📦 Collection: {COLLECTION_NAME} | 🔢 Chunks: {COLLECTION.count()}")
    except Exception as e:
        st.warning(f"Could not read collection count: {e}")

# Build the Bedrock-backed Strands model + agent (use defaults)
    bedrock_model = BedrockModel(model_id=DEFAULT_LLM_MODEL_ID, temperature=0.2, region=AWS_REGION)
    agent = Agent(model=bedrock_model, tools=[
        tool_retrieve_chunks, 
        tool_order_lookup, 
        tool_customer_profile, 
        tool_product_search, 
        tool_order_status_summary, 
        tool_inventory_check
    ])

# Display conversation history
if st.session_state["conversation_history"]:
    for i, message in enumerate(st.session_state["conversation_history"]):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
else:
    # Welcome message
    with st.chat_message("assistant"):
        st.write("Hello! I'm your AI assistant for order management and product information. How can I help you today?")
        st.write("You can ask me about:")
        st.write("• Products and inventory")
        st.write("• Order status and tracking")
        st.write("• Customer information")
        st.write("• Or anything else you need help with!")

# Chat input at the bottom
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to history
    st.session_state["conversation_history"].append({
        "role": "user",
        "content": prompt,
        "timestamp": time.time()
    })
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Build conversation context for the agent
            conversation_context = "Previous conversation:\n"
            for msg in st.session_state["conversation_history"][:-1]:  # Exclude the current user message
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
            
            conversation_context += f"\nCurrent user question: {prompt}"
            
            system_preamble = (
                "You are a helpful assistant for an order management system. You have access to these specialized tools: "
                "1) `tool_retrieve_chunks` - search through indexed documents for additional context "
                "2) `tool_order_lookup` - get detailed information about a specific order by ID "
                "3) `tool_customer_profile` - get customer profile and order history by customer_id or email "
                "4) `tool_product_search` - search products by category, price range, stock status, or search terms "
                "5) `tool_order_status_summary` - get overview of all orders, revenue, and status breakdown "
                "6) `tool_inventory_check` - check stock levels for products or categories "
                "Use the appropriate tool(s) based on the user's question, then provide a helpful answer. "
                "Maintain context from the conversation history and provide relevant follow-up suggestions. "
                "If you use any context, cite it inline as [Source 1], [Source 2], etc. "
                "If no context is relevant, say so."
            )
            
            response = agent(f"{system_preamble}\n\n{conversation_context}")
            
            # Get sources used in this response
            sources = _get_last_sources()
            
            # Add assistant response to history
            st.session_state["conversation_history"].append({
                "role": "assistant",
                "content": str(response),
                "sources": sources,
                "timestamp": time.time()
            })
            
            # Display the response
            st.write(str(response))
            
            # Show sources in a subtle way
            if sources:
                with st.expander("📄 Sources", expanded=False):
                    for j, source in enumerate(sources, 1):
                        src = source["meta"].get("source", "unknown")
                        chk = source["meta"].get("chunk", None)
                        dist = source.get("distance", None)
                        title = f"Source {j} — {src}" + (f" chunk {chk}" if chk is not None else "") + (f" (distance: {dist:.4f})" if isinstance(dist, (int, float)) else "")
                        with st.expander(title, expanded=False):
                            st.write(source["text"] or "")

# Simple controls at the bottom
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🗑️ Clear", help="Clear conversation"):
        st.session_state["conversation_history"] = []
        st.rerun()
with col2:
    if st.button("📋 Export", help="Download chat"):
        if st.session_state["conversation_history"]:
            chat_text = "Conversation Export\n" + "="*50 + "\n\n"
            for message in st.session_state["conversation_history"]:
                role = "User" if message["role"] == "user" else "Assistant"
                chat_text += f"{role}: {message['content']}\n\n"
            
            st.download_button(
                label="Download",
                data=chat_text,
                file_name=f"conversation_{int(time.time())}.txt",
                mime="text/plain"
            )
with col3:
    if st.button("🔄 New Topic", help="Start fresh"):
        st.session_state["conversation_history"] = []
        st.rerun()
