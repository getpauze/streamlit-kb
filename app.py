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
    txt_files = [f for f in os.listdir(index_source_dir)]
    
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
    return True


# ===========================================================
# 🛠️ Retrieval Tool (Strands) — NO Streamlit calls inside!
# ===========================================================
@tool
def tool_retrieve_chunks(question: str) -> str:
    """
    Comprehensive retrieval tool for searching the knowledge base index.
    
    Args:
        question: Natural language query to search for
    
    Returns:
        Relevant information from the knowledge base with product-focused formatting
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
        return f"[Info] No product information found for query: '{question}'"

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

    # 4) Product-focused formatting
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


# ===========================================================
# 🛠️ Order Management Tools
# ===========================================================

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


@tool
def tool_place_order(customer_id: str, product_id: str, quantity: int, shipping_address: str) -> str:
    """
    Place a new order for a customer.
    
    Args:
        customer_id: Customer ID placing the order
        product_id: Product ID to order
        quantity: Number of units to order
        shipping_address: Delivery address for the order
    
    Returns:
        Order confirmation with details or error message
    """
    try:
        # Load all required CSV files
        orders_path = os.path.join(DATA_DIR, "oms", "orders.csv")
        customers_path = os.path.join(DATA_DIR, "oms", "customers.csv")
        products_path = os.path.join(DATA_DIR, "oms", "products.csv")
        
        if not all(os.path.exists(p) for p in [orders_path, customers_path, products_path]):
            return "[Error] Required CSV files not found"
        
        # Load data
        orders_df = pd.read_csv(orders_path)
        customers_df = pd.read_csv(customers_path)
        products_df = pd.read_csv(products_path)
        
        # Validate customer exists
        customer = customers_df[customers_df['customer_id'] == customer_id]
        if customer.empty:
            available_customers = customers_df['customer_id'].tolist()[:5]
            return f"[Error] Customer '{customer_id}' not found. Available customers: {', '.join(available_customers)}"
        
        customer_info = customer.iloc[0]
        
        # Validate product exists
        product = products_df[products_df['product_id'] == product_id]
        if product.empty:
            available_products = products_df['product_id'].tolist()
            return f"[Error] Product '{product_id}' not found. Available products: {', '.join(available_products)}"
        
        product_info = product.iloc[0]
        
        # Check stock availability
        if product_info['stock_quantity'] < quantity:
            return f"[Error] Insufficient stock. Available: {product_info['stock_quantity']} units, Requested: {quantity} units"
        
        # Validate quantity
        if quantity <= 0:
            return "[Error] Quantity must be greater than 0"
        
        # Generate new order ID
        existing_order_ids = orders_df['order_id'].tolist()
        order_counter = len(existing_order_ids) + 1
        new_order_id = f"ORD-{order_counter:03d}"
        
        # Calculate pricing
        unit_price = product_info['price']
        total_amount = unit_price * quantity
        
        # Get current date
        from datetime import datetime
        order_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create new order record
        new_order = {
            'order_id': new_order_id,
            'customer_id': customer_id,
            'product_id': product_id,
            'quantity': quantity,
            'unit_price': unit_price,
            'total_amount': total_amount,
            'order_date': order_date,
            'status': 'processing',
            'shipping_address': shipping_address
        }
        
        # Add order to orders dataframe
        new_order_df = pd.DataFrame([new_order])
        updated_orders_df = pd.concat([orders_df, new_order_df], ignore_index=True)
        
        # Update inventory
        products_df.loc[products_df['product_id'] == product_id, 'stock_quantity'] -= quantity
        
        # Update customer total orders
        customers_df.loc[customers_df['customer_id'] == customer_id, 'total_orders'] += 1
        
        # Save updated data
        updated_orders_df.to_csv(orders_path, index=False)
        products_df.to_csv(products_path, index=False)
        customers_df.to_csv(customers_path, index=False)
        
        # Build confirmation response
        output_lines = []
        output_lines.append("✅ Order Successfully Placed!")
        output_lines.append("=" * 40)
        output_lines.append(f"Order ID: {new_order_id}")
        output_lines.append(f"Customer: {customer_info['name']} ({customer_id})")
        output_lines.append(f"Product: {product_info['name']}")
        output_lines.append(f"Quantity: {quantity} units")
        output_lines.append(f"Unit Price: ${unit_price:.2f}")
        output_lines.append(f"Total Amount: ${total_amount:.2f}")
        output_lines.append(f"Order Date: {order_date}")
        output_lines.append(f"Status: Processing")
        output_lines.append(f"Shipping Address: {shipping_address}")
        output_lines.append("")
        output_lines.append(f"📦 Updated Stock: {product_info['stock_quantity'] - quantity} units remaining")
        output_lines.append("")
        output_lines.append("📧 Order confirmation will be sent to the customer's email address.")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"[Error] Failed to place order: {str(e)}"
        



# ===========================================================
# 🖥️ Streamlit UI
# ===========================================================
# IKEA Color Scheme CSS
st.markdown("""
<style>
    /* IKEA Color Palette */
    :root {
        --ikea-blue: #0058A3;
        --ikea-yellow: #FFD700;
        --ikea-light-blue: #E8F4FD;
        --ikea-dark-blue: #003D82;
        --ikea-gray: #F5F5F5;
        --ikea-dark-gray: #333333;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main h1 {
        color: var(--ikea-blue);
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--ikea-light-blue);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: var(--ikea-blue);
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--ikea-blue);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--ikea-dark-blue);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 88, 163, 0.3);
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background-color: var(--ikea-yellow);
        color: var(--ikea-dark-blue);
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #FFC700;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 215, 0, 0.3);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid var(--ikea-blue);
    }
    
    /* Success boxes */
    .stAlert[data-testid="stAlert"]:has(.stMarkdown:contains("✅")) {
        background-color: #E8F5E8;
        border-left-color: #28A745;
    }
    
    /* Warning boxes */
    .stAlert[data-testid="stAlert"]:has(.stMarkdown:contains("⚠️")) {
        background-color: #FFF3CD;
        border-left-color: #FFC107;
    }
    
    /* Error boxes */
    .stAlert[data-testid="stAlert"]:has(.stMarkdown:contains("❌")) {
        background-color: #F8D7DA;
        border-left-color: #DC3545;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: var(--ikea-gray);
        border-radius: 12px;
        margin: 0.5rem 0;
        padding: 1rem;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="stChatMessageUser"]) {
        background-color: var(--ikea-light-blue);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--ikea-light-blue);
        color: var(--ikea-blue);
        font-weight: 600;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #E0E0E0;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--ikea-blue);
        box-shadow: 0 0 0 2px rgba(0, 88, 163, 0.2);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: var(--ikea-blue);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #E0E0E0;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--ikea-blue);
        box-shadow: 0 0 0 2px rgba(0, 88, 163, 0.2);
    }
    
    
    /* Chat container improvements */
    .stChatMessage {
        margin-bottom: 1rem;
    }
    
    /* Chat input styling */
    .stChatInput > div > div > textarea {
        border-radius: 12px;
        border: 1px solid #E0E0E0;
        padding: 12px 16px;
        font-size: 0.95rem;
    }
    
    .stChatInput > div > div > textarea:focus {
        border-color: var(--ikea-blue);
        box-shadow: 0 0 0 2px rgba(0, 88, 163, 0.1);
    }
    
    /* Reduce default Streamlit header spacing */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Hide default Streamlit header padding */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Remove top padding from main container */
    .main > div {
        padding-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Custom centered header with minimal spacing
st.markdown("""
<div style="text-align: center; margin: -2rem 0 0.5rem 0;">
    <h1 style="color: #333; font-size: 1.8rem; font-weight: 600; margin: 0; padding: 0;">
        🏠 IKEA AI Customer Support Agent
    </h1>
</div>
""", unsafe_allow_html=True)
# Track whether we've created an index during this session
if "vectorstore_loaded" not in st.session_state:
    st.session_state["vectorstore_loaded"] = False

# Initialize conversation history
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []


# -----------------------------------------------------------
# Live Chat Interface
# -----------------------------------------------------------
# Streamlit reruns can drop globals — rehydrate Chroma from disk if needed

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
    elif load_collection_from_persist():
        st.session_state["vectorstore_loaded"] = True
    else:
        st.error("❌ No index found. Please ensure the index exists in the data folder.")

if COLLECTION is None:
    load_collection_from_persist()

# Build the Bedrock-backed Strands model + agent (use defaults)
bedrock_model = BedrockModel(model_id=DEFAULT_LLM_MODEL_ID, temperature=0.2, region_name=AWS_REGION)
agent = Agent(model=bedrock_model, tools=[
    tool_retrieve_chunks, 
    tool_order_lookup, 
    tool_customer_profile, 
    tool_order_status_summary, 
    tool_inventory_check,
    tool_place_order
])

# Display conversation history
if st.session_state["conversation_history"]:
    for message in st.session_state["conversation_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
else:
    # Welcome message when no conversation history
    with st.chat_message("assistant"):
        st.write("Hello! I'm your IKEA AI Customer Support Agent. How can I help you today?")

# Chat input at the bottom
if prompt := st.chat_input("Ask me anything about IKEA products or your order..."):
    # Add user message to history and immediately display it
    st.session_state["conversation_history"].append({
        "role": "user",
        "content": prompt,
        "timestamp": time.time()
    })
    
    # Display the user's message immediately
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate assistant response
    try:
        # Build conversation context for the agent
        conversation_context = "Previous conversation:\n"
        for msg in st.session_state["conversation_history"][:-1]:  # Exclude the current user message
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['content']}\n"
        
        conversation_context += f"\nCurrent user question: {prompt}"
        
        system_prompt = """You are an IKEA AI Customer Support Agent. Your primary goal is to assist customers accurately and efficiently while strictly protecting their privacy and IKEA's internal data. 

        🔒 SECURITY & PRIVACY MANDATE: Your #1 Priority
        - NEVER divulge Personally Identifiable Information (PII) like full customer names, shipping addresses, phone numbers, or full email addresses unless you are verifying information the customer has *already provided*.
        - NEVER share internal business data or tools you have access to.
        - ALWAYS authenticate the user before accessing placing an order or accessing order details.
        - ALWAYS authenticate the user before accessing customer details.

        ---

        ### Authentication Workflow
        Before using any `ORDER MANAGEMENT TOOLS` for a specific customer, you MUST first authenticate them by asking one of the two pieces of information, such as:
        1. The Order ID OR
        2. The email address used to place the order.

        Only after the tool confirms a match should you proceed to answer their specific question.

        ---

        ### Available Tools

        You must use the appropriate tool to get real data before responding.

         🔍 KNOWLEDGE BASE TOOLS (Public Information):
         * `tool_retrieve_chunks(question)` - Search the knowledge base for information about IKEA products. Do not provide information that is not present in the result of this tool.

        🛍️ PRODUCT TOOLS (Public Information):
        * `tool_inventory_check(product_id, category, low_stock_threshold=10)` - Check public inventory levels for specific products or categories.

        📦 ORDER MANAGEMENT TOOLS (Requires Authentication):
        * `tool_order_lookup(order_id)` - Get details for a specific order. **Only answer the user's direct question.** Do not recite all the data from this tool. For example, if asked for shipping status, only provide the shipping status.
        * `tool_customer_profile(customer_id, email)` - Use this tool primarily to **verify** customer information, not to disclose it.
        * `tool_place_order(customer_id, product_id, quantity, shipping_address)` - Place a new order. Before executing, you must confirm all details (product, quantity, and address) with the customer one final time.

        ---

        ### Usage Guidelines & Principles

        * Principle of Least Privilege: When answering a question after authentication, provide **only the information requested**.
        * Bad Example: User asks "Has my order shipped?" and you reply with the order number, full contents, shipping address, and tracking number.
        * Good Example: User asks "Has my order shipped?" and you reply "Yes, I can confirm order #[Order ID] has shipped. Would you like the tracking information?"
        * Be Helpful, Friendly, and Professional: Maintain the IKEA brand voice.
        * Handle Errors Gracefully: If a tool returns an error or no data, inform the customer and ask for clarification (e.g., "I couldn't find that order ID. Could you please double-check the number?").
        """
    
        # Display assistant thinking message immediately
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            
            with thinking_placeholder.container():
                with st.spinner("Thinking..."):
                    response = agent(f"{system_prompt}\n\n{conversation_context}")
            
            # Debug: Check if response is valid
            if response is None or str(response).strip() == "":
                response = "I apologize, but I'm having trouble processing your request. Please try again."                                                                                      
            # Replace thinking message with actual response
            thinking_placeholder.write(str(response))
        
        # Get sources used in this response
        sources = _get_last_sources()
        
        # Add assistant response to history
        st.session_state["conversation_history"].append({
            "role": "assistant",
            "content": str(response),
            "sources": sources,
            "timestamp": time.time()
        })
        
    except Exception as e:
        # Handle any errors in response generation
        error_response = f"I apologize, but I encountered an error: {str(e)}. Please try again."
        
        # Display error message immediately
        with st.chat_message("assistant"):
            st.write(error_response)
        
        st.session_state["conversation_history"].append({
            "role": "assistant",
            "content": error_response,
            "sources": [],
            "timestamp": time.time()
        })
    
    # Rerun to show the new messages in the conversation history
    st.rerun()
