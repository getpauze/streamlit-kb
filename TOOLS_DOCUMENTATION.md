# 🛠️ Order Management Tools Documentation

This document describes the specialized tools available in the order management system.

## 📋 Available Tools

### 1. 🔍 Order Lookup (`tool_order_lookup`)

**Purpose**: Get detailed information about a specific order by its ID.

**Parameters**:
- `order_id` (str): The order ID to look up (e.g., "ORD-001")

**Returns**: Complete order details including:
- Order information (ID, date, status, quantity, price, total)
- Customer details (name, email, phone, status)
- Product information (name, category, description, stock)
- Shipping address

**Example Queries**:
- "Show me details for order ORD-001"
- "What's the status of order ORD-005?"
- "Look up order ORD-010"

---

### 2. 👤 Customer Profile (`tool_customer_profile`)

**Purpose**: Get customer profile and complete order history.

**Parameters**:
- `customer_id` (str, optional): Customer ID to look up
- `email` (str, optional): Customer email to look up (alternative to customer_id)

**Returns**: Customer information including:
- Customer details (ID, name, email, phone, registration date, status)
- Order history with product names and amounts
- Total spent across all orders

**Example Queries**:
- "Show me John Smith's order history"
- "What orders has sarah.j@email.com placed?"
- "Get customer profile for CUST-003"

---

### 3. 🛍️ Product Search (`tool_product_search`)

**Purpose**: Search for products with multiple filter options.

**Parameters**:
- `category` (str, optional): Product category to filter by
- `min_price` (float, optional): Minimum price filter
- `max_price` (float, optional): Maximum price filter
- `in_stock` (bool, optional): Only show products in stock (default: True)
- `search_term` (str, optional): Search term for product name/description

**Returns**: Filtered product list with:
- Product ID, name, category, price
- Stock quantity and description
- Results count

**Example Queries**:
- "Show me all Electronics under $100"
- "Find products with 'wireless' in the name"
- "What furniture items are in stock?"
- "Show me products between $50 and $150"

---

### 4. 📊 Order Status Summary (`tool_order_status_summary`)

**Purpose**: Get business overview and key metrics.

**Parameters**: None

**Returns**: Business metrics including:
- Total orders and revenue
- Average order value
- Order status breakdown (processing, shipped, delivered, cancelled)
- Recent orders list

**Example Queries**:
- "What's our total revenue?"
- "How many orders are processing?"
- "Show me recent orders"
- "Give me a business summary"

---

### 5. 📦 Inventory Check (`tool_inventory_check`)

**Purpose**: Check stock levels and inventory status.

**Parameters**:
- `product_id` (str, optional): Specific product ID to check
- `category` (str, optional): Check all products in a category
- `low_stock_threshold` (int, optional): Threshold for low stock warning (default: 10)

**Returns**: Inventory status including:
- Current stock levels
- Low stock warnings
- Out-of-stock alerts
- Category-specific inventory

**Example Queries**:
- "What's the stock level for PROD-001?"
- "Which Electronics items are low on stock?"
- "Show me all out-of-stock products"
- "Check inventory for Furniture category"

---

## 🎯 Common Use Cases

### Customer Service Scenarios:
1. **Order Status Inquiry**: Use `tool_order_lookup` to check specific orders
2. **Customer History**: Use `tool_customer_profile` to view customer details and past orders
3. **Product Availability**: Use `tool_inventory_check` to verify stock levels
4. **Product Search**: Use `tool_product_search` to help customers find products

### Business Analytics:
1. **Revenue Tracking**: Use `tool_order_status_summary` for business metrics
2. **Inventory Management**: Use `tool_inventory_check` for stock monitoring
3. **Order Analysis**: Use `tool_order_status_summary` for order trends

### Product Management:
1. **Product Discovery**: Use `tool_product_search` with various filters
2. **Stock Monitoring**: Use `tool_inventory_check` for inventory alerts
3. **Category Analysis**: Use `tool_product_search` by category

## 🔧 Technical Notes

- All tools automatically load data from CSV files in the `data/` directory
- Tools handle missing data gracefully with appropriate error messages
- Cross-referenced data is automatically joined (e.g., orders with customer and product details)
- All monetary values are formatted with proper currency symbols
- Stock levels include visual indicators (✅ In Stock, ⚠️ Low Stock, 🚫 Out of Stock)

## 📁 Data Files

The tools work with these CSV files:
- `orders.csv` - Order transactions
- `customers.csv` - Customer information  
- `products.csv` - Product catalog

Make sure these files exist in the `data/` directory for the tools to function properly.
