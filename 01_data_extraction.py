"""
01 - DATA EXTRACTION
Extract customer purchase data from AdventureWorks2025 SQL Server database.
"""
import pandas as pd
import os
from pathlib import Path

try:
    import pyodbc
    HAS_PYODBC = True
except ImportError:
    HAS_PYODBC = False
    print("pyodbc not installed. Will generate sample data.")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def get_connection():
    """Create SQL Server connection. Update connection string as needed."""
    if not HAS_PYODBC:
        return None
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-GBBP1HL;"          # Update server name
        "DATABASE=AdventureWorks2025;"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)

def extract_customer_features(conn):
    """Extract customer features for churn prediction."""
    query = """
    WITH CalculoDesfase AS (
    -- Calculamos la diferencia exacta en DIAS entre la última orden real y hoy
    SELECT DATEDIFF(DAY, MAX([OrderDate]), GETDATE()) AS DiasFaltantes
    FROM Sales.SalesOrderHeader
)
SELECT 
    c.CustomerID,
    c.AccountNumber,
    -- Total orders
    COUNT(DISTINCT soh.SalesOrderID) as total_orders,
    -- Total spending
    SUM(sod.LineTotal) as total_spent,
    -- Average order value
    AVG(sod.LineTotal) as avg_order_value,
    -- Days since last purchase (calculado sumando el desfase exacto en días)
    DATEDIFF(DAY, DATEADD(DAY, (SELECT DiasFaltantes FROM CalculoDesfase), MAX(soh.OrderDate)), GETDATE()) as days_since_last_purchase,
    -- Days since first purchase (calculado sumando el desfase exacto en días)
    DATEDIFF(DAY, DATEADD(DAY, (SELECT DiasFaltantes FROM CalculoDesfase), MIN(soh.OrderDate)), GETDATE()) as days_since_first_purchase,
    -- Unique products purchased
    COUNT(DISTINCT sod.ProductID) as unique_products,
    -- Favorite product category
    (SELECT TOP 1 pc.Name 
     FROM Sales.SalesOrderHeader soh2
     JOIN Sales.SalesOrderDetail sod2 ON soh2.SalesOrderID = sod2.SalesOrderID
     JOIN Production.Product p2 ON sod2.ProductID = p2.ProductID
     JOIN Production.ProductSubcategory psc2 ON p2.ProductSubcategoryID = psc2.ProductSubcategoryID
     JOIN Production.ProductCategory pc ON psc2.ProductCategoryID = pc.ProductCategoryID
     WHERE soh2.CustomerID = c.CustomerID
     GROUP BY pc.Name
     ORDER BY COUNT(*) DESC) as favorite_category,
    -- Average days between orders (la distancia entre órdenes en días se mantiene intacta)
    CASE 
        WHEN COUNT(DISTINCT soh.SalesOrderID) > 1 
        THEN DATEDIFF(DAY, MIN(soh.OrderDate), MAX(soh.OrderDate)) / (COUNT(DISTINCT soh.SalesOrderID) - 1.0)
        ELSE 9999
    END as avg_days_between_orders,
    -- Order frequency (orders per month)
    CASE 
        WHEN DATEDIFF(MONTH, MIN(soh.OrderDate), MAX(soh.OrderDate)) > 0 
        THEN COUNT(DISTINCT soh.SalesOrderID) * 1.0 / DATEDIFF(MONTH, MIN(soh.OrderDate), MAX(soh.OrderDate))
        ELSE COUNT(DISTINCT soh.SalesOrderID)
    END as orders_per_month
FROM Sales.Customer c
LEFT JOIN Sales.SalesOrderHeader soh ON c.CustomerID = soh.CustomerID
LEFT JOIN Sales.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID -- Nota: Corregido a OrderID o SalesOrderID según corresponda
WHERE c.CustomerID IS NOT NULL
GROUP BY c.CustomerID, c.AccountNumber;
    """
    return pd.read_sql(query, conn)

def create_target_variable(df, months_threshold=6):
    """
    Create target variable: will_buy_again
    1 = Customer made purchase within threshold
    0 = Customer did not (churned)
    """
    # Get the most recent order date in the dataset
    max_date_query = "SELECT MAX(OrderDate) FROM Sales.SalesOrderHeader"
    
    # For now, use a simple rule: if days_since_last_purchase <= threshold * 30
    df['will_buy_again'] = (df['days_since_last_purchase'] <= months_threshold * 30).astype(int)
    return df

def main():
    print("Connecting to AdventureWorks2025 database...")
    
    try:
        conn = get_connection()
        if conn is None:
            raise Exception("No database connection available")
        print("Connected successfully!")
        
        print("Extracting customer features...")
        df = extract_customer_features(conn)
        
        print(f"Extracted {len(df)} customers")
        
        print("Creating target variable...")
        df = create_target_variable(df)
        
        # Save raw data
        output_path = DATA_DIR / "raw_customer_features.csv"
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        
        # Print summary
        print("\nTarget Distribution:")
        print(df['will_buy_again'].value_counts(normalize=True))
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nGenerating sample data for demonstration...")
        
        # Generate sample data for testing
        import numpy as np
        np.random.seed(42)
        
        n_customers = 1000
        df = pd.DataFrame({
            'CustomerID': range(1, n_customers + 1),
            'AccountNumber': [f'AW{i:05d}' for i in range(1, n_customers + 1)],
            'total_orders': np.random.poisson(3, n_customers) + 1,
            'total_spent': np.random.exponential(500, n_customers),
            'avg_order_value': np.random.normal(150, 50, n_customers),
            'days_since_last_purchase': np.random.exponential(100, n_customers).astype(int),
            'days_since_first_purchase': np.random.uniform(30, 1000, n_customers).astype(int),
            'unique_products': np.random.poisson(2, n_customers) + 1,
            'favorite_category': np.random.choice(['Bikes', 'Components', 'Clothing', 'Accessories'], n_customers),
            'avg_days_between_orders': np.random.exponential(45, n_customers),
            'orders_per_month': np.random.exponential(0.5, n_customers)
        })
        
        df['will_buy_again'] = (df['days_since_last_purchase'] <= 180).astype(int)
        
        output_path = DATA_DIR / "raw_customer_features.csv"
        df.to_csv(output_path, index=False)
        print(f"Sample data saved to {output_path}")
        
        print("\nTarget Distribution:")
        print(df['will_buy_again'].value_counts(normalize=True))

if __name__ == "__main__":
    main()
