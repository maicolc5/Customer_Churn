"""
01 - DATA EXTRACTION
Extract customer purchase data from AdventureWorks2025 SQL Server database.
"""
import pandas as pd
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
        "SERVER=DESKTOP-GBBP1HL;"
        "DATABASE=AdventureWorks2025;"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)


def extract_customer_features(conn):
    """Extract historical features and future purchases around a cutoff."""
    query = """
WITH Fechas AS (
    SELECT
        MAX(OrderDate) AS FechaMaximaOriginal,
        CAST(GETDATE() AS date) AS FechaHoy,
        DATEDIFF(DAY, MAX(OrderDate), CAST(GETDATE() AS date)) AS DiasDesfase,
        DATEADD(MONTH, -6, CAST(GETDATE() AS date)) AS FechaCorte
    FROM Sales.SalesOrderHeader
),
ComprasHistoricas AS (
    SELECT
        c.CustomerID,
        c.AccountNumber,
        customer_address.City AS city,
        COUNT(DISTINCT soh.SalesOrderID) AS total_orders,
        SUM(sod.LineTotal) AS total_spent,
        AVG(sod.LineTotal) AS avg_order_value,
        DATEDIFF(
            DAY,
            DATEADD(DAY, f.DiasDesfase, MAX(soh.OrderDate)),
            f.FechaHoy
        ) AS days_since_last_purchase,
        DATEDIFF(
            DAY,
            DATEADD(DAY, f.DiasDesfase, MIN(soh.OrderDate)),
            f.FechaHoy
        ) AS days_since_first_purchase,
        COUNT(DISTINCT sod.ProductID) AS unique_products,
        (SELECT TOP 1 pc.Name
         FROM Sales.SalesOrderHeader soh2
         JOIN Sales.SalesOrderDetail sod2 ON soh2.SalesOrderID = sod2.SalesOrderID
         JOIN Production.Product p2 ON sod2.ProductID = p2.ProductID
         JOIN Production.ProductSubcategory psc2 ON p2.ProductSubcategoryID = psc2.ProductSubcategoryID
         JOIN Production.ProductCategory pc ON psc2.ProductCategoryID = pc.ProductCategoryID
         WHERE soh2.CustomerID = c.CustomerID
           AND DATEADD(DAY, f.DiasDesfase, soh2.OrderDate) <= f.FechaCorte
         GROUP BY pc.Name
         ORDER BY COUNT(*) DESC) AS favorite_category,
        CASE
            WHEN COUNT(DISTINCT soh.SalesOrderID) > 1
            THEN DATEDIFF(DAY, MIN(soh.OrderDate), MAX(soh.OrderDate)) /
                 (COUNT(DISTINCT soh.SalesOrderID) - 1.0)
            ELSE 9999
        END AS avg_days_between_orders,
        CASE
            WHEN DATEDIFF(MONTH, MIN(soh.OrderDate), MAX(soh.OrderDate)) > 0
            THEN COUNT(DISTINCT soh.SalesOrderID) * 1.0 /
                 DATEDIFF(MONTH, MIN(soh.OrderDate), MAX(soh.OrderDate))
            ELSE COUNT(DISTINCT soh.SalesOrderID)
        END AS orders_per_month
    FROM Sales.Customer c
    CROSS JOIN Fechas f
    OUTER APPLY (
        SELECT TOP 1
            a.City
        FROM Person.BusinessEntityAddress bea
        JOIN Person.Address a
            ON bea.AddressID = a.AddressID
        WHERE bea.BusinessEntityID = c.PersonID
        ORDER BY bea.AddressID
    ) AS customer_address
    LEFT JOIN Sales.SalesOrderHeader soh
        ON c.CustomerID = soh.CustomerID
        AND DATEADD(DAY, f.DiasDesfase, soh.OrderDate) <= f.FechaCorte
    LEFT JOIN Sales.SalesOrderDetail sod
        ON soh.SalesOrderID = sod.SalesOrderID
    WHERE c.CustomerID IS NOT NULL
    GROUP BY
        c.CustomerID,
        c.AccountNumber,
        customer_address.City,
        f.FechaCorte,
        f.DiasDesfase,
        f.FechaHoy
),
ComprasFuturas AS (
    SELECT
        soh.CustomerID,
        COUNT(DISTINCT CASE
            WHEN DATEADD(DAY, f.DiasDesfase, soh.OrderDate)
                 <= DATEADD(DAY, 90, f.FechaCorte)
            THEN soh.SalesOrderID
        END) AS future_orders_90d,
        COUNT(DISTINCT soh.SalesOrderID) AS future_orders_6m
    FROM Sales.SalesOrderHeader soh
    CROSS JOIN Fechas f
        WHERE DATEADD(DAY, f.DiasDesfase, soh.OrderDate) > f.FechaCorte
            AND DATEADD(DAY, f.DiasDesfase, soh.OrderDate) <= f.FechaHoy
    GROUP BY soh.CustomerID
)
SELECT
    h.*,
    COALESCE(f.future_orders_90d, 0) AS future_orders_90d,
    COALESCE(f.future_orders_6m, 0) AS future_orders_6m,
    CASE
        WHEN h.total_orders > 0 AND COALESCE(f.future_orders_6m, 0) > 0
            THEN 'existing_returned'
        WHEN h.total_orders > 0 AND COALESCE(f.future_orders_6m, 0) = 0
            THEN 'existing_not_returned'
        WHEN h.total_orders = 0 AND COALESCE(f.future_orders_6m, 0) > 0
            THEN 'new_customer'
        ELSE 'never_purchased'
    END AS customer_cohort
FROM ComprasHistoricas h
LEFT JOIN ComprasFuturas f ON h.CustomerID = f.CustomerID;
    """
    return pd.read_sql(query, conn)


def create_target_variable(df):
    """Create targets for short-term and six-month repurchase."""
    if 'customer_cohort' not in df.columns:
        has_history = df['total_orders'] > 0
        has_future_purchase = df['future_orders_6m'] > 0
        df['customer_cohort'] = 'never_purchased'
        df.loc[has_history & has_future_purchase, 'customer_cohort'] = 'existing_returned'
        df.loc[has_history & ~has_future_purchase, 'customer_cohort'] = 'existing_not_returned'
        df.loc[~has_history & has_future_purchase, 'customer_cohort'] = 'new_customer'
    df['will_buy_soon'] = (df['future_orders_90d'] > 0).astype(int)
    df['will_buy_again_6m'] = (df['future_orders_6m'] > 0).astype(int)
    return df.drop(columns=['future_orders_90d', 'future_orders_6m'])


def create_sample_data(n_customers=1000):
    """Create reproducible data when SQL Server is unavailable."""
    import numpy as np

    np.random.seed(42)
    df = pd.DataFrame({
        'CustomerID': range(1, n_customers + 1),
        'AccountNumber': [f'AW{i:05d}' for i in range(1, n_customers + 1)],
        'total_orders': np.random.poisson(3, n_customers) + 1,
        'total_spent': np.random.exponential(500, n_customers),
        'avg_order_value': np.random.normal(150, 50, n_customers),
        'days_since_last_purchase': np.random.exponential(100, n_customers).astype(int),
        'days_since_first_purchase': np.random.uniform(30, 1000, n_customers).astype(int),
        'unique_products': np.random.poisson(2, n_customers) + 1,
        'favorite_category': np.random.choice(
            ['Bikes', 'Components', 'Clothing', 'Accessories'], n_customers
        ),
        'city': np.random.choice(
            ['Seattle', 'Boston', 'Phoenix', 'Dallas'], n_customers
        ),
        'avg_days_between_orders': np.random.exponential(45, n_customers),
        'orders_per_month': np.random.exponential(0.5, n_customers),
        'future_orders_90d': np.random.binomial(1, 0.20, n_customers),
        'future_orders_6m': np.random.binomial(1, 0.35, n_customers)
    })
    return create_target_variable(df)


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

        df = create_target_variable(df)
        output_path = DATA_DIR / "raw_customer_features.csv"
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        print("\nTarget Distribution:")
        print(df['will_buy_again'].value_counts(normalize=True))
        conn.close()

    except Exception as error:
        print(f"Error: {error}")
        print("\nGenerating sample data for demonstration...")
        df = create_sample_data()
        output_path = DATA_DIR / "raw_customer_features.csv"
        df.to_csv(output_path, index=False)
        print(f"Sample data saved to {output_path}")
        print("\nTarget Distribution:")
        print(df['will_buy_again'].value_counts(normalize=True))


if __name__ == "__main__":
    main()
