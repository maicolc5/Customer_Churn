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
            DATEADD(MONTH, -6, CAST(GETDATE() AS date)) AS FechaCorte,
            DATEADD(
                DAY,
                -DATEDIFF(DAY, MAX(OrderDate), CAST(GETDATE() AS date)),
                DATEADD(MONTH, -6, CAST(GETDATE() AS date))
            ) AS FechaCorteOriginal,
            DATEADD(
                DAY,
                -DATEDIFF(DAY, MAX(OrderDate), CAST(GETDATE() AS date)),
                DATEADD(DAY, 90, DATEADD(MONTH, -6, CAST(GETDATE() AS date)))
            ) AS Fecha90Original,
            DATEADD(
                DAY,
                -DATEDIFF(DAY, MAX(OrderDate), CAST(GETDATE() AS date)),
                CAST(GETDATE() AS date)
            ) AS FechaHoyOriginal
    FROM Sales.SalesOrderHeader
),
FavoriteCategories AS (
    SELECT
        soh.CustomerID,
        pc.Name,
        ROW_NUMBER() OVER (
            PARTITION BY soh.CustomerID
            ORDER BY COUNT(*) DESC, pc.Name
        ) AS category_rank
    FROM Sales.SalesOrderHeader soh
    JOIN Sales.SalesOrderDetail sod
        ON soh.SalesOrderID = sod.SalesOrderID
    JOIN Production.Product p
        ON sod.ProductID = p.ProductID
    JOIN Production.ProductSubcategory psc
        ON p.ProductSubcategoryID = psc.ProductSubcategoryID
    JOIN Production.ProductCategory pc
        ON psc.ProductCategoryID = pc.ProductCategoryID
    CROSS JOIN Fechas f
    WHERE soh.OrderDate <= f.FechaCorteOriginal
    GROUP BY soh.CustomerID, pc.Name
),
HistoricalOrders AS (
    SELECT
        soh.CustomerID,
        COUNT(DISTINCT soh.SalesOrderID) AS total_orders,
        SUM(sod.LineTotal) AS total_spent,
        AVG(sod.LineTotal) AS avg_order_value,
        MIN(soh.OrderDate) AS first_order_date,
        MAX(soh.OrderDate) AS last_order_date,
        COUNT(DISTINCT sod.ProductID) AS unique_products,
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
    FROM Sales.SalesOrderHeader soh
    JOIN Sales.SalesOrderDetail sod
        ON soh.SalesOrderID = sod.SalesOrderID
    CROSS JOIN Fechas f
    WHERE soh.OrderDate <= f.FechaCorteOriginal
    GROUP BY soh.CustomerID
),
ComprasHistoricas AS (
    SELECT
        c.CustomerID,
        c.AccountNumber,
        COALESCE(person_address.City, store_address.City, geo.City) AS city,
        COALESCE(person_address.Country, store_address.Country, geo.county_name) AS country,
        COALESCE(person_address.PostalCode, store_address.PostalCode, geo.PostalCode) AS postal_code,
        geo.Latitude AS latitude,
        geo.Longitude AS longitude,
        geo.population AS population,
        COALESCE(ho.total_orders, 0) AS total_orders,
        COALESCE(ho.total_spent, 0) AS total_spent,
        COALESCE(ho.avg_order_value, 0) AS avg_order_value,
        COALESCE(DATEDIFF(DAY, ho.last_order_date, f.FechaHoyOriginal), 0) AS days_since_last_purchase,
        COALESCE(DATEDIFF(DAY, ho.first_order_date, f.FechaHoyOriginal), 0) AS days_since_first_purchase,
        COALESCE(ho.unique_products, 0) AS unique_products,
        favorite_category.Name AS favorite_category,
        COALESCE(ho.avg_days_between_orders, 9999) AS avg_days_between_orders,
        COALESCE(ho.orders_per_month, 0) AS orders_per_month
    FROM Sales.Customer c
    CROSS JOIN Fechas f
    OUTER APPLY (
        SELECT TOP 1
            a.City,
            a.PostalCode,
            COALESCE(cr.Name, 'Unknown') AS Country
        FROM Person.BusinessEntityAddress bea
        JOIN Person.Address a
            ON bea.AddressID = a.AddressID
        LEFT JOIN Person.StateProvince sp
            ON a.StateProvinceID = sp.StateProvinceID
        LEFT JOIN Person.CountryRegion cr
            ON sp.CountryRegionCode = cr.CountryRegionCode
        WHERE bea.BusinessEntityID = c.PersonID
        ORDER BY bea.AddressID
    ) AS person_address
    OUTER APPLY (
        SELECT TOP 1
            a.City,
            a.PostalCode,
            COALESCE(cr.Name, 'Unknown') AS Country
        FROM Person.BusinessEntityAddress bea
        JOIN Person.Address a
            ON bea.AddressID = a.AddressID
        LEFT JOIN Person.StateProvince sp
            ON a.StateProvinceID = sp.StateProvinceID
        LEFT JOIN Person.CountryRegion cr
            ON sp.CountryRegionCode = cr.CountryRegionCode
        WHERE bea.BusinessEntityID = c.StoreID
        ORDER BY bea.AddressID
    ) AS store_address
    LEFT JOIN GeoData.dbo.CustomerPopulation AS geo
        ON geo.CustomerID = c.CustomerID
    LEFT JOIN FavoriteCategories AS favorite_category
        ON favorite_category.CustomerID = c.CustomerID
        AND favorite_category.category_rank = 1
    LEFT JOIN HistoricalOrders ho
        ON ho.CustomerID = c.CustomerID
    WHERE c.CustomerID IS NOT NULL
),
ComprasFuturas AS (
    SELECT
        soh.CustomerID,
        COUNT(DISTINCT CASE
                        WHEN soh.OrderDate <= f.Fecha90Original
            THEN soh.SalesOrderID
        END) AS future_orders_90d,
        COUNT(DISTINCT soh.SalesOrderID) AS future_orders_6m
    FROM Sales.SalesOrderHeader soh
    CROSS JOIN Fechas f
    WHERE soh.OrderDate > f.FechaCorteOriginal
        AND soh.OrderDate <= f.FechaHoyOriginal
    GROUP BY
        soh.CustomerID,
        f.Fecha90Original
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
        print("\nTarget Distribution - 6 months:")
        print(df['will_buy_again_6m'].value_counts(normalize=True))

    except Exception as error:
        print(f"Error: {error}")

    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()
