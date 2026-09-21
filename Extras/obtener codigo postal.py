import time
import pandas as pd
from sqlalchemy import create_engine
from geopy.geocoders import ArcGIS
from geopy.extra.rate_limiter import RateLimiter

# 1. Configuración de conexión
server = "DESKTOP-GBBP1HL"
source_db = "AdventureWorks2025"
target_db = "GeoData"

engine_aw = create_engine(
    f"mssql+pyodbc://{server}/{source_db}"
    "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)
engine_geo = create_engine(
    f"mssql+pyodbc://{server}/{target_db}"
    "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

# 2. Obtener los códigos postales únicos por ciudad y país
query = """
SELECT DISTINCT
    a.PostalCode,
    a.City,
    pc.Name AS CountryRegionName
FROM Person.Address a
LEFT JOIN Person.StateProvince sp
    ON sp.StateProvinceID = a.StateProvinceID
LEFT JOIN Person.CountryRegion pc
    ON pc.CountryRegionCode = sp.CountryRegionCode
WHERE a.PostalCode IS NOT NULL
ORDER BY a.City, a.PostalCode;
"""

print("Consultando códigos postales únicos desde AdventureWorks2025...")
postal_df = pd.read_sql(query, con=engine_aw)

# 3. Eliminar duplicados exactos de PostalCode + City + Country
postal_df = postal_df.drop_duplicates(subset=['PostalCode', 'City', 'CountryRegionName']).reset_index(drop=True)
print(f"Se encontraron {len(postal_df)} códigos postales únicos para geocodificar.")

# 4. Geocodificar cada postal code único con ArcGIS
locator = ArcGIS()
geocode = RateLimiter(locator.geocode, min_delay_seconds=1, max_retries=2, swallow_exceptions=True)

geo_rows = []
for _, row in postal_df.iterrows():
    query_text = f"{row['PostalCode']}, {row['City']}, {row['CountryRegionName']}"
    try:
        location = geocode(query_text)
        lat = location.latitude if location else None
        lon = location.longitude if location else None
    except Exception:
        lat = None
        lon = None

    geo_rows.append({
        "PostalCode": row['PostalCode'],
        "City": row['City'],
        "CountryRegionName": row['CountryRegionName'],
        "Latitude": lat,
        "Longitude": lon,
    })

    time.sleep(0.5)

postal_geo = pd.DataFrame(geo_rows)

# 5. Dejar solo el dataset limpio final
final_df = postal_df.merge(
    postal_geo,
    on=['PostalCode', 'City', 'CountryRegionName'],
    how='left'
)

final_df = final_df[['PostalCode', 'City', 'CountryRegionName', 'Latitude', 'Longitude']]

print(final_df.head())
print(f"\nSe recuperaron {len(final_df)} registros. Lat/Long disponibles en {final_df['Latitude'].notna().sum()} filas.")

# 6. Guardar en geo_customers
nombre_tabla = "geo_customers"
final_df.to_sql(name=nombre_tabla, con=engine_geo, schema='dbo', if_exists='replace', index=False)
print(f"Datos guardados en dbo.{nombre_tabla} dentro de la base GeoData.")

print("Proceso finalizado.")