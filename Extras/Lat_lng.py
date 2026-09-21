import os
import pandas as pd
from sqlalchemy import create_engine

# 1. Configuración de rutas y base de datos
ruta_parquet = r"C:\Users\maico\OneDrive\Desktop\DATA SCIENTIST\Customer_Churn\Extras\geonames-all-cities-with-a-population-1000.parquet"
servidor = "localhost"              # Cambia por tu servidor si no es local (ej: 'localhost\\SQLEXPRESS')
base_datos = "GeoData"             # <--- Escribe aquí el nombre de tu base de datos
tabla_destino = "geonames_cities"

# Verificar si el archivo existe antes de iniciar
if not os.path.exists(ruta_parquet):
    raise FileNotFoundError(f"No se encontró el archivo en: {ruta_parquet}")

# 2. Cadena de conexión (Autenticación de Windows)
conexion_url = f"mssql+pyodbc://@{servidor}/{base_datos}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
engine = create_engine(conexion_url)

# 3. Lectura del archivo Parquet
print("Leyendo archivo Parquet de GeoNames...")
df = pd.read_parquet(ruta_parquet)

# 4. Limpieza rápida de nombres de columnas (SQL Server no acepta ciertos caracteres especiales)
df.columns = df.columns.str.replace('[^a-zA-Z0-0_]', '_', regex=True)

# 5. Carga de datos a SQL Server
print(f"Cargando {len(df):,} filas en la tabla '{tabla_destino}'...")

# Usamos 'replace' para que cree la tabla automáticamente.
# 'chunksize=5000' y 'method=multi' aceleran drásticamente la inserción en SQL Server.
df.to_sql(
    name=tabla_destino,
    con=engine,
    if_exists='replace',
    index=False,
    chunksize=5000,
    method='multi'
)

print("¡Carga completada con éxito en SQL Server!")