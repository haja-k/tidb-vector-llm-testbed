import pymysql
from config import Config

# Connection details from .env file
host = Config.TIDB_HOST
port = Config.TIDB_PORT
user = Config.TIDB_USER
password = Config.TIDB_PASSWORD
database = Config.TIDB_DATABASE

try:
    connection = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )
    with connection.cursor() as cursor:
        cursor.execute("SELECT VERSION();")
        result = cursor.fetchone()
        print(f"Connected successfully! TiDB version: {result[0]}")

        # Test user permissions
        cursor.execute("SELECT USER();")
        user_result = cursor.fetchone()
        print(f"Connected as: {user_result[0]}")

    connection.close()
except pymysql.Error as e:
    print(f"MySQL/TiDB Error: {e}")
    print("Check credentials, permissions, and TiProxy setup.")
except Exception as e:
    print(f"General error: {e}")