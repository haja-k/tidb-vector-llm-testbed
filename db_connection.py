"""
Database connection module for TiDB.
Handles connection setup and table creation for vector storage.
"""

from sqlalchemy import create_engine, text, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from tidb_vector.sqlalchemy import VectorType
from config import Config

Base = declarative_base()


class DocumentVector(Base):
    """
    SQLAlchemy model for storing document embeddings in TiDB.
    """
    __tablename__ = Config.TABLE_NAME
    
    id = Column(String(255), primary_key=True)
    document = Column(Text, nullable=False)
    meta = Column(Text)  # JSON string for metadata
    embedding = Column(VectorType(dim=Config.VECTOR_DIMENSION), nullable=False)


class TiDBConnection:
    """
    Manages TiDB database connections and vector table operations.
    """
    
    def __init__(self):
        """Initialize TiDB connection with configuration settings."""
        self.connection_string = Config.get_tidb_connection_string()
        self.engine = None
        self.session = None
        
    def connect(self):
        """
        Establish connection to TiDB cluster.
        Returns:
            engine: SQLAlchemy engine instance
        """
        try:
            print(f"Connecting to TiDB at {Config.TIDB_HOST}:{Config.TIDB_PORT}...")
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT VERSION()"))
                version = result.fetchone()[0]
                print(f"Successfully connected to TiDB version: {version}")
            
            # Create session maker
            Session = sessionmaker(bind=self.engine)
            self.Session = Session
            self.session = Session()
            
            return self.engine
            
        except Exception as e:
            print(f"Error connecting to TiDB: {e}")
            raise
    
    def create_vector_table(self, drop_existing=False):
        """
        Create vector index table in TiDB.
        
        Args:
            drop_existing: If True, drop existing table before creating
        """
        try:
            with self.engine.connect() as conn:
                if drop_existing:
                    print(f"Dropping existing table {Config.TABLE_NAME} if exists...")
                    drop_table = text(f"DROP TABLE IF EXISTS {Config.TABLE_NAME}")
                    conn.execute(drop_table)
                    conn.commit()
                
                # Check if table exists
                check_table = text(f"""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = '{Config.TIDB_DATABASE}' 
                    AND table_name = '{Config.TABLE_NAME}'
                """)
                result = conn.execute(check_table)
                table_exists = result.fetchone()[0] > 0
                
                if not table_exists:
                    print(f"Creating vector table {Config.TABLE_NAME}...")
                    create_table = text(f"""
                        CREATE TABLE {Config.TABLE_NAME} (
                            id VARCHAR(255) PRIMARY KEY,
                            document TEXT NOT NULL,
                            meta TEXT,
                            embedding VECTOR({Config.VECTOR_DIMENSION}) NOT NULL
                        )
                    """)
                    conn.execute(create_table)
                    conn.commit()
                    print(f"Vector table created with dimension {Config.VECTOR_DIMENSION}.")
                else:
                    print(f"Vector table {Config.TABLE_NAME} already exists.")
            
            # Set up TiFlash replica (required for vector indexes)
            with self.engine.connect() as conn:
                print("Setting up TiFlash replica for vector index...")
                try:
                    # Set TiFlash replica count to 1
                    set_replica = text(f"""
                        ALTER TABLE {Config.TABLE_NAME} SET TIFLASH REPLICA 1
                    """)
                    conn.execute(set_replica)
                    conn.commit()
                    print("TiFlash replica set successfully. Waiting for replica to be available...")
                    
                    # Wait for TiFlash replica to be available (with timeout)
                    import time
                    max_wait = 60  # Maximum wait time in seconds
                    wait_interval = 2  # Check every 2 seconds
                    elapsed = 0
                    
                    while elapsed < max_wait:
                        check_replica = text(f"""
                            SELECT TIFLASH_REPLICA, AVAILABLE 
                            FROM information_schema.tiflash_replica 
                            WHERE TABLE_SCHEMA = '{Config.TIDB_DATABASE}' 
                            AND TABLE_NAME = '{Config.TABLE_NAME}'
                        """)
                        result = conn.execute(check_replica)
                        row = result.fetchone()
                        
                        if row and row[1] == 1:  # AVAILABLE = 1
                            print("TiFlash replica is available.")
                            break
                        
                        time.sleep(wait_interval)
                        elapsed += wait_interval
                        if elapsed % 10 == 0:  # Print status every 10 seconds
                            print(f"Still waiting for TiFlash replica... ({elapsed}s elapsed)")
                    
                    if elapsed >= max_wait:
                        print(f"Warning: TiFlash replica setup timed out after {max_wait}s. Vector index creation may fail.")
                
                except Exception as e:
                    print(f"Note: TiFlash replica setup encountered an issue: {e}")
                    print("Attempting to create vector index anyway...")
            
            # Create vector index for efficient similarity search
            with self.engine.connect() as conn:
                # Check if index exists
                check_index = text(f"""
                    SELECT COUNT(*) 
                    FROM information_schema.statistics 
                    WHERE table_schema = '{Config.TIDB_DATABASE}' 
                    AND table_name = '{Config.TABLE_NAME}' 
                    AND index_name = 'vector_idx'
                """)
                result = conn.execute(check_index)
                index_exists = result.fetchone()[0] > 0
                
                if not index_exists:
                    print("Creating vector index for similarity search...")
                    create_index = text(f"""
                        ALTER TABLE {Config.TABLE_NAME} 
                        ADD VECTOR INDEX vector_idx((VEC_COSINE_DISTANCE(embedding)))
                    """)
                    conn.execute(create_index)
                    conn.commit()
                    print("Vector index created successfully.")
                else:
                    print("Vector index already exists.")
            
            print("Vector table setup completed successfully.")
            
        except Exception as e:
            print(f"Error creating vector table: {e}")
            raise
    
    def close(self):
        """Close database connections."""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
        print("Database connections closed.")
