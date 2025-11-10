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
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    metadata = Column(Text)  # JSON string for metadata
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
            if drop_existing:
                print(f"Dropping existing table {Config.TABLE_NAME} if exists...")
                Base.metadata.drop_all(self.engine, tables=[DocumentVector.__table__])
            
            print(f"Creating vector table {Config.TABLE_NAME}...")
            Base.metadata.create_all(self.engine, tables=[DocumentVector.__table__])
            
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
