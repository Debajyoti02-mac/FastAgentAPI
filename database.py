import os

from sqlalchemy import Integer, Column, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


# Create URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///.SQL_DataBase.db"
)


# Create engine
fallback_sqlite = "sqlite:///.SQL_DataBase.db"

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    try:
        pg_url = DATABASE_URL.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1
        )
        temp_engine = create_engine(pg_url)
        # Verify connection
        with temp_engine.connect() as conn:
            pass
        engine = temp_engine
    except Exception as e:
        print(f"[database.py] PostgreSQL connection error ({e}). Falling back to SQLite: {fallback_sqlite}")
        engine = create_engine(
            fallback_sqlite,
            connect_args={"check_same_thread": False}
        )


# Declare base
class base(DeclarativeBase):
    pass


# Session create
session = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)


# Create DB
class QueryHistory(base):
    __tablename__ = "Database_FAST"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(String)
    limit = Column(Integer)
    answer = Column(String)


base.metadata.create_all(bind=engine)


def create_db():
    db = session()
    try:
        yield db
    finally:
        db.close()