from sqlalchemy import Integer , Column , String , create_engine 
from sqlalchemy.orm import DeclarativeBase , sessionmaker 

# Create URL 
DATABASE_URL="sqlite:///.SQL_DataBase.db"

# Create engine 
engine = create_engine(url=DATABASE_URL , connect_args={'check_same_thread':False})

# declare base 
class base(DeclarativeBase):
    pass
# session create 
session = sessionmaker(bind=engine , autoflush=False , autocommit=False)

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
    finally : 
        db.close()
    
    
    