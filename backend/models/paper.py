# backend/models/paper.py  文档阅读
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, ARRAY
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from database import Base
import uuid

class PaperAnalysis(Base):
    __tablename__ = "paper_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(UUID(as_uuid=True), nullable=False)
    filename = Column(String(255), nullable=False)
    content = Column(Text)
    translated_content = Column(Text)
    translation_lang = Column(String(10))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    original_content = Column(Text)
    processed_content = Column(Text)
    file_type = Column(String(10))
    file_size = Column(Integer)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))

class PaperQuestion(Base):
    __tablename__ = "paper_questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    paper_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False, server_default='{}')