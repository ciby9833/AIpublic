# backend/models/paper.py  文档阅读
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, ARRAY, Boolean
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, JSONB
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
    
    # 添加行号相关字段
    line_mapping = Column(JSONB)  # 存储行号与内容的映射关系
    total_lines = Column(Integer)  # 总行数
    translation_line_mapping = Column(JSONB)  # 存储翻译后的行号映射
    
    # 添加扫描件标记字段
    is_scanned = Column(Boolean, default=False)  # 标记是否为扫描件

    # 添加索引存储相关字段
    embeddings = Column(JSONB)  # 存储嵌入向量
    documents = Column(JSONB)  # 存储分块后的文档内容
    index_built = Column(Boolean, default=False)  # 标记是否已构建索引

    # 新增字段：文档摘要
    summary = Column(Text)  # 文档摘要，用于会话显示
    # 文档标签，便于分类和搜索
    tags = Column(ARRAY(String), server_default='{}')

    # 添加结构化数据字段
    structured_data = Column(JSONB)  # 用于存储Excel等结构化文件的JSON数据 513待更新

class PaperQuestion(Base):
    __tablename__ = "paper_questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    paper_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False, server_default='{}')