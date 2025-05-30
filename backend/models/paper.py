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

    # ✅ 新增网页支持字段
    source_url = Column(String(2048))  # 网页源URL，支持长URL
    web_metadata = Column(JSONB)  # 网页元数据（标题、描述、关键词等）
    web_links = Column(JSONB)  # 网页中提取的链接信息
    web_images = Column(JSONB)  # 网页中的图片信息
    fetch_time = Column(DateTime(timezone=True))  # 网页抓取时间
    content_hash = Column(String(64))  # 内容哈希，用于检测变更

class PaperQuestion(Base):
    __tablename__ = "paper_questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    paper_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False, server_default='{}')

# ✅ 新增网页索引表（独立存储网页向量索引）
class WebIndex(Base):
    __tablename__ = "web_indexes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    web_id = Column(UUID(as_uuid=True), ForeignKey('paper_analysis.paper_id'), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # 分块索引
    content = Column(Text, nullable=False)  # 分块内容
    embedding = Column(JSONB)  # 向量嵌入
    chunk_metadata = Column(JSONB)  # 分块元数据
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# ✅ 新增网页更新监控表
class WebMonitor(Base):
    __tablename__ = "web_monitors"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    web_id = Column(UUID(as_uuid=True), ForeignKey('paper_analysis.paper_id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    url = Column(String(2048), nullable=False)
    last_check = Column(DateTime(timezone=True), server_default=func.now())
    last_content_hash = Column(String(64))
    check_interval_hours = Column(Integer, default=24)  # 检查间隔（小时）
    is_active = Column(Boolean, default=True)
    change_detected = Column(Boolean, default=False)
    notification_sent = Column(Boolean, default=False)