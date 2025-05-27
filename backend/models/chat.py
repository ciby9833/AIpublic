# backend/models/chat.py 会话模型 会话关联主题和文档
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Boolean, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from database import Base
from datetime import datetime
import uuid

# 会话主题/收藏模型
class ChatTopic(Base):
    __tablename__ = "chat_topics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    sessions = relationship("ChatTopicSession", back_populates="topic")

# 会话与主题的多对多关联表
class ChatTopicSession(Base):
    __tablename__ = "chat_topic_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_id = Column(UUID(as_uuid=True), ForeignKey('chat_topics.id', ondelete='CASCADE'), nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey('chat_sessions.id', ondelete='CASCADE'), nullable=False)
    added_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 关联关系
    topic = relationship("ChatTopic", back_populates="sessions")
    # 不需要双向关系，避免循环引用
    # session = relationship("ChatSession", back_populates="topics")

# 消息片段模型 - 用于存储大型消息的分片
class MessageFragment(Base):
    __tablename__ = "message_fragments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey('chat_messages.id', ondelete='CASCADE'), nullable=False)
    fragment_index = Column(Integer, nullable=False)  # 片段索引
    content = Column(JSONB, nullable=False)  # 片段内容
    content_hash = Column(String(32))  # 内容哈希，用于校验
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# 文件上传模型 - 用于无需分析的临时文件
class ChatFile(Base):
    __tablename__ = "chat_files"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey('chat_sessions.id'), nullable=False)
    message_id = Column(UUID(as_uuid=True), ForeignKey('chat_messages.id'), nullable=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50))  # MIME类型
    file_size = Column(Integer)  # 文件大小（字节）
    file_path = Column(String(512))  # 存储路径
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 额外描述
    description = Column(Text)  # 文件描述
    is_processed = Column(Boolean, default=False)  # 是否已处理
    processing_status = Column(String(20), default="pending")  # pending, processing, completed, failed
