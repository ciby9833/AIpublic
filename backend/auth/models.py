from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
import uuid
from sqlalchemy.sql import func

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    feishu_user_id = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    en_name = Column(String(255))
    email = Column(String(255))
    mobile = Column(String(255))
    avatar_url = Column(String(1024))
    tenant_key = Column(String(255), nullable=False)
    
    # 认证相关
    access_token = Column(String(255))
    refresh_token = Column(String(255))
    token_expires_at = Column(DateTime)
    
    # 用户状态
    is_active = Column(Boolean, default=True)
    last_login_at = Column(DateTime)
    login_count = Column(Integer, default=0)

    # 关联关系 历史对话模型需要
    chat_sessions = relationship("ChatSession", back_populates="user")
    
    # 审计字段
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def update_login(self, access_token: str, expires_at: datetime):
        """更新登录信息"""
        self.access_token = access_token
        self.token_expires_at = expires_at
        self.last_login_at = datetime.utcnow()
        self.login_count += 1

# 修改ChatSession模型
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    paper_id = Column(UUID(as_uuid=True), nullable=True)  # 保留向后兼容
    title = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    session_type = Column(String(20), default="general")  # "general" 或 "document"
    paper_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)  # 添加此行支持多文档
    is_ai_only = Column(Boolean, default=False)  # 添加此行支持纯AI对话
    
    # 关联关系
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    documents = relationship("SessionDocument", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('chat_sessions.id', ondelete='CASCADE'), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' 或 'assistant'
    content = Column(Text, nullable=False)  # 保留兼容性
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    sources = Column(JSONB)  # 存储参考来源信息
    confidence = Column(Float)  # 存储回答的可信度
    document_id = Column(UUID(as_uuid=True), nullable=True)  # 当消息引用特定文档时
    message_type = Column(String(20), default="text")  # text, file, image 等
    
    # 新增字段：存储富文本内容
    rich_content = Column(JSONB, nullable=True)  # 存储结构化的富文本内容 数据库需要更新
    
    # 关联关系
    session = relationship("ChatSession", back_populates="messages")

# 新增会话-文档关联表
class SessionDocument(Base):
    __tablename__ = "session_documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('chat_sessions.id', ondelete='CASCADE'), nullable=False)
    paper_id = Column(UUID(as_uuid=True), nullable=False)  # 引用paper_analysis.paper_id
    added_at = Column(DateTime(timezone=True), server_default=func.now())
    order = Column(Integer, default=0)
    filename = Column(String(255))
    
    # 关联关系
    session = relationship("ChatSession", back_populates="documents")
