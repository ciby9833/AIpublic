# backend/models/chat.py 会话模型 会话关联主题和文档
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Boolean, Float, ARRAY
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

# ✅ 新增网页收藏夹模型
class WebBookmark(Base):
    __tablename__ = "web_bookmarks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    web_id = Column(UUID(as_uuid=True), ForeignKey('paper_analysis.paper_id'), nullable=False)
    folder_name = Column(String(100), default="默认")  # 收藏夹分组
    notes = Column(Text)  # 用户备注
    tags = Column(ARRAY(String), server_default='{}')  # 用户标签
    is_favorite = Column(Boolean, default=False)  # 是否加星标
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# ✅ 新增内容类型枚举配置表
class ContentTypeConfig(Base):
    __tablename__ = "content_type_configs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_type = Column(String(20), nullable=False)  # "document", "web", "mixed"
    display_name = Column(String(50), nullable=False)  # 显示名称
    icon = Column(String(50))  # 图标名称
    color = Column(String(20))  # 颜色代码
    description = Column(Text)  # 描述
    is_active = Column(Boolean, default=True)
    sort_order = Column(Integer, default=0)

# ✅ 新增智能推荐模型
class ContentRecommendation(Base):
    __tablename__ = "content_recommendations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    content_id = Column(UUID(as_uuid=True), nullable=False)  # 推荐的内容ID
    content_type = Column(String(20), nullable=False)  # "document" 或 "web"
    recommendation_type = Column(String(20), nullable=False)  # "similar", "related", "trending"
    score = Column(Float, default=0.0)  # 推荐分数
    reason = Column(Text)  # 推荐理由
    is_shown = Column(Boolean, default=False)  # 是否已展示
    is_clicked = Column(Boolean, default=False)  # 是否已点击
    created_at = Column(DateTime(timezone=True), server_default=func.now())
