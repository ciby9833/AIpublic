# backend/feishu_approval/approval.py 审批实例模型 飞书审批实例模型
from sqlalchemy import Column, String, DateTime, Boolean, JSON, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from database import Base
import uuid
from datetime import datetime

class ApprovalInstance(Base):
    __tablename__ = "approval_instances"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    instance_code = Column(String(255), unique=True, nullable=False)
    approval_code = Column(String(255), nullable=False)
    approval_name = Column(String(255))
    form = Column(JSON)
    status = Column(String(50))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    requester_id = Column(String(255))
    department_id = Column(String(255))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    is_deleted = Column(Boolean, default=False)
    serial_number = Column(String(255))
    task_list = Column(JSON)
    cc_list = Column(JSON)
    comment_list = Column(JSON)
    timeline = Column(JSON)
    widget_list = Column(JSON)
    raw_response = Column(JSON)
    
    # 审计字段
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)