# backend/feishu_approval/approval_service.py 审批服务 飞书审批服务
import os
import httpx
import logging
from typing import Optional, Dict
from datetime import datetime
from sqlalchemy.orm import Session
from feishu_approval.approval import ApprovalInstance
import json
import uuid

logger = logging.getLogger(__name__)

class ApprovalService:
    def __init__(self, db: Session):
        self.db = db
        self.app_id = os.getenv("FEISHU_APP_ID")
        self.app_secret = os.getenv("FEISHU_APP_SECRET")
        self.base_url = "https://open.feishu.cn/open-apis"
        
    async def _get_tenant_access_token(self) -> str:
        """获取tenant_access_token"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/auth/v3/tenant_access_token/internal",
                    json={
                        "app_id": self.app_id,
                        "app_secret": self.app_secret
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"获取tenant_access_token失败: {response.text}")
                    raise Exception(f"获取tenant_access_token失败: {response.text}")
                
                data = response.json()
                return data.get("tenant_access_token")
        except Exception as e:
            logger.error(f"获取tenant_access_token异常: {str(e)}")
            raise
    
    async def get_approval_instance(self, instance_code: str, user_id: Optional[uuid.UUID] = None) -> Dict:
        """获取审批实例详情"""
        try:
            # 1. 获取access_token
            access_token = await self._get_tenant_access_token()
            
            # 2. 调用飞书API获取审批实例详情
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/approval/v4/instances/{instance_code}",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json; charset=utf-8"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"获取审批实例详情失败: {response.text}")
                    return {"error": f"获取审批实例详情失败: {response.text}"}
                
                data = response.json()
                
                if data.get("code") != 0:
                    logger.error(f"飞书API返回错误: {data}")
                    return {"error": f"飞书API返回错误: {data.get('msg')}"}
                
                instance_data = data.get("data", {}).get("instance", {})
                
                # 3. 处理返回的数据
                instance = {
                    "instance_code": instance_data.get("instance_code", ""),
                    "approval_code": instance_data.get("approval_code", ""),
                    "approval_name": instance_data.get("approval_name", ""),
                    "form": instance_data.get("form", []),
                    "status": instance_data.get("status", ""),
                    "requester_id": instance_data.get("requester", {}).get("id", ""),
                    "department_id": instance_data.get("requester", {}).get("department_id", ""),
                    "start_time": self._parse_timestamp(instance_data.get("start_time", 0)),
                    "end_time": self._parse_timestamp(instance_data.get("end_time", 0)),
                    "is_deleted": instance_data.get("is_deleted", False),
                    "serial_number": instance_data.get("serial_number", ""),
                    "task_list": instance_data.get("task_list", []),
                    "cc_list": instance_data.get("cc_list", []),
                    "comment_list": instance_data.get("comment_list", []),
                    "timeline": instance_data.get("timeline", []),
                    "widget_list": instance_data.get("widget_list", []),
                    "raw_response": data
                }
                
                # 4. 保存到数据库
                if user_id:
                    self._save_approval_instance(instance, user_id)
                
                return instance
        except Exception as e:
            logger.error(f"获取审批实例异常: {str(e)}")
            return {"error": f"获取审批实例异常: {str(e)}"}
    
    def _parse_timestamp(self, timestamp: int) -> Optional[datetime]:
        """解析飞书时间戳为datetime对象"""
        if not timestamp:
            return None
        # 飞书时间戳为秒级
        return datetime.fromtimestamp(timestamp)
    
    def _save_approval_instance(self, instance_data: Dict, user_id: uuid.UUID) -> None:
        """保存审批实例到数据库"""
        try:
            # 检查是否已存在
            existing = self.db.query(ApprovalInstance).filter(
                ApprovalInstance.instance_code == instance_data["instance_code"]
            ).first()
            
            if existing:
                # 更新现有记录
                for key, value in instance_data.items():
                    if key != "raw_response":  # 避免覆盖原始响应
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
                self.db.commit()
                logger.info(f"更新审批实例: {instance_data['instance_code']}")
            else:
                # 创建新记录
                approval_instance = ApprovalInstance(
                    instance_code=instance_data["instance_code"],
                    approval_code=instance_data["approval_code"],
                    approval_name=instance_data["approval_name"],
                    form=instance_data["form"],
                    status=instance_data["status"],
                    user_id=user_id,
                    requester_id=instance_data["requester_id"],
                    department_id=instance_data["department_id"],
                    start_time=instance_data["start_time"],
                    end_time=instance_data["end_time"],
                    is_deleted=instance_data["is_deleted"],
                    serial_number=instance_data["serial_number"],
                    task_list=instance_data["task_list"],
                    cc_list=instance_data["cc_list"],
                    comment_list=instance_data["comment_list"],
                    timeline=instance_data["timeline"],
                    widget_list=instance_data["widget_list"],
                    raw_response=instance_data["raw_response"]
                )
                
                self.db.add(approval_instance)
                self.db.commit()
                logger.info(f"保存审批实例: {instance_data['instance_code']}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"保存审批实例异常: {str(e)}")
            raise