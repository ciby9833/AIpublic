# backend/feishu_approval /approval_controller.py 飞书审批控制器
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from database import get_db
from auth.oauth import get_current_user
from feishu_approval.approval_service import ApprovalService
from typing import Optional
import uuid

router = APIRouter()

@router.get("/api/approvals/instance/{instance_code}")
async def get_approval_instance(
    instance_code: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """获取审批实例详情"""
    try:
        # 验证用户
        if not current_user:
            raise HTTPException(
                status_code=401,
                detail={"code": "UNAUTHORIZED", "message": "未授权的请求"}
            )
        
        # 调用服务
        approval_service = ApprovalService(db)
        result = await approval_service.get_approval_instance(
            instance_code=instance_code,
            user_id=current_user.id
        )
        
        # 检查错误
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail={"code": "API_ERROR", "message": result["error"]}
            )
        
        return {
            "code": 0,
            "data": {
                "instance": result
            },
            "msg": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "SERVER_ERROR", "message": f"服务器内部错误: {str(e)}"}
        )