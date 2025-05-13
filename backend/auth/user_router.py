# backend/auth/user_router.py  用户管理
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from database import get_db
from .models import User
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .oauth import get_current_user
from datetime import datetime

# 创建安全机制
security = HTTPBearer()

# 创建一个新的路由器,使用不同的前缀避免与 oauth 路由冲突
router = APIRouter(prefix="/auth/users")

# 修改认证依赖
async def get_authenticated_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """获取认证用户"""
    token = credentials.credentials
    return await get_current_user(db=db, access_token=token)

@router.get("", response_model=dict)
async def get_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_authenticated_user)
):
    """获取用户列表,带分页"""
    try:
        # 计算分页偏移
        skip = (page - 1) * page_size
        
        # 获取总数
        total = db.query(User).count()
        
        # 获取分页数据
        users = (db.query(User)
                .order_by(User.created_at.desc())
                .offset(skip)
                .limit(page_size)
                .all())
        
        # 转换为响应格式
        user_list = [{
            "id": str(user.id),
            "name": user.name,
            "email": user.email,
            "en_name": user.en_name,
            "avatar_url": user.avatar_url,
            "last_login": user.last_login_at,
            "status": "active" if user.is_active else "inactive",
            "login_count": user.login_count,
            "created_at": user.created_at,
            "tenant_key": user.tenant_key
        } for user in users]
        
        return {
            "items": user_list,
            "total": total,
            "page": page,
            "page_size": page_size
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "USER_LIST_ERROR", "message": str(e)}
        )

@router.get("/search", response_model=dict)
async def search_users(
    q: str = Query(""),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_authenticated_user)
):
    """搜索用户"""
    try:
        # 构建搜索查询
        query = db.query(User)
        if q:
            query = query.filter(
                (User.name.ilike(f"%{q}%")) |
                (User.email.ilike(f"%{q}%")) |
                (User.en_name.ilike(f"%{q}%"))
            )
        
        # 获取结果
        users = query.order_by(User.created_at.desc()).all()
        
        # 转换为响应格式
        user_list = [{
            "id": str(user.id),
            "name": user.name,
            "email": user.email,
            "en_name": user.en_name,
            "avatar_url": user.avatar_url,
            "last_login": user.last_login_at,
            "status": "active" if user.is_active else "inactive",
            "login_count": user.login_count,
            "created_at": user.created_at,
            "tenant_key": user.tenant_key
        } for user in users]
        
        return {
            "items": user_list,
            "total": len(user_list)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "USER_SEARCH_ERROR", "message": str(e)}
        )