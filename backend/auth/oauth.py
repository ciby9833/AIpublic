# backend/auth/oauth.py  飞书授权   
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
import httpx
import os
from typing import Optional
import traceback
import logging
import json
import urllib.parse
from datetime import datetime, timedelta
import secrets
import base64
from sqlalchemy.orm import Session
from .models import User
from fastapi import Depends
from database import get_db

router = APIRouter()

FEISHU_APP_ID = os.getenv("FEISHU_APP_ID")  # 飞书应用ID
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET")  # 飞书应用密钥
FEISHU_REDIRECT_URI = os.getenv("FEISHU_REDIRECT_URI")  # 飞书回调URL
FEISHU_TENANT_KEY = os.getenv("FEISHU_TENANT_KEY")  # 飞书组织ID
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")  # 前端URL

# 添加常量配置飞书token过期时间
TOKEN_EXPIRE_HOURS = 168  # token 过期时间，168小时

logger = logging.getLogger(__name__)

@router.get("/api/auth/feishu/login")
async def feishu_login():
    """生成飞书授权链接"""
    # 生成随机状态值
    state = secrets.token_urlsafe(16)
    
    auth_url = (
        "https://open.feishu.cn/open-apis/authen/v1/index"
        f"?app_id={FEISHU_APP_ID}"
        f"&redirect_uri={FEISHU_REDIRECT_URI}"
        f"&state={state}"  # 添加状态参数
    )
    
    logger.info(f"Generated auth URL with state: {state}")
    return {"auth_url": auth_url}

@router.get("/api/auth/feishu/callback")
async def feishu_callback(code: str, state: str, db: Session = Depends(get_db)):
    """处理飞书回调"""
    try:
        logger.info(f"Received callback with state: {state}")
        logger.info("Starting Feishu callback process")
        
        # 1. 获取访问令牌
        async with httpx.AsyncClient() as client:
            logger.info("Requesting access token from Feishu")
            token_response = await client.post(
                "https://open.feishu.cn/open-apis/authen/v1/access_token",
                json={
                    "app_id": FEISHU_APP_ID,
                    "app_secret": FEISHU_APP_SECRET,
                    "grant_type": "authorization_code",
                    "code": code
                }
            )
            token_data = token_response.json()
            logger.info("Access token received successfully")
            
            if token_response.status_code != 200:
                logger.error(f"Failed to get access token: {token_data}")
                return RedirectResponse(
                    url=f"{FRONTEND_URL}/login?error=token_failed"
                )
            
            access_token = token_data["data"]["access_token"]

        # 2. 获取用户信息
        async with httpx.AsyncClient() as client:
            logger.info("Requesting user info from Feishu")
            user_response = await client.get(
                "https://open.feishu.cn/open-apis/authen/v1/user_info",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            user_data = user_response.json()
            
            if user_response.status_code != 200:
                logger.error(f"Failed to get user info: {user_data}")
                return RedirectResponse(
                    url=f"{FRONTEND_URL}/login?error=user_info_failed"
                )

            logger.info(f"User info received for user: {user_data['data']['name']}")

            # 添加响应内容检查
            if not user_data.get("data"):
                logger.error(f"Invalid user data response: {user_data}")
                return RedirectResponse(
                    url=f"{FRONTEND_URL}/login?error=invalid_user_data"
                )
            
            required_fields = ["name", "tenant_key"]
            missing_fields = [
                field for field in required_fields 
                if not user_data["data"].get(field)
            ]
            
            if missing_fields:
                logger.error(f"Missing required fields: {missing_fields}")
                return RedirectResponse(
                    url=f"{FRONTEND_URL}/login?error=missing_user_data"
                )

            # 3. 验证用户所属组织
            user_tenant_key = user_data.get("data", {}).get("tenant_key")
            
            if not user_tenant_key:
                logger.error("No tenant key in user data")
                return RedirectResponse(
                    url=f"{FRONTEND_URL}/login?error=no_tenant_key"
                )

            # 4. 验证用户是否属于配置的组织
            if user_tenant_key != FEISHU_TENANT_KEY:
                logger.warning(f"Unauthorized tenant access attempt: {user_tenant_key}")
                return RedirectResponse(
                    url=f"{FRONTEND_URL}/login?error=unauthorized_org"
                )

            # 5. 添加过期时间
            expires_at = datetime.now() + timedelta(hours=TOKEN_EXPIRE_HOURS)
            logger.info(f"Token will expire at: {expires_at}")

            # 获取用户信息后，存储或更新用户记录
            feishu_user_id = user_data["data"]["user_id"]
            
            # 查找现有用户
            user = db.query(User).filter(User.feishu_user_id == feishu_user_id).first()
            
            # 准备用户数据
            user_info = {
                "feishu_user_id": feishu_user_id,
                "name": user_data["data"]["name"],
                "en_name": user_data["data"].get("en_name"),
                "email": user_data["data"].get("email"),
                "mobile": user_data["data"].get("mobile"),
                "avatar_url": user_data["data"].get("avatar_url"),
                "tenant_key": user_data["data"]["tenant_key"],
                "access_token": access_token,
                "token_expires_at": expires_at
            }

            if user:
                # 更新现有用户
                for key, value in user_info.items():
                    setattr(user, key, value)
                user.update_login(access_token, expires_at)
                logger.info(f"Updated user information for: {user.name}")
            else:
                # 创建新用户
                user = User(**user_info)
                user.last_login_at = datetime.utcnow()
                user.login_count = 1
                db.add(user)
                logger.info(f"Created new user: {user.name}")

            try:
                db.commit()
                logger.info("Database transaction committed successfully")
            except Exception as e:
                db.rollback()
                logger.error(f"Database transaction failed: {str(e)}")
                raise

            # 构建认证数据时添加用户ID
            auth_data = {
                "status": "success",
                "user_info": {
                    "id": str(user.id),  # UUID需要转换为字符串
                    "name": user.name,
                    "en_name": user.en_name,
                    "email": user.email,
                    "avatar_url": user.avatar_url,
                    "tenant_key": user.tenant_key,
                    "feishu_user_id": user.feishu_user_id
                },
                "access_token": access_token,
                "expires_at": expires_at.timestamp()
            }

            # 添加日志记录
            logger.info(f"Auth data for user {user.name}: {auth_data}")
            
            # 确保所有必要字段都存在
            if not all([
                auth_data.get("status"),
                auth_data.get("user_info"),
                auth_data.get("access_token"),
                auth_data.get("expires_at")
            ]):
                logger.error(f"Missing required fields in auth data: {auth_data}")
                return RedirectResponse(
                    url=f"{FRONTEND_URL}/login?error=incomplete_data"
                )

            # 7. 使用URL参数传递认证数据并重定向到前端
            auth_data_json = json.dumps(auth_data, ensure_ascii=False)
            # 使用 base64 编码避免特殊字符问题
            auth_data_encoded = base64.b64encode(auth_data_json.encode('utf-8')).decode('ascii')
            # 使用URL参数传递，避免Cookie跨域问题
            redirect_url = f"{FRONTEND_URL}/auth/callback?data={auth_data_encoded}"
            
            logger.info("Authentication successful, redirecting to frontend")
            logger.info(f"Encoded auth data length: {len(auth_data_encoded)}")
            logger.info(f"Encoded auth data preview: {auth_data_encoded[:100]}...")
            logger.info(f"Redirect URL: {redirect_url}")
            
            return RedirectResponse(url=redirect_url)

    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        logger.error(traceback.format_exc())
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error=auth_failed"
        )

# 刷新token
@router.get("/api/auth/refresh")
async def refresh_token(
    current_token: str = None,
    db: Session = Depends(get_db)
):
    try:
        # 优先从查询参数获取token，如果没有则从依赖注入获取
        if not current_token:
            # 这里可以添加从请求头获取token的逻辑
            # 但为了兼容性，暂时保持查询参数方式
            return JSONResponse(
                status_code=400,
                content={"error": "No token provided"}
            )
        
        # 查找用户
        user = db.query(User).filter(User.access_token == current_token).first()
        if not user:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid token"}
            )
        
        # 检查token是否过期
        if user.token_expires_at < datetime.utcnow():
            return JSONResponse(
                status_code=401,
                content={"error": "Token expired"}
            )
            
        # 生成新的过期时间
        new_expires_at = datetime.now() + timedelta(hours=TOKEN_EXPIRE_HOURS)
        
        # 更新用户token过期时间
        user.token_expires_at = new_expires_at
        db.commit()
        
        logger.info(f"Token refreshed for user: {user.name}, new expires_at: {new_expires_at}")
        
        return {
            "expires_at": new_expires_at.timestamp()
        }
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"error": "Token refresh failed"}
        )

# 验证当前用户用于查询数据库用户
async def get_current_user(
    db: Session = Depends(get_db),
    access_token: str = None
) -> User:
    """验证当前用户"""
    if not access_token:
        # 从请求头获取 token
        raise HTTPException(
            status_code=401,
            detail={"code": "NO_TOKEN", "message": "No access token provided"}
        )
    
    # 查找用户
    user = db.query(User).filter(User.access_token == access_token).first()
    if not user:
        raise HTTPException(
            status_code=401,
            detail={"code": "INVALID_TOKEN", "message": "Invalid access token"}
        )
    
    # 检查 token 是否过期
    if user.token_expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=401,
            detail={"code": "TOKEN_EXPIRED", "message": "Token has expired"}
        )
    
    return user