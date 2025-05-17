# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Body
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import asyncio
from dotenv import load_dotenv
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
from enum import Enum
import logging
import traceback
from datetime import datetime
from services.term_extractor import GeminiTermExtractor
from services.glossary_manager import GlossaryManager
from services.document_processor import DocumentProcessor
from services.document_chunker import DocumentChunker
import time
import json
from sqlalchemy.orm import Session
from fastapi import Depends
from database import get_db
from sqlalchemy.sql import text
from services.local_glossary_manager import LocalGlossaryManager
from auth.oauth import router as auth_router, get_current_user
from auth.user_router import router as user_router
from services.distance_calculator import DistanceCalculator
from io import BytesIO
from services.task_manager import TaskManager
import base64
import urllib.parse
from services.paper_analyzer import PaperAnalyzerService
import uuid
from fastapi import Header
from pydantic import BaseModel
from auth.models import ChatSession, ChatMessage, User, SessionDocument
from models.paper import PaperAnalysis  # 使用正确的类名
from feishu_approval.approval_controller import router as approval_router # 审批控制器 飞书审批控制器   

# 加载环境变量
load_dotenv()

# 修改日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('debug.log'),  # 直接在当前目录创建日志文件
    ]
)
logger = logging.getLogger(__name__)

# 定义翻译服务类型
class TranslatorType(str, Enum):
    DEEPL = "deepl"
    GOOGLE = "google"

# 翻译服务的抽象基类
class TranslatorService(ABC):
    @abstractmethod
    async def translate_document(self, file_content: bytes, filename: str, target_lang: str) -> bytes:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

# DeepL翻译服务实现
class DeepLTranslator(TranslatorService):
    def __init__(self):
        self.api_key = os.getenv("DEEPL_API_KEY")
        self.api_type = os.getenv("DEEPL_API_TYPE", "free")
        self.base_url = "https://api.deepl.com" if self.api_type.lower() == "pro" else "https://api-free.deepl.com"
        self.api_url = f"{self.base_url}/v2"
        
        # DeepL API 支持的语言代码映射
        self.lang_code_map = {
            'zh': 'ZH',    # 中文
            'en': 'EN',    # 英文
            'id': 'ID',    # 印尼文
        }

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _normalize_lang_code(self, lang_code: str) -> str:
        """标准化语言代码"""
        if not lang_code or lang_code.lower() == 'auto':
            raise ValueError("Source language must be specified when using glossaries")
            
        lang_code = lang_code.lower()
        if lang_code not in self.lang_code_map:
            raise ValueError(f"Unsupported language code: {lang_code}")
        return self.lang_code_map[lang_code]

    async def translate_document(self, file_content: bytes, filename: str, source_lang: str, target_lang: str, glossary_id: Optional[str] = None) -> dict:
        """翻译文档并返回结果"""
        if not self.api_key:
            raise ValueError("DeepL API key not configured")

        try:
            # 在使用术语表时验证源语言
            if glossary_id and (not source_lang or source_lang.lower() == 'auto'):
                raise ValueError("Source language must be specified when using glossaries")

            # 标准化语言代码
            normalized_source_lang = self._normalize_lang_code(source_lang)
            normalized_target_lang = self._normalize_lang_code(target_lang)

            # 正确的请求格式
            files = {
                'file': (filename, file_content)  # DeepL API 要求的格式
            }
            data = {
                'auth_key': self.api_key,
                'target_lang': normalized_target_lang
            }
            
            # 只有在指定时才添加 source_lang
            if source_lang and source_lang.lower() != 'auto':
                data['source_lang'] = normalized_source_lang
            
            # 只有在使用术语表时才添加 glossary_id
            if glossary_id:
                data['glossary_id'] = glossary_id

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/document",
                    files=files,
                    data=data,  # 使用 data 而不是 json
                    timeout=30.0
                )

                if response.status_code != 200:
                    error_msg = response.text
                    logger.error(f"DeepL API error: {error_msg}")
                    raise ValueError(f"DeepL API error: {error_msg}")

                return response.json()

        except Exception as e:
            logger.error(f"Document translation error: {str(e)}")
            raise

    async def translate_text(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> dict:
        """翻译文本"""
        if not self.api_key:
            raise ValueError("DeepL API key not configured")

        try:
            data = {
                "text": [text],
                "target_lang": self._normalize_lang_code(target_lang)
            }

            if source_lang:
                data["source_lang"] = self._normalize_lang_code(source_lang)

            headers = {
                "Authorization": f"DeepL-Auth-Key {self.api_key}",
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient() as client:
                # 使用 v2 endpoint
                response = await client.post(
                    f"{self.api_url}/translate",
                    json=data,
                    headers=headers
                )

                if response.status_code != 200:
                    error_msg = response.text
                    logger.error(f"DeepL API error: {error_msg}")
                    raise ValueError(f"DeepL API error: {error_msg}")

                return response.json()

        except Exception as e:
            logger.error(f"Text translation error: {str(e)}")
            raise

    async def check_document_status(self, document_id: str, document_key: str) -> dict:
        """检查文档翻译状态"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/document/{document_id}",
                    data={'document_key': document_key},  # 使用 data 而不是 json
                    headers={'Authorization': f"DeepL-Auth-Key {self.api_key}"}
                )
                
                if response.status_code != 200:
                    raise ValueError(f"Status check failed: {response.text}")
                    
                return response.json()
        except Exception as e:
            logger.error(f"Error checking document status: {str(e)}")
            raise

    async def get_document_result(self, document_id: str, document_key: str) -> bytes:
        """获取翻译结果"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/document/{document_id}/result",
                    data={'document_key': document_key},  # 使用 data 而不是 json
                    headers={'Authorization': f"DeepL-Auth-Key {self.api_key}"}
                )
                
                if response.status_code != 200:
                    raise ValueError(f"Download failed: {response.text}")
                    
                return response.content
        except Exception as e:
            logger.error(f"Error downloading document: {str(e)}")
            raise

# Google翻译服务实现
class GoogleTranslator(TranslatorService):
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
        self.project_id = os.getenv("GOOGLE_PROJECT_ID")
        try:
            from google.cloud import translate_v3
            from google.oauth2 import service_account
            self.translate_v3 = translate_v3
            self.service_account = service_account
            self._init_client()
        except ImportError:
            self.translate_v3 = None

    def is_available(self) -> bool:
        return bool(self.translate_v3 and self.api_key and self.project_id)

    def _init_client(self):
        if self.is_available():
            credentials = self.service_account.Credentials.from_service_account_info({
                "type": "service_account",
                "project_id": self.project_id,
                "private_key": self.api_key,
                # 其他必要的凭证信息...
            })
            self.client = self.translate_v3.TranslationServiceClient(credentials=credentials)

    async def translate_document(self, file_content: bytes, filename: str, target_lang: str) -> bytes:
        if not self.is_available():
            raise ValueError("Google Translate API not configured or package not installed")

        try:
            parent = f"projects/{self.project_id}/locations/global"
            
            # 构建请求
            request = self.translate_v3.TranslateTextRequest(
                parent=parent,
                contents=[file_content.decode('utf-8')],
                mime_type=self._get_mime_type(filename),
                source_language_code="auto",
                target_language_code=target_lang,
            )
            
            # 发送翻译请求
            response = self.client.translate_text(request)
            
            # 返回翻译结果
            if response.translations:
                return response.translations[0].translated_text.encode('utf-8')
            else:
                raise ValueError("No translation result")
                
        except Exception as e:
            raise Exception(f"Google Translate API error: {str(e)}")
    
    def _get_mime_type(self, filename: str) -> str:
        ext = filename.lower().split('.')[-1]
        mime_types = {
            'pdf': 'application/pdf',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'txt': 'text/plain'
        }
        return mime_types.get(ext, 'application/octet-stream')

# 翻译服务工厂
class TranslatorFactory:
    _instances: Dict[TranslatorType, TranslatorService] = {}

    @classmethod
    def get_translator(cls, translator_type: TranslatorType) -> TranslatorService:
        if translator_type not in cls._instances:
            if translator_type == TranslatorType.DEEPL:
                cls._instances[translator_type] = DeepLTranslator()
            elif translator_type == TranslatorType.GOOGLE:
                cls._instances[translator_type] = GoogleTranslator()
        
        translator = cls._instances[translator_type]
        if not translator.is_available():
            raise ValueError(f"{translator_type} translator is not available")
            
        return translator

app = FastAPI(title="CargoPPT Translation API")

# 添加 CORS 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://translation.jtcargo.co.id"],  # 添加服务器域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth_router)  # 先注册 auth 路由
app.include_router(user_router, prefix="/api")  # 再注册用户路由
app.include_router(approval_router) # 添加飞书审批路由
# 添加一个辅助函数来从当前请求中获取用户ID - 必须在使用前定义
async def get_current_user_id(
    authorization: Optional[str] = Header(None, description="Bearer token"),
    db: Session = Depends(get_db)
):
    """从请求头中提取用户ID"""
    if not authorization:
        return None
        
    try:
        # 提取 token
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail={"code": "INVALID_TOKEN_FORMAT", "message": "Token must be Bearer"}
            )
        
        token = authorization.replace("Bearer ", "")
        
        # 验证用户
        user = await get_current_user(db=db, access_token=token)
        return str(user.id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail={"code": "AUTHENTICATION_ERROR", "message": str(e)}
        )

@app.get("/api/translators")
def get_available_translators():
    """获取可用的翻译服务列表"""
    translators = {
        TranslatorType.DEEPL: DeepLTranslator(),
        TranslatorType.GOOGLE: GoogleTranslator()
    }
    
    return {
        "translators": [
            {
                "id": translator_type.value,
                "name": "DeepL" if translator_type == TranslatorType.DEEPL else "Google Translate",
                "available": translator.is_available()
            }
            for translator_type, translator in translators.items()
        ],
        "default": os.getenv("DEFAULT_TRANSLATOR", TranslatorType.DEEPL.value)
    }

# 添加文件大小限制常量
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes

# 设置超时时间
TIMEOUT = 480.0  # 480秒

@app.post("/api/translate")
async def translate_document(
    file: UploadFile = File(...),
    source_lang: str = Form(...),
    target_lang: str = Form(...),
    use_glossary: bool = Form(True),
    db: Session = Depends(get_db)
):
    try:
        # 1. 基础验证
        if use_glossary and (not source_lang or source_lang.lower() == 'auto'):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_SOURCE_LANGUAGE",
                    "message": "Source language must be specified when using glossaries"
                }
            )

        # 验证文件类型和大小
        if not file.filename.lower().endswith(('.pdf', '.docx', '.pptx')):
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_FILE_TYPE", "message": "Only PDF, DOCX, and PPTX files are supported"}
            )

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail={"code": "FILE_TOO_LARGE", "message": "File size exceeds limit"}
            )

        # 2. 提取文档文本
        doc_processor = DocumentProcessor()
        text_content = await doc_processor.process_file_async(content, file.filename)
        logger.info(f"Extracted text content length: {len(text_content)}")

        glossary_id = None
        if use_glossary and text_content:
            try:
                term_extractor = GeminiTermExtractor()
                glossary_manager = GlossaryManager(db)

                # 3. 提取新术语
                logger.info("Starting term extraction...")
                new_terms = await term_extractor.extract_terms(text_content, source_lang, target_lang)
                logger.info(f"Extracted {len(new_terms)} new terms")

                if new_terms:
                    try:
                        # 4. 获取或创建主术语表
                        existing_glossary = await glossary_manager.get_or_create_main_glossary(
                            source_lang, 
                            target_lang
                        )
                        
                        # 5. 更新术语表
                        result = await glossary_manager.update_main_glossary(
                            source_lang,
                            target_lang,
                            new_terms
                        )
                        glossary_id = result["glossary_id"]
                        logger.info(f"Updated glossary with ID: {glossary_id}")
                    
                    except Exception as e:
                        logger.error(f"Error updating glossary: {str(e)}")
                        logger.error(traceback.format_exc())
                        # 如果更新失败，尝试使用现有术语表
                        if existing_glossary:
                            glossary_id = existing_glossary.get("glossary_id")
                            logger.info(f"Falling back to existing glossary: {glossary_id}")
                else:
                    # 如果没有新术语，使用现有术语表
                    main_glossary = await glossary_manager.get_or_create_main_glossary(
                        source_lang,
                        target_lang
                    )
                    glossary_id = main_glossary["glossary_id"]
                    logger.info(f"Using existing glossary: {glossary_id}")

            except Exception as e:
                logger.error(f"Error in glossary management: {str(e)}")
                logger.error(traceback.format_exc())
                glossary_id = None

        # 6. 执行翻译
        translator = DeepLTranslator()
        try:
            result = await translator.translate_document(
                file_content=content,
                filename=file.filename,
                source_lang=source_lang,
                target_lang=target_lang,
                glossary_id=glossary_id
            )

            return {
                "document_id": result["document_id"],
                "document_key": result["document_key"],
                "glossary_id": glossary_id,
                "has_glossary": bool(glossary_id)
            }

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={"code": "TRANSLATION_ERROR", "message": str(e)}
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"code": "UNEXPECTED_ERROR", "message": "An unexpected error occurred"}
        )


# 修改状态检查端点以包含术语表信息
@app.post("/api/translate/{document_id}/status")
async def check_translation_status(document_id: str, document_key: str = Form(...)):
    try:
        translator = DeepLTranslator()
        status = await translator.check_document_status(document_id, document_key)
        return status
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": "STATUS_CHECK_ERROR", "message": str(e)}
        )

# 在文件开头添加自定义异常类
class CharacterLimitError(Exception):
    """DeepL API character limit reached exception"""
    pass

# 修改文档翻译下载端点
@app.post("/api/translate/{document_id}/result")
async def download_document(document_id: str, document_key: str = Form(...)):
    try:
        translator = DeepLTranslator()
        result = await translator.get_document_result(document_id, document_key)
        
        return Response(
            content=result,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=translated_document"
            }
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": "DOWNLOAD_ERROR", "message": str(e)}
        )

# 同样修改文本翻译端点
@app.post("/api/translate/text")
async def translate_text(
    text: str = Form(...),
    target_lang: str = Form(...),
):
    try:
        translator = DeepLTranslator()
        if not translator.api_key:
            raise HTTPException(status_code=500, detail="DeepL API key not configured")

        base_url = translator.api_url.replace("/document", "")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/translate",
                headers={
                    "Authorization": f"DeepL-Auth-Key {translator.api_key}"
                },
                json={
                    "text": [text],
                    "target_lang": target_lang
                }
            )
            
            if response.status_code != 200:
                error_data = response.json()
                error_message = error_data.get("message", "")
                
                # 检查是否是字符限制错误
                if "Character limit reached" in error_message:
                    logger.error("Translation character limit reached")
                    raise CharacterLimitError("Monthly character limit reached. Please contact administrator.")
                
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Translation failed: {response.text}"
                )
            
            return response.json()

    except CharacterLimitError as e:
        # 返回特定的错误代码和消息
        raise HTTPException(
            status_code=429,  # Too Many Requests
            detail={
                "code": "CHARACTER_LIMIT_REACHED",
                "message": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Text translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/health/db")
async def check_db_health(db: Session = Depends(get_db)):
    try:
        # 执行简单查询
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "message": "Database connection successful"}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "message": str(e)}
        )


@app.post("/api/create-glossary")
async def create_glossary(
    file: UploadFile = File(...),
    primary_lang: str = Form(...),
    name: str = Form("Auto Generated Glossary"),
    db: Session = Depends(get_db)
):
    try:
        # 1. 读取文件内容
        content = await file.read()
        
        # 2. 使用 DocumentProcessor 提取文本
        doc_processor = DocumentProcessor()
        text_content = await doc_processor.process_file_async(content, file.filename)
        
        if not text_content:
            raise HTTPException(
                status_code=400,
                detail={"code": "TEXT_EXTRACTION_ERROR", "message": "Failed to extract text from document"}
            )
        
        # 3. 生成术语表 payload
        term_extractor = GeminiTermExtractor()
        glossary_manager = GlossaryManager(db)
        glossary_payload = await term_extractor.create_glossary_payload(
            text_content, 
            primary_lang,
            name
        )
        
        # 4. 创建 DeepL 术语表
        result = await glossary_manager.create_glossary(glossary_payload)
        
        return {
            "status": "success",
            "glossary_id": result["glossary_id"],
            "dictionaries": result["dictionaries"]
        }
        
    except Exception as e:
        logger.error(f"Glossary creation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": "GLOSSARY_CREATION_ERROR", "message": str(e)}
        )

# 获取所有术语表 前端使用的api
@app.get("/api/glossaries")
async def list_glossaries(db: Session = Depends(get_db)):
    try:
        glossary_manager = GlossaryManager(db)
        glossaries = await glossary_manager.list_glossaries()
        return {"glossaries": glossaries}
    except Exception as e:
        logger.error(f"Error listing glossaries: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": "GLOSSARY_LIST_ERROR", "message": str(e)}
        )

# 获取特定术语表
@app.get("/api/glossaries/{glossary_id}")
async def get_glossary(glossary_id: str, db: Session = Depends(get_db)):
    try:
        glossary_manager = GlossaryManager(db)
        glossary = await glossary_manager.get_glossary(glossary_id)
        return glossary
    except Exception as e:
        logger.error(f"Error getting glossary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": "GLOSSARY_GET_ERROR", "message": str(e)}
        )

# 获取术语表条目
@app.get("/api/glossaries/{glossary_id}/entries")
async def get_glossary_entries(glossary_id: str, db: Session = Depends(get_db)):
    try:
        glossary_manager = GlossaryManager(db)
        entries = await glossary_manager.get_entries(glossary_id)
        return Response(content=entries, media_type="text/plain")
    except Exception as e:
        logger.error(f"Error getting glossary entries: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": "GLOSSARY_ENTRIES_ERROR", "message": str(e)}
        )

# 更新术语表
@app.patch("/api/glossaries/{glossary_id}")
async def update_glossary(glossary_id: str, payload: dict, db: Session = Depends(get_db)):
    try:
        glossary_manager = GlossaryManager(db)
        result = await glossary_manager.update_glossary(glossary_id, payload)
        return result
    except Exception as e:
        logger.error(f"Error updating glossary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": "GLOSSARY_UPDATE_ERROR", "message": str(e)}
        )

# 删除术语表
@app.delete("/api/glossaries/{glossary_id}")
async def delete_glossary(glossary_id: str, db: Session = Depends(get_db)):
    try:
        glossary_manager = GlossaryManager(db)
        await glossary_manager.delete_glossary(glossary_id)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error deleting glossary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": "GLOSSARY_DELETE_ERROR", "message": str(e)}
        )

# 搜索术语表和词汇明细本地数据库查询
@app.get("/api/glossaries-search")
async def search_glossaries(
    name: Optional[str] = Query(None, description="术语表名称"),
    start_date: Optional[str] = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="结束日期 (YYYY-MM-DD)"),
    source_lang: Optional[str] = Query(None, description="源语言"),
    target_lang: Optional[str] = Query(None, description="目标语言"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(10, ge=1, le=100, description="每页数量"),
    db: Session = Depends(get_db)
):
    try:
        # 处理日期参数
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "INVALID_DATE_FORMAT",
                        "message": "Start date should be in YYYY-MM-DD format"
                    }
                )
                
        if end_date:
            try:
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
                end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "INVALID_DATE_FORMAT",
                        "message": "End date should be in YYYY-MM-DD format"
                    }
                )

        # 使用 LocalGlossaryManager 进行本地数据库查询
        local_manager = LocalGlossaryManager(db)
        
        results = await local_manager.search_glossaries_and_entries(
            name=name,
            start_date=start_datetime,
            end_date=end_datetime,
            source_lang=source_lang,
            target_lang=target_lang,
            page=page,
            page_size=page_size
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching glossaries: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SEARCH_ERROR",
                "message": str(e)
            }
        )


# 获取术语表详细信息 前端专用
@app.get("/api/glossaries/{glossary_id}/details")
async def get_glossary_details(glossary_id: str, db: Session = Depends(get_db)):
    try:
        glossary_manager = GlossaryManager(db)
        
        # 首先检查术语表是否存在
        try:
            await glossary_manager.get_glossary(glossary_id)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "GLOSSARY_NOT_FOUND",
                    "message": f"Glossary with ID {glossary_id} not found"
                }
            )
        
        # 获取详细信息
        details = await glossary_manager.get_glossary_details(glossary_id)
        return details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting glossary details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "GLOSSARY_DETAILS_ERROR",
                "message": "Failed to retrieve glossary details"
            }
        )

# 更新术语表条目的目标术语本地数据库
@app.put("/api/glossary-entries/{entry_id}")
async def update_glossary_entry(
    entry_id: int,
    target_term: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    try:
        local_manager = LocalGlossaryManager(db)
        result = await local_manager.update_glossary_entry(entry_id, target_term)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "ENTRY_NOT_FOUND", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error updating glossary entry: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": "UPDATE_ERROR", "message": str(e)}
        )

@app.delete("/api/glossary-entries/{entry_id}")
async def delete_glossary_entry(
    entry_id: int,
    db: Session = Depends(get_db)
):
    try:
        local_manager = LocalGlossaryManager(db)
        await local_manager.delete_glossary_entry(entry_id)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={"code": "ENTRY_NOT_FOUND", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error deleting glossary entry: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": "DELETE_ERROR", "message": str(e)}
        )

# 创建任务管理器实例
task_manager = None

def get_task_manager(db: Session = Depends(get_db)):
    """获取任务管理器实例"""
    global task_manager
    if task_manager is None:
        task_manager = TaskManager(db)
    return task_manager

@app.post("/api/calculate-distance")
async def calculate_distance(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        if not file.filename.endswith('.xlsx'):
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_FILE_TYPE", "message": "Only Excel (.xlsx) files are supported"}
            )

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail={"code": "FILE_TOO_LARGE", "message": "File size exceeds limit"}
            )

        task_manager = get_task_manager(db)
        task_id = await task_manager.add_task(content, file.filename)
        
        return {
            "task_id": task_id,
            "status": "queued"
        }

    except Exception as e:
        logger.error(f"Error adding task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": "TASK_ERROR", "message": str(e)}
        )

@app.get("/api/tasks")
async def get_all_tasks(db: Session = Depends(get_db)):
    """获取所有任务"""
    task_manager = get_task_manager(db)
    return task_manager.get_all_tasks()

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """获取任务状态"""
    task_manager = get_task_manager(db)
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail={"code": "TASK_NOT_FOUND", "message": "Task not found"}
        )
    return task

@app.get("/api/tasks/{task_id}/download")
async def download_result(task_id: str, db: Session = Depends(get_db)):
    try:
        task_manager = get_task_manager(db)
        task = task_manager.get_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=404,
                detail={"code": "TASK_NOT_FOUND", "message": "Task not found"}
            )
            
        if task['status'] != 'completed':
            raise HTTPException(
                status_code=400,
                detail={"code": "RESULT_NOT_READY", "message": "Task is not completed yet"}
            )
            
        if not task.get('result_data'):
            raise HTTPException(
                status_code=404,
                detail={"code": "RESULT_NOT_FOUND", "message": "Result data not found"}
            )
        
        try:
            # 修改：使用 URL 安全的文件名处理
            result_data = base64.b64decode(task['result_data'])
            filename = task['result_filename'] or f"result_{task_id}.xlsx"
            
            # 确保文件名是 URL 安全的
            safe_filename = urllib.parse.quote(filename)
            
            return Response(
                content=result_data,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}",
                    "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                }
            )
        except Exception as e:
            logger.error(f"Error decoding result: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={"code": "DECODE_ERROR", "message": f"Failed to decode result: {str(e)}"}
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading result: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": "DOWNLOAD_ERROR", "message": str(e)}
        )

#使用google ai实现文本翻译
@app.post("/api/translate/multilingual")
async def translate_multilingual_text(text: str = Form(...)):
    try:
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "EMPTY_INPUT",
                    "message": "Input text cannot be empty"
                }
            )

        term_extractor = GeminiTermExtractor()
        result = await term_extractor.translate_text_with_language_detection(text)
        
        # 确保返回的结果格式正确
        return {
            "status": "success",
            "translations": {
                "detected_language": result.get("detected_language", "auto"),
                "english": result.get("english", ""),
                "chinese": result.get("chinese", ""),
                "indonesian": result.get("indonesian", "")
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multilingual translation error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "translations": {
                "detected_language": "auto",
                "english": "",
                "chinese": "",
                "indonesian": "",
                "error": str(e)
            }
        }

# 取消任务 经纬度计算前端专用 
@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, db: Session = Depends(get_db)):
    """取消任务"""
    try:
        task_manager = get_task_manager(db)
        try:
            success = task_manager.cancel_task(task_id)
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "code": "TASK_NOT_FOUND",
                        "message": "Task not found"
                    }
                )
            return {"status": "success", "message": "Task cancelled successfully"}
            
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_OPERATION",
                    "message": str(e)
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CANCEL_ERROR",
                "message": "Failed to cancel task"
            }
        )

@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str, db: Session = Depends(get_db)):
    """删除任务"""
    try:
        task_manager = get_task_manager(db)
        try:
            success = task_manager.delete_task(task_id)
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "code": "TASK_NOT_FOUND",
                        "message": "Task not found"
                    }
                )
            return {"status": "success", "message": "Task deleted successfully"}
            
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_OPERATION",
                    "message": str(e)
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DELETE_ERROR",
                "message": "Failed to delete task"
            }
        )

# 创建服务实例独立的阅读文档
paper_analyzer = None

def get_paper_analyzer(db: Session = Depends(get_db)):
    global paper_analyzer
    if paper_analyzer is None:
        paper_analyzer = PaperAnalyzerService(db)
    return paper_analyzer

@app.post("/api/paper/analyze")
async def analyze_paper(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),  # Add optional session_id parameter
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id)  # Make user_id optional
):
    try:
        # 先检查会话文档数量限制（如果提供了会话ID）
        if session_id and user_id:
            # 验证会话存在且属于该用户
            session = db.query(ChatSession).filter(
                ChatSession.id == uuid.UUID(session_id),
                ChatSession.user_id == uuid.UUID(user_id)
            ).first()
            
            if session:
                # 检查会话中的文档数量
                doc_count = db.query(SessionDocument).filter(
                    SessionDocument.session_id == uuid.UUID(session_id)
                ).count()
                
                # 如果已经达到最大限制，拒绝上传
                if doc_count >= 10:  # MAX_DOCUMENTS_PER_SESSION
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "code": "SESSION_DOCUMENT_LIMIT_REACHED",
                            "message": "一个会话最多支持10个文档，请创建新会话"
                        }
                    )

        # 验证文件类型
        allowed_extensions = ('.pdf', '.docx', '.doc', '.pptx', '.ppt', 
                            '.xlsx', '.xls', '.txt', '.md')
        if not file.filename.lower().endswith(allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_FILE_TYPE",
                    "message": "Only PDF, Word, PowerPoint, Excel, TXT, and MD files are supported"
                }
            )
        
        content = await file.read()
        analyzer = get_paper_analyzer(db)
        result = await analyzer.analyze_paper(content, file.filename)
        
        # 如果上传成功且提供了会话ID，自动添加到会话
        if session_id and user_id and result.get("status") == "success" and result.get("paper_id"):
            try:
                paper_id = result["paper_id"]
                await analyzer.add_document_to_session(session_id, paper_id, user_id)
                result["added_to_session"] = True
            except Exception as session_error:
                # 只记录错误，不影响文件分析返回
                print(f"Error adding document to session: {str(session_error)}")
                result["added_to_session"] = False
                result["session_error"] = str(session_error)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "PAPER_ANALYSIS_ERROR", "message": str(e)}
        )

@app.post("/api/paper/ask")
async def ask_paper_question(
    question: str = Body(...),
    paper_id: str = Body(...),
    db: Session = Depends(get_db)
):
    try:
        analyzer = get_paper_analyzer(db)
        result = await analyzer.ask_question(question, paper_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "QUESTION_ERROR", "message": str(e)}
        )

@app.get("/api/paper/{paper_id}/content")
async def get_paper_content(
    paper_id: str,
    db: Session = Depends(get_db)
):
    try:
        analyzer = get_paper_analyzer(db)
        content = await analyzer.get_paper_content(paper_id)
        return {"content": content}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "CONTENT_ERROR", "message": str(e)}
        )

@app.get("/api/paper/{paper_id}/questions")
async def get_question_history(
    paper_id: str,
    db: Session = Depends(get_db)
):
    try:
        analyzer = get_paper_analyzer(db)
        history = await analyzer.get_question_history(paper_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "HISTORY_ERROR", "message": str(e)}
        )

@app.post("/api/paper/{paper_id}/translate")
async def translate_paper(
    paper_id: str,
    target_lang: str = Body(...),  # 直接接收字符串
    db: Session = Depends(get_db)
):
    try:
        analyzer = PaperAnalyzerService(db)
        result = await analyzer.translate_paper(paper_id, target_lang)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "TRANSLATION_ERROR", "message": str(e)}
        )

@app.get("/api/paper/supported-languages")
async def get_supported_languages(db: Session = Depends(get_db)):
    try:
        analyzer = PaperAnalyzerService(db)
        languages = await analyzer.get_supported_languages()
        return {"languages": languages}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "LANGUAGES_ERROR", "message": str(e)}
        )

@app.post("/api/paper/{paper_id}/download")
async def download_translation(
    paper_id: str,
    target_lang: str = Body(...),
    format: str = Body(...),
    db: Session = Depends(get_db)
):
    try:
        analyzer = get_paper_analyzer(db)
        content = await analyzer.download_translation(paper_id, target_lang, format)
        
        # 设置正确的 Content-Type
        content_type = {
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'pdf': 'application/pdf',
            'md': 'text/markdown'
        }.get(format, 'application/octet-stream')
        
        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=translated.{format}"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "DOWNLOAD_ERROR", "message": str(e)}
        )

# Define request body model
class MessageRequest(BaseModel):
    message: str
    
# Update the send_chat_message endpoint
@app.post("/api/chat-sessions/{session_id}/messages")
async def send_message_to_session(
    session_id: str,
    request: dict,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    try:
        print(f"API: Message request received for session {session_id}, user {authorization}")
        
        # 从authorization header中提取用户ID
        user_id = None
        if authorization:
            if authorization.startswith("Bearer "):
                token = authorization.replace("Bearer ", "")
                # 获取用户信息
                user = await get_current_user(db=db, access_token=token)
                if user:
                    user_id = str(user.id)
        
        if not user_id:
            print(f"API: Authentication failed for session {session_id}")
            raise HTTPException(
                status_code=401,
                detail={"code": "AUTHENTICATION_ERROR", "message": "Invalid authentication"}
            )
        
        # 验证用户是否有权限访问此会话
        session = db.query(ChatSession).filter(
            ChatSession.id == uuid.UUID(session_id),
            ChatSession.user_id == uuid.UUID(user_id)
        ).first()
        
        if not session:
            print(f"API: Session {session_id} not found for user {user_id}")
            raise HTTPException(
                status_code=404,
                detail={"code": "SESSION_NOT_FOUND", "message": "Chat session not found"}
            )
            
        analyzer = get_paper_analyzer(db)
        
        # 发送消息
        result = await analyzer.send_message(
            session_id=session_id,
            message=request.get("message"),
            user_id=user_id  # 使用提取出来的user_id而不是raw authorization
        )
        
        # 提取并记录消息状态信息
        message_status = result.pop("_message_status", {"stored": True})
        
        # 如果存储失败，记录到系统日志
        if not message_status.get("stored", True):
            print(f"WARNING: Message storage incomplete for session {session_id}, user {user_id}")
            # 可选：添加到专门的错误跟踪表
            try:
                from models.system import StorageFailure
                failure = StorageFailure(
                    session_id=uuid.UUID(session_id),
                    user_id=uuid.UUID(user_id),
                    message_content=request.get("message")[:1000],  # 保存一部分消息内容
                    user_message_id=message_status.get("user_message_id"),
                    ai_message_id=message_status.get("ai_message_id"),
                    failed_at=datetime.utcnow()
                )
                db.add(failure)
                db.commit()
            except Exception as log_error:
                print(f"Failed to log storage failure: {str(log_error)}")
        
        # 返回结果
        return result
        
    except Exception as e:
        print(f"API Error in send_message_to_session: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail={"code": "SERVER_ERROR", "message": str(e)}
        )

class SessionRequest(BaseModel):
    title: str = None
    paper_ids: List[str] = None  # 支持多文档
    is_ai_only: bool = False  # 新增无文档AI聊天支持

class DocumentRequest(BaseModel):
    paper_id: str

# 1. 会话管理 API

# 获取所有会话
@app.get("/api/chat-sessions")
async def get_all_chat_sessions(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    try:
        # 获取用户的所有活跃会话
        sessions = db.query(ChatSession).filter(
            ChatSession.user_id == uuid.UUID(user_id),
            ChatSession.is_active == True
        ).order_by(ChatSession.updated_at.desc()).all()
        
        result = []
        for session in sessions:
            # 获取消息数量
            message_count = db.query(ChatMessage).filter(
                ChatMessage.session_id == session.id
            ).count()
            
            # 获取最后一条消息
            last_message = db.query(ChatMessage).filter(
                ChatMessage.session_id == session.id
            ).order_by(ChatMessage.created_at.desc()).first()
            
            # 获取关联的文档
            session_docs = db.query(SessionDocument).filter(
                SessionDocument.session_id == session.id
            ).all()
            
            documents = []
            for doc in session_docs:
                documents.append({
                    "id": str(doc.id),
                    "paper_id": str(doc.paper_id),
                    "filename": doc.filename
                })
            
            result.append({
                "id": str(session.id),
                "title": session.title,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "session_type": session.session_type,
                "is_ai_only": session.is_ai_only,
                "message_count": message_count,
                "last_message": last_message.content if last_message else "",
                "documents": documents
            })
        
        return result
    except Exception as e:
        import traceback
        print(f"Sessions fetch error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"code": "SESSIONS_FETCH_ERROR", "message": str(e)}
        )

# 创建新会话
@app.post("/api/chat-sessions")
async def create_chat_session(
    request: SessionRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    try:
        # 获取分析器实例
        analyzer = get_paper_analyzer(db)
        
        # 支持AI-only对话模式
        if request.is_ai_only:
            # 创建不绑定文档的AI聊天会话
            result = await analyzer.create_ai_chat_session(
                user_id=user_id,
                title=request.title or "AI 对话"
            )
        elif request.paper_ids and len(request.paper_ids) > 0:
            # 创建绑定多个文档的会话
            # 直接使用 create_chat_session 方法，不要调用不存在的方法
            result = await analyzer.create_chat_session(
                user_id=user_id,
                title=request.title,
                paper_ids=request.paper_ids,
                session_type="document"
            )
        else:
            # 错误检查
            raise HTTPException(
                status_code=400, 
                detail={
                    "code": "INVALID_REQUEST", 
                    "message": "必须指定至少一个文档ID，或启用AI-only模式"
                }
            )
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "SESSION_CREATION_ERROR", "message": str(e)}
        )

# 获取会话详情
@app.get("/api/chat-sessions/{session_id}")
async def get_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    try:
        # 验证用户是否有权限访问此会话
        session = db.query(ChatSession).filter(
            ChatSession.id == uuid.UUID(session_id),
            ChatSession.user_id == uuid.UUID(user_id)
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail={"code": "SESSION_NOT_FOUND", "message": "Chat session not found"}
            )
            
        # 获取消息计数
        message_count = db.query(ChatMessage).filter(
            ChatMessage.session_id == session.id
        ).count()
        
        # 获取最后一条消息
        last_message = db.query(ChatMessage).filter(
            ChatMessage.session_id == session.id
        ).order_by(ChatMessage.created_at.desc()).first()
        
        # 获取关联的文档
        paper_ids = session.paper_ids or []
        
        return {
            "id": str(session.id),
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "paper_ids": paper_ids,
            "is_ai_only": session.is_ai_only,
            "message_count": message_count,
            "last_message": last_message.content if last_message else ""
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "SESSION_FETCH_ERROR", "message": str(e)}
        )

# 2. 消息管理API

# 获取会话消息
@app.get("/api/chat-sessions/{session_id}/messages")
async def get_chat_messages(
    session_id: str,
    limit: int = Query(20, description="最大消息数量, 默认20"),
    before_id: Optional[str] = Query(None, description="获取此ID之前的消息"),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    try:
        # 验证会话是否存在且属于该用户
        try:
            session = db.query(ChatSession).filter(
                ChatSession.id == uuid.UUID(session_id),
                ChatSession.user_id == uuid.UUID(user_id)
            ).first()
            
            if not session:
                return {
                    "messages": [],
                    "has_more": False
                }
        except Exception as e:
            print(f"Error verifying session: {str(e)}")
            return {
                "messages": [],
                "has_more": False
            }
            
        # 获取会话消息
        analyzer = get_paper_analyzer(db)
        try:
            messages = await analyzer.get_chat_history(session_id, limit=limit, before_id=before_id)
            
            # 确保响应包含所需的字段
            if isinstance(messages, dict) and "messages" in messages:
                return messages
            else:
                # 如果返回的不是预期的格式，进行转换
                return {
                    "messages": messages if isinstance(messages, list) else [],
                    "has_more": False
                }
                
        except Exception as e:
            print(f"Error in analyzer.get_chat_history: {str(e)}")
            return {
                "messages": [],
                "has_more": False
            }
            
    except Exception as e:
        print(f"Unhandled error in get_chat_messages: {str(e)}")
        # 即使有错误也返回空消息列表，避免500错误
        return {
            "messages": [],
            "has_more": False
        }


# 3. 文档管理API

# 添加文档到会话
@app.post("/api/chat-sessions/{session_id}/documents")
async def add_document_to_session(
    session_id: str,
    request: DocumentRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    try:
        # 验证用户是否有权限访问此会话
        session = db.query(ChatSession).filter(
            ChatSession.id == uuid.UUID(session_id),
            ChatSession.user_id == uuid.UUID(user_id)
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail={"code": "SESSION_NOT_FOUND", "message": "Chat session not found"}
            )
        
        # 检查会话中的文档数量是否超出限制
        doc_count = db.query(SessionDocument).filter(
            SessionDocument.session_id == uuid.UUID(session_id)
        ).count()
        
        if doc_count >= 10:  # MAX_DOCUMENTS_PER_SESSION
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "SESSION_DOCUMENT_LIMIT_REACHED", 
                    "message": "一个会话最多支持10个文档，请创建新会话"
                }
            )
        
        # 获取分析器实例
        analyzer = get_paper_analyzer(db)
        
        # 添加文档到会话
        result = await analyzer.add_document_to_session(
            session_id=session_id,
            paper_id=request.paper_id,
            user_id=user_id
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "DOCUMENT_ADD_ERROR", "message": str(e)}
        )

# 获取会话的文档
@app.get("/api/chat-sessions/{session_id}/documents")
async def get_session_documents(
    session_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    try:
        # 验证用户是否有权限访问此会话
        session = db.query(ChatSession).filter(
            ChatSession.id == uuid.UUID(session_id),
            ChatSession.user_id == uuid.UUID(user_id)
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail={"code": "SESSION_NOT_FOUND", "message": "Chat session not found"}
            )
        
        # 获取分析器实例
        analyzer = get_paper_analyzer(db)
        
        # 使用PaperAnalyzerService中的get_session_documents方法获取文档
        documents = await analyzer.get_session_documents(session_id, user_id)
        
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "DOCUMENTS_FETCH_ERROR", "message": str(e)}
        )

# 移除会话中的文档
@app.delete("/api/chat-sessions/{session_id}/documents/{document_id}")
async def remove_document_from_session(
    session_id: str,
    document_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    try:
        # 验证用户是否有权限访问此会话
        session = db.query(ChatSession).filter(
            ChatSession.id == uuid.UUID(session_id),
            ChatSession.user_id == uuid.UUID(user_id)
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail={"code": "SESSION_NOT_FOUND", "message": "Chat session not found"}
            )
        
        # 获取分析器实例
        analyzer = get_paper_analyzer(db)
        
        # 从会话中移除文档
        result = await analyzer.remove_document_from_session(
            session_id=session_id,
            paper_id=document_id,
            user_id=user_id
        )
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "DOCUMENT_REMOVE_ERROR", "message": str(e)}
        )

# 4. 保留旧版API路径的兼容支持
@app.post("/api/paper/{paper_id}/chat-sessions")
async def create_chat_session_legacy(
    paper_id: str,
    request: SessionRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    # 使用新的请求格式
    new_request = SessionRequest(
        title=request.title,
        paper_ids=[paper_id],
        is_ai_only=False
    )
    # 调用新API
    return await create_chat_session(new_request, db, user_id)

@app.get("/api/paper/{paper_id}/chat-sessions")
async def get_chat_sessions_legacy(
    paper_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    try:
        # 获取用户在特定文档下的所有会话
        sessions = db.query(ChatSession).filter(
            ChatSession.user_id == uuid.UUID(user_id),
            ChatSession.is_active == True,
            ChatSession.paper_ids.contains([paper_id])
        ).order_by(ChatSession.updated_at.desc()).all()
        
        result = []
        for session in sessions:
            # 获取消息计数
            message_count = db.query(ChatMessage).filter(
                ChatMessage.session_id == session.id
            ).count()
            
            # 获取最后一条消息
            last_message = db.query(ChatMessage).filter(
                ChatMessage.session_id == session.id
            ).order_by(ChatMessage.created_at.desc()).first()
            
            result.append({
                "id": str(session.id),
                "title": session.title,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "paper_id": paper_id,  # 兼容旧版API
                "message_count": message_count,
                "last_message": last_message.content if last_message else ""
            })
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "SESSIONS_FETCH_ERROR", "message": str(e)}
        )

# 请求体模型定义
class UpdateSessionTitleRequest(BaseModel):
    title: str

# 删除会话API
@app.delete("/api/chat-sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    try:
        # 获取分析器实例
        analyzer = get_paper_analyzer(db)
        
        # 调用服务方法
        result = await analyzer.delete_chat_session(
            session_id=session_id,
            user_id=user_id
        )
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "SESSION_DELETE_ERROR", "message": str(e)}
        )

# 更新会话标题API
@app.patch("/api/chat-sessions/{session_id}/title")
async def update_chat_session_title(
    session_id: str,
    request: UpdateSessionTitleRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    try:
        # 验证标题不能为空
        if not request.title or not request.title.strip():
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_TITLE", "message": "会话标题不能为空"}
            )
            
        # 获取分析器实例
        analyzer = get_paper_analyzer(db)
        
        # 调用服务方法
        result = await analyzer.update_chat_session_title(
            session_id=session_id,
            user_id=user_id,
            new_title=request.title.strip()
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "SESSION_UPDATE_ERROR", "message": str(e)}
        )

@app.post("/api/chat-sessions/{session_id}/stream")
async def stream_message_to_session(
    session_id: str,
    request: dict,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    try:
        # 从authorization header中提取用户ID
        user_id = None
        if authorization:
            if authorization.startswith("Bearer "):
                token = authorization.replace("Bearer ", "")
                user = await get_current_user(db=db, access_token=token)
                if user:
                    user_id = str(user.id)
        
        if not user_id:
            return StreamingResponse(
                content=iter([json.dumps({"error": "认证失败", "done": True})]),
                media_type="application/json"
            )
        
        # 验证会话存在
        session = db.query(ChatSession).filter(
            ChatSession.id == uuid.UUID(session_id),
            ChatSession.user_id == uuid.UUID(user_id)
        ).first()
        
        if not session:
            return StreamingResponse(
                content=iter([json.dumps({"error": "会话不存在", "done": True})]),
                media_type="application/json"
            )
        
        # 获取分析器实例并调用流式消息方法
        analyzer = get_paper_analyzer(db)
        
        async def response_stream():
            async for chunk in analyzer.stream_message(
                session_id=session_id,
                message=request.get("message"),
                user_id=user_id
            ):
                yield json.dumps(chunk) + "\n"
        
        return StreamingResponse(
            content=response_stream(),
            media_type="application/json"
        )
        
    except Exception as e:
        print(f"Stream API error: {str(e)}")
        return StreamingResponse(
            content=iter([json.dumps({"error": str(e), "done": True})]),
            media_type="application/json"
        )

