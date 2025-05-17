# backend/services/paper_analyzer/__init__.py
from .ai_manager import AIManager
from .paper_processor import PaperProcessor
from .rag_retriever import RAGRetriever
from .translator import Translator
from sqlalchemy.orm import Session
from models.paper import PaperAnalysis, PaperQuestion
import uuid
from sqlalchemy import func
from docx import Document
from fpdf import FPDF
import markdown
from io import BytesIO
import tempfile
import os
from datetime import datetime
from typing import List, Dict, Optional, Union
from auth.models import ChatSession, ChatMessage, SessionDocument
from models.chat import ChatFile, ChatTopic, ChatTopicSession
from contextlib import contextmanager
import time
import json
from celery import Celery
import logging
import asyncio

logger = logging.getLogger(__name__)

class PaperAnalyzerService:
    def __init__(self, db: Session):
        self.db = db
        self.ai_manager = AIManager()
        self.paper_processor = PaperProcessor()
        self.rag_retriever = RAGRetriever(db)
        self.translator = Translator()
        # 最大文档数限制
        self.MAX_DOCUMENTS_PER_SESSION = 10

    async def analyze_paper(self, file_content: bytes, filename: str):
        """分析文档文件"""
        try:
            # 1. 处理文档内容
            processed_result = await self.paper_processor.process(file_content, filename)
            
            # 2. 生成唯一的paper_id
            paper_id = str(uuid.uuid4())
            
            # 3. 先保存到数据库，确保记录存在
            paper_analysis = PaperAnalysis(
                paper_id=uuid.UUID(paper_id),
                filename=filename,
                content=processed_result["content"],
                line_mapping=processed_result["line_mapping"],
                total_lines=processed_result["total_lines"],
                is_scanned=any(line.get("is_scanned", False) 
                              for line in processed_result["line_mapping"].values()),
                # 添加结构化数据
                structured_data=processed_result.get("structured_data"),
                # 确保这些字段初始化为空
                documents=None,
                embeddings=None,
                index_built=False
            )
            self.db.add(paper_analysis)
            self.db.commit()
            
            # 4. 构建向量索引
            try:
                await self.rag_retriever.build_index(
                    processed_result["content"], 
                    paper_id,
                    processed_result["line_mapping"],
                    self.db,  # 显式传递db连接
                    processed_result.get("structured_data")  # 传递结构化数据
                )
                # 确认索引构建完成
                paper_analysis.index_built = True
                self.db.commit()
            except Exception as e:
                self.db.rollback()
                print(f"Index building error: {str(e)}")
                # 不中断流程，允许用户先查看文档内容
            
            # 5. 返回完整的数据
            return {
                "status": "success",
                "message": "文档分析完成",
                "paper_id": paper_id,
                "content": processed_result["content"],
                "line_mapping": processed_result["line_mapping"] or {},
                "total_lines": processed_result["total_lines"],
                "is_scanned": paper_analysis.is_scanned,
                "has_structured_data": "structured_data" in processed_result  # 指示是否包含结构化数据
            }
        except Exception as e:
            self.db.rollback()
            print(f"Paper analysis error: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def get_paper_content(self, paper_id: str) -> str:
        """获取文档内容"""
        try:
            paper = self.db.query(PaperAnalysis).filter(
                PaperAnalysis.paper_id == uuid.UUID(paper_id)
            ).first()
            
            if not paper:
                raise Exception("Paper not found")
                
            return paper.content
        except Exception as e:
            raise Exception(f"Failed to get paper content: {str(e)}")

    async def ask_question(self, question: str, paper_id: str = None, context: str = None, session_id: str = None, user_id: str = None):
        """向文档提问，支持会话模式和单次问答模式"""
        try:
            # 如果提供了会话ID，直接使用send_message处理
            if session_id and user_id:
                return await self.send_message(
                    session_id=session_id,
                    message=question,
                    user_id=user_id
                )
            
            # 否则，使用传统的问答模式
            
            # 验证paper_id
            if paper_id == "current_paper_id" or not paper_id:
                latest_paper = self.db.query(PaperAnalysis).order_by(
                    PaperAnalysis.created_at.desc()
                ).first()
                if not latest_paper:
                    return {
                        "status": "error",
                        "message": "没有找到可用的文档"
                    }
                paper_id = str(latest_paper.paper_id)

            # 获取相关上下文
            try:
                # 检查数据库中是否存在这个paper_id
                paper = self.db.query(PaperAnalysis).filter(
                    PaperAnalysis.paper_id == uuid.UUID(paper_id)
                ).first()
                
                if not paper:
                    return {
                        "status": "error",
                        "message": f"Paper ID {paper_id} not found in database"
                    }
                    
                retrieval_context = await self.rag_retriever.get_relevant_context(question, paper_id, history=context)
            except Exception as e:
                import traceback
                print(f"Context retrieval error for paper {paper_id}: {str(e)}")
                print(traceback.format_exc())
                return {
                    "status": "error",
                    "message": f"获取上下文失败: {str(e)}"
                }
            
            # 获取 AI 回答，传入上下文
            response = await self.ai_manager.get_response(question, retrieval_context, history=context)
            
            # 单次问答模式 - 作为兼容旧版API保留，但同时创建一个会话记录
            # 存储到PaperQuestion表 (旧模式)
            paper_question = PaperQuestion(
                question=question,
                answer=response["answer"],
                paper_ids=[uuid.UUID(paper_id)]
            )
            self.db.add(paper_question)
            
            # 如果提供了用户ID，同时创建会话记录 (新模式)
            if user_id:
                # 创建一个临时会话
                temp_session = ChatSession(
                    paper_id=uuid.UUID(paper_id),
                    user_id=uuid.UUID(user_id),
                    title=f"单次问答: {question[:30]}..." if len(question) > 30 else question,
                    session_type="document"
                )
                self.db.add(temp_session)
                self.db.flush()  # 获取生成的ID
                
                # 创建文档关联
                session_doc = SessionDocument(
                    session_id=temp_session.id,
                    paper_id=uuid.UUID(paper_id),
                    order=0,
                    filename=paper.filename
                )
                self.db.add(session_doc)
                
                # 添加用户消息和AI回复
                user_message = ChatMessage(
                    session_id=temp_session.id,
                    role='user',
                    content=question,
                    message_type='markdown'
                )
                self.db.add(user_message)
                
                assistant_message = ChatMessage(
                    session_id=temp_session.id,
                    role='assistant',
                    content=response["answer"],
                    sources=response["sources"],
                    confidence=response["confidence"],
                    message_type='markdown'
                )
                self.db.add(assistant_message)
            
            self.db.commit()
            
            # 确保返回的数据都是可序列化的
            return {
                "status": "success",
                "response": {
                    "answer": str(response["answer"]),
                    "sources": [
                        {
                            "line_number": int(source["line_number"]),
                            "content": str(source["content"]),
                            "page": int(source["page"]),
                            "start_pos": int(source["start_pos"]),
                            "end_pos": int(source["end_pos"]),
                            "is_scanned": bool(source["is_scanned"]),
                            "similarity": float(source["similarity"])
                        }
                        for source in response["sources"]
                    ],
                    "confidence": float(response["confidence"])
                }
            }
        except Exception as e:
            self.db.rollback()
            return {
                "status": "error", 
                "message": str(e)
            }

    async def get_question_history(self, paper_id: str):
        """获取问答历史"""
        try:
            questions = self.db.query(PaperQuestion).filter(
                PaperQuestion.paper_id == uuid.UUID(paper_id)
            ).order_by(PaperQuestion.created_at.desc()).all()
            
            return [{
                "question": q.question,
                "answer": q.answer,
                "created_at": q.created_at.isoformat()
            } for q in questions]
        except Exception as e:
            raise Exception(f"Failed to get question history: {str(e)}")

    async def translate_paper(self, paper_id: str, target_lang: str):
        """翻译文档内容"""
        try:
            # 获取文档内容
            paper = self.db.query(PaperAnalysis).filter(
                PaperAnalysis.paper_id == uuid.UUID(paper_id)
            ).first()
            
            if not paper:
                raise Exception("Paper not found")
            
            # 验证目标语言
            if target_lang not in self.translator.supported_languages:
                raise ValueError(f"Unsupported target language: {target_lang}")
            
            # 翻译内容
            translated_content = await self.translator.translate_text(
                paper.content,
                target_lang
            )
            
            # 更新数据库中的翻译内容
            paper.translated_content = translated_content
            paper.translation_lang = target_lang
            paper.translation_updated_at = func.now()
            self.db.commit()
            
            return {
                "status": "success",
                "content": translated_content,
                "language": target_lang
            }
        except Exception as e:
            self.db.rollback()
            return {
                "status": "error",
                "message": str(e)
            }

    async def get_supported_languages(self):
        """获取支持的语言列表"""
        return self.translator.get_supported_languages()

    async def download_translation(self, paper_id: str, target_lang: str, format: str) -> bytes:
        """下载翻译结果"""
        try:
            # 获取文档内容
            paper = self.db.query(PaperAnalysis).filter(
                PaperAnalysis.paper_id == uuid.UUID(paper_id)
            ).first()
            
            if not paper:
                raise Exception("Paper not found")
            
            # 获取翻译内容
            translated_content = await self.translator.translate_text(
                paper.content,
                target_lang
            )
            
            # 根据格式转换内容
            if format == 'docx':
                return await self._convert_to_docx(translated_content)
            elif format == 'pdf':
                return await self._convert_to_pdf(translated_content)
            elif format == 'md':
                return translated_content.encode('utf-8')
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            raise Exception(f"Download error: {str(e)}")

    async def _convert_to_docx(self, content: str) -> bytes:
        """转换为 Word 文档"""
        try:
            doc = Document()
            for paragraph in content.split('\n'):
                if paragraph.strip():
                    doc.add_paragraph(paragraph)
            
            # 保存到内存
            output = BytesIO()
            doc.save(output)
            output.seek(0)
            return output.getvalue()
        except Exception as e:
            raise Exception(f"Failed to convert to DOCX: {str(e)}")

    async def _convert_to_pdf(self, content: str) -> bytes:
        """转换为 PDF 文档"""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # 处理内容
            for line in content.split('\n'):
                if line.strip():
                    pdf.multi_cell(0, 10, txt=line)
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_path = temp_file.name
            
            # 保存到临时文件
            pdf.output(temp_path)
            
            # 读取文件内容
            with open(temp_path, 'rb') as f:
                content = f.read()
            
            # 删除临时文件
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return content
            
        except Exception as e:
            # 确保清理临时文件
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise Exception(f"Failed to convert to PDF: {str(e)}")

    async def create_chat_session(self, user_id: str, title: str = None, paper_ids: List[str] = None, session_type: str = "document") -> dict:
        """创建新的对话会话，支持无文档和多文档模式"""
        try:
            # Debug logs
            print(f"Creating session: user_id={user_id}, title={title}, paper_ids={paper_ids}, type={session_type}")
            
            if not title:
                if session_type == "document" and paper_ids and len(paper_ids) > 0:
                    # 获取第一个文档的文件名作为标题
                    paper = self.db.query(PaperAnalysis).filter(
                        PaperAnalysis.paper_id == uuid.UUID(paper_ids[0])
                    ).first()
                    title = f"关于 {paper.filename if paper else '文档'} 的对话"
                else:
                    title = f"新对话 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            
            # 创建会话
            session = ChatSession(
                user_id=uuid.UUID(user_id),
                title=title,
                session_type=session_type,
                # 如果是单文档模式，保持向后兼容，为paper_id赋值
                paper_id=uuid.UUID(paper_ids[0]) if session_type == "document" and paper_ids and len(paper_ids) > 0 else None
            )
            self.db.add(session)
            self.db.flush()  # 获取生成的ID
            
            # 如果有文档IDs，创建关联关系
            if paper_ids and len(paper_ids) > 0:
                # 检查文档数量是否超出限制
                if len(paper_ids) > self.MAX_DOCUMENTS_PER_SESSION:
                    raise ValueError(f"一个会话最多支持{self.MAX_DOCUMENTS_PER_SESSION}个文档")
                
                # 关联文档
                for i, paper_id in enumerate(paper_ids):
                    # 获取文档信息
                    paper = self.db.query(PaperAnalysis).filter(
                        PaperAnalysis.paper_id == uuid.UUID(paper_id)
                    ).first()
                    
                    if not paper:
                        raise ValueError(f"文档ID {paper_id} 不存在")
                    
                    # 创建关联
                    session_doc = SessionDocument(
                        session_id=session.id,
                        paper_id=uuid.UUID(paper_id),
                        order=i,
                        filename=paper.filename
                    )
                    self.db.add(session_doc)
            
            self.db.commit()
            
            # 返回会话信息
            documents = []
            if paper_ids and len(paper_ids) > 0:
                # 获取关联的文档信息
                session_docs = self.db.query(SessionDocument).filter(
                    SessionDocument.session_id == session.id
                ).order_by(SessionDocument.order).all()
                
                documents = [{
                    "id": str(doc.id),
                    "paper_id": str(doc.paper_id),
                    "filename": doc.filename,
                    "order": doc.order
                } for doc in session_docs]
            
            return {
                "id": str(session.id),
                "title": session.title,
                "created_at": session.created_at.isoformat() if session.created_at else datetime.utcnow().isoformat(),
                "updated_at": session.updated_at.isoformat() if session.updated_at else datetime.utcnow().isoformat(),
                "paper_id": str(session.paper_id) if session.paper_id else None,  # 向后兼容
                "session_type": session.session_type,
                "message_count": 0,
                "last_message": "",
                "documents": documents
            }
        except Exception as e:
            self.db.rollback()
            print(f"Error creating chat session: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise Exception(f"Failed to create chat session: {str(e)}")

    async def get_user_chat_sessions(self, user_id: str, paper_id: str = None) -> List[dict]:
        """获取用户的所有对话会话"""
        try:
            # 基础查询
            query = self.db.query(ChatSession).filter(
                ChatSession.user_id == uuid.UUID(user_id),
                ChatSession.is_active == True
            )
            
            # 如果指定了paper_id，只获取包含该文档的会话
            if paper_id:
                # 方法1：使用传统的paper_id字段（向后兼容）
                # 方法2：通过SessionDocument关联查询
                query = query.filter(
                    (ChatSession.paper_id == uuid.UUID(paper_id)) | 
                    (ChatSession.id.in_(
                        self.db.query(SessionDocument.session_id).filter(
                            SessionDocument.paper_id == uuid.UUID(paper_id)
                        )
                    ))
                )
            
            # 按最后更新时间排序
            sessions = query.order_by(ChatSession.updated_at.desc()).all()
            
            # 构建响应
            result = []
            for session in sessions:
                # 获取最后一条消息
                last_message = self.db.query(ChatMessage).filter(
                    ChatMessage.session_id == session.id
                ).order_by(ChatMessage.created_at.desc()).first()
                
                # 获取关联的文档信息
                documents = []
                if session.session_type == "document":
                    session_docs = self.db.query(SessionDocument).filter(
                        SessionDocument.session_id == session.id
                    ).order_by(SessionDocument.order).all()
                    
                    documents = [{
                        "id": str(doc.id),
                        "paper_id": str(doc.paper_id),
                        "filename": doc.filename,
                        "order": doc.order
                    } for doc in session_docs]
                
                # 消息数量
                message_count = self.db.query(ChatMessage).filter(
                    ChatMessage.session_id == session.id
                ).count()
                
                result.append({
                    "id": str(session.id),
                    "title": session.title,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "paper_id": str(session.paper_id) if session.paper_id else None,  # 向后兼容
                    "session_type": session.session_type,
                    "message_count": message_count,
                    "last_message": last_message.content if last_message else "",
                    "documents": documents
                })
            
            return result
        except Exception as e:
            raise Exception(f"Failed to get chat sessions: {str(e)}")

    async def get_chat_history(self, session_id: str, limit: int = 20, before_id: Optional[str] = None) -> dict:
        """获取会话历史消息"""
        try:
            # 简化错误处理，直接使用try/except包裹整个逻辑
            session_uuid = uuid.UUID(session_id)
            
            # 简化查询，直接获取消息
            query = self.db.query(ChatMessage).filter(
                ChatMessage.session_id == session_uuid
            )
            
            # 处理before_id参数
            if before_id:
                try:
                    before_uuid = uuid.UUID(before_id)
                    before_message = self.db.query(ChatMessage).filter(
                        ChatMessage.id == before_uuid
                    ).first()
                    if before_message:
                        query = query.filter(
                            ChatMessage.created_at < before_message.created_at
                        )
                except Exception:
                    # 忽略无效的before_id
                    pass
            
            # 获取按时间排序的消息 - 确保按创建时间正序排列
            messages = query.order_by(ChatMessage.created_at.asc()).limit(limit).all()
            
            # 构建简化的响应数据
            result = []
            for message in messages:
                try:
                    # 避免任何可能的空值或类型错误
                    message_data = {
                "id": str(message.id),
                        "role": message.role or "",
                        "content": message.content or "",
                        "created_at": message.created_at.isoformat() if message.created_at else "",
                        "sources": message.sources if isinstance(message.sources, list) else [],
                        "confidence": float(message.confidence) if message.confidence is not None else 0.0
                    }
                    
                    # 安全地处理rich_content
                    if message.rich_content and isinstance(message.rich_content, list):
                        message_data["reply"] = message.rich_content
                    else:
                        message_data["reply"] = [{"type": "markdown", "content": message.content or ""}]
                    
                    result.append(message_data)
                except Exception as msg_error:
                    # 如果单个消息处理失败，记录错误但继续处理其他消息
                    print(f"Error processing message {message.id}: {str(msg_error)}")
                    continue
            
            return {
                "messages": result,
                "has_more": len(messages) >= limit
            }
            
        except Exception as e:
            # 记录详细错误，但返回空结果而不是抛出异常
            import traceback
            print(f"Error in get_chat_history: {str(e)}")
            print(traceback.format_exc())
            return {
                "messages": [],
                "has_more": False
            }

    async def send_message(self, session_id: str, message: str, user_id: str) -> dict:
        """发送消息到特定会话，使用独立事务处理用户消息和AI响应"""
        # 添加消息状态跟踪
        message_status = {"stored": False, "user_message_id": None, "ai_message_id": None}
        user_message = None
        user_message_stored = False
        
        # 第一个事务：存储用户消息
        try:
            # 生成精确的当前时间戳
            current_time = datetime.utcnow()
            
            # 保存用户消息，使用明确的时间戳
            user_message = ChatMessage(
                session_id=uuid.UUID(session_id),
                role='user',
                content=message,
                message_type='markdown',
                created_at=current_time
            )
            self.db.add(user_message)
            self.db.commit()  # 提交用户消息事务
            user_message_stored = True
            message_status["user_message_id"] = str(user_message.id)
            print(f"User message stored with ID: {user_message.id}")
        except Exception as user_msg_error:
            self.db.rollback()
            print(f"Error storing user message: {str(user_msg_error)}")
            # 继续尝试处理
        
        # 获取会话和生成响应 - 不在事务中进行
        session = None
        response = None
        processing_errors = []
        context_history = ""
        
        try:
            # 获取会话信息
            session = self.db.query(ChatSession).filter(
                ChatSession.id == uuid.UUID(session_id),
                ChatSession.user_id == uuid.UUID(user_id)
            ).first()
            
            if not session:
                error_msg = "Chat session not found"
                return {
                    "user_message": self._format_message(user_message),
                    "ai_message": self._format_ai_message({
                        "answer": error_msg,
                        "sources": [],
                        "confidence": 0.0,
                        "reply": [{"type": "markdown", "content": error_msg}]
                    })
                }
            
            # 获取历史消息作为上下文 - 独立查询
            try:
                history_messages = self.db.query(ChatMessage).filter(
                    ChatMessage.session_id == uuid.UUID(session_id)
                ).order_by(ChatMessage.created_at.asc()).all()
                
                # 构建上下文历史
                context_history = "\n".join([
                    f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                    for msg in history_messages[-8:]
                ])
            except Exception as history_error:
                print(f"Error getting message history: {str(history_error)}")
                context_history = ""  # 使用空历史
            
            # 生成回答 - 此处是计算密集型操作，不在事务中进行
            # [保留现有的AI响应生成代码]...
            
        except Exception as process_error:
            print(f"Error in message processing: {str(process_error)}")
            response = {
                "answer": f"处理消息时发生错误: {str(process_error)}",
                "sources": [],
                "confidence": 0.0,
                "reply": [{"type": "markdown", "content": f"处理消息时发生错误: {str(process_error)}"}]
            }
        
        # 第三个事务：存储AI响应
        ai_message = None
        try:
            # 使用独立事务存储AI响应
            if response:
                ai_message = await self._save_ai_response(session_id, response)
                if ai_message:
                    message_status["ai_message_id"] = str(ai_message.id)
                    
                    # 更新会话时间 - 轻量级操作
                    if session:
                        session.updated_at = datetime.utcnow()
                        self.db.commit()
                    message_status["stored"] = True
                    print(f"AI response successfully stored with ID: {ai_message.id}")
            
        except Exception as save_error:
            self.db.rollback()
            print(f"Error saving AI response: {str(save_error)}")
            # 创建一个错误响应
            message_status["error"] = str(save_error)
        
        # 返回结果
        return {
            "user_message": self._format_message(user_message),
            "ai_message": self._format_ai_message(response, ai_message),
            "_message_status": message_status
        }

    async def _save_message_fragments(self, message_id, content_chunks):
        """可靠的消息片段存储，使用独立事务"""
        from models.chat import MessageFragment
        
        successful = False
        retries = 0
        max_retries = 3
        
        while not successful and retries < max_retries:
            try:
                # 使用独立事务会话
                from sqlalchemy.orm import sessionmaker
                from database import engine
                
                SessionLocal = sessionmaker(bind=engine)
                fragment_session = SessionLocal()
                
                try:
                    # 先清除可能存在的旧片段
                    fragment_session.query(MessageFragment).filter(
                        MessageFragment.message_id == message_id
                    ).delete()
                    
                    # 批量添加新片段
                    fragments = []
                    for i, chunk in enumerate(content_chunks):
                        fragment = MessageFragment(
                            message_id=message_id,
                            fragment_index=i,
                            content=chunk,
                            created_at=datetime.utcnow(),
                            content_hash=self._calculate_content_hash(chunk)  # 添加内容哈希校验
                        )
                        fragments.append(fragment)
                    
                    fragment_session.bulk_save_objects(fragments)
                    fragment_session.commit()
                    
                    # 验证片段数量和内容
                    stored_fragments = fragment_session.query(MessageFragment).filter(
                        MessageFragment.message_id == message_id
                    ).order_by(MessageFragment.fragment_index).all()
                    
                    if len(stored_fragments) == len(content_chunks):
                        # 验证内容哈希
                        validated = True
                        for i, fragment in enumerate(stored_fragments):
                            expected_hash = self._calculate_content_hash(content_chunks[i])
                            if fragment.content_hash != expected_hash:
                                validated = False
                                print(f"Fragment {i} hash mismatch for message {message_id}")
                                break
                        
                        if validated:
                            successful = True
                            print(f"Successfully stored and verified {len(fragments)} fragments for message {message_id}")
                    else:
                        print(f"Fragment count mismatch: expected {len(content_chunks)}, got {len(stored_fragments)}")
                except Exception as inner_error:
                    fragment_session.rollback()
                    print(f"Error in fragment transaction: {str(inner_error)}")
                    raise
                finally:
                    fragment_session.close()
                
                if successful:
                    return True
                
            except Exception as outer_error:
                print(f"Fragment storage attempt {retries+1} failed: {str(outer_error)}")
                retries += 1
                await asyncio.sleep(0.5)  # 短暂延迟后重试
        
        return successful

    def _calculate_content_hash(self, content):
        """计算内容哈希用于校验"""
        import hashlib
        if isinstance(content, list) or isinstance(content, dict):
            content = json.dumps(content, sort_keys=True)
        return hashlib.md5(str(content).encode('utf-8')).hexdigest()

    async def _save_ai_response(self, session_id, response):
        """保存AI回答到数据库，增强大型内容处理"""
        try:
            print(f"BEGIN _save_ai_response for session {session_id}")
            # 对大型rich_content进行处理
            reply = response.get("reply", [])
            if reply and len(reply) > 0:
                # 估计大小
                serialized = json.dumps(reply)
                print(f"Rich content size: {len(serialized)} bytes")
                
                # 如果超过阈值，使用独立事务拆分存储
                if len(serialized) > 100000:  # 约100KB
                    print(f"Content exceeds size limit, splitting into fragments")
                    
                    # 创建主消息，包含摘要
                    main_message = ChatMessage(
                        session_id=uuid.UUID(session_id),
                        role='assistant',
                        content=response.get("answer", "")[:5000],
                        sources=response.get("sources", [])[:5],
                        confidence=response.get("confidence", 0.0),
                        message_type='markdown',
                        rich_content=[{
                            "type": "markdown", 
                            "content": "此回复包含大量数据，已分片存储以提高性能"
                        }],
                        has_fragments=True,
                        created_at=datetime.utcnow()
                    )
                    self.db.add(main_message)
                    
                    # 使用flush确保ID生成但不提交事务
                    self.db.flush()
                    print(f"Created main message with ID: {main_message.id}")
                    
                    # 使用独立事务处理分片存储
                    chunks = self.split_content(reply)
                    fragments_stored = await self._save_message_fragments(main_message.id, chunks)
                    
                    if not fragments_stored:
                        # 如果片段存储失败，添加警告到主消息
                        main_message.rich_content.append({
                            "type": "markdown",
                            "content": "⚠️ 警告：消息片段存储失败，内容可能不完整"
                        })
                    
                    return main_message
            
            # 常规消息处理逻辑
            ai_time = datetime.utcnow()
            assistant_message = ChatMessage(
                session_id=uuid.UUID(session_id),
                role='assistant',
                content=response.get("answer", ""),
                sources=response.get("sources", []),
                confidence=response.get("confidence", 0.0),
                message_type='markdown',
                rich_content=reply,
                created_at=ai_time
            )
            self.db.add(assistant_message)
            self.db.flush()
            print(f"Created regular message with ID: {assistant_message.id}")
            return assistant_message
        except Exception as e:
            print(f"Error saving AI response: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    def split_content(self, content, max_size=50000):
        """将大型内容拆分成多个片段"""
        if isinstance(content, list):
            result = []
            current_chunk = []
            current_size = 0
            
            for item in content:
                # 估算项目大小
                item_size = len(json.dumps(item))
                
                if current_size + item_size > max_size and current_chunk:
                    # 当前块已满，保存并创建新块
                    result.append(current_chunk)
                    current_chunk = [item]
                    current_size = item_size
                else:
                    # 添加到当前块
                    current_chunk.append(item)
                    current_size += item_size
            
            # 添加最后一个块
            if current_chunk:
                result.append(current_chunk)
            
            return result
        else:
            # 如果不是列表，返回原内容
            return [content]

    def _format_message(self, message):
        """将消息对象格式化为API响应格式"""
        if not message:
            # 生成更具唯一性的ID
            unique_id = f"msg_user_{datetime.utcnow().isoformat()}_{uuid.uuid4().hex[:8]}"
            return {
                "id": unique_id,
                "role": "user",
                "content": "",
                "created_at": datetime.utcnow().isoformat(),
                "sources": [],
                "confidence": 0,
                "reply": []
            }
        
        return {
            "id": str(message.id),
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at.isoformat() if message.created_at else datetime.utcnow().isoformat(),
            "sources": [],
            "confidence": 0,
            "reply": []
        }

    def _format_ai_message(self, response, message=None):
        """将AI响应格式化为API响应格式"""
        if message:
            return {
                "id": str(message.id),
                "role": "assistant",
                "content": message.content,
                "created_at": message.created_at.isoformat() if message.created_at else datetime.utcnow().isoformat(),
                "sources": message.sources or [],
                "confidence": message.confidence or 0,
                "reply": message.rich_content or [{"type": "markdown", "content": message.content or ""}]
            }
        
        # 生成更具唯一性的ID
        unique_id = f"msg_ai_{datetime.utcnow().isoformat()}_{uuid.uuid4().hex[:8]}"
        
        # 修复：处理response为None的情况
        if response is None:
            return {
                "id": unique_id,
                "role": "assistant",
                "content": "处理请求时发生错误",
                "created_at": datetime.utcnow().isoformat(),
                "sources": [],
                "confidence": 0.0,
                "reply": [{"type": "markdown", "content": "处理请求时发生错误"}]
            }
        
        # 使用响应对象创建格式化消息
        return {
            "id": unique_id,
            "role": "assistant",
            "content": response.get("answer", ""),
            "created_at": datetime.utcnow().isoformat(),
            "sources": response.get("sources", []),
            "confidence": response.get("confidence", 0.0),
            "reply": response.get("reply", [{"type": "markdown", "content": response.get("answer", "")}])
        }

    async def create_ai_chat_session(self, user_id: str, title: str = None) -> dict:
        """创建一个无文档的普通AI对话会话"""
        return await self.create_chat_session(
            user_id=user_id,
            title=title or f"AI对话 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            paper_ids=None,
            session_type="general"
        )

    async def remove_document_from_session(self, session_id: str, document_id: str, user_id: str) -> dict:
        """从会话中移除文档"""
        try:
            # 验证会话存在且属于该用户
            session = self.db.query(ChatSession).filter(
                ChatSession.id == uuid.UUID(session_id),
                ChatSession.user_id == uuid.UUID(user_id)
            ).first()
            
            if not session:
                raise ValueError("会话不存在或不属于该用户")
            
            # 查找要删除的文档关联
            doc = self.db.query(SessionDocument).filter(
                SessionDocument.id == uuid.UUID(document_id),
                SessionDocument.session_id == uuid.UUID(session_id)
            ).first()
            
            if not doc:
                raise ValueError("文档不在当前会话中或不存在")
            
            # 获取文档数量
            doc_count = self.db.query(SessionDocument).filter(
                SessionDocument.session_id == uuid.UUID(session_id)
            ).count()
            
            # 如果是唯一文档，需要特殊处理
            if doc_count == 1:
                # 如果是最后一个文档，清除session.paper_id并修改类型
                session.paper_id = None
                session.session_type = "general"
            elif session.paper_id == doc.paper_id:
                # 如果删除的是主文档，更新为另一个文档
                next_doc = self.db.query(SessionDocument).filter(
                    SessionDocument.session_id == uuid.UUID(session_id),
                    SessionDocument.id != uuid.UUID(document_id)
                ).first()
                
                if next_doc:
                    session.paper_id = next_doc.paper_id
            
            # 删除文档关联
            self.db.delete(doc)
            
            # 更新其他文档的顺序
            remaining_docs = self.db.query(SessionDocument).filter(
                SessionDocument.session_id == uuid.UUID(session_id)
            ).order_by(SessionDocument.order).all()
            
            for i, remaining_doc in enumerate(remaining_docs):
                remaining_doc.order = i
            
            self.db.commit()
            
            return {
                "status": "success",
                "message": "Document removed from session",
                "session_id": session_id,
                "session_type": session.session_type
            }
        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to remove document from session: {str(e)}")

    async def get_session_documents(self, session_id: str, user_id: str) -> List[dict]:
        """获取会话关联的所有文档"""
        try:
            # 验证会话存在且属于该用户
            session = self.db.query(ChatSession).filter(
                ChatSession.id == uuid.UUID(session_id),
                ChatSession.user_id == uuid.UUID(user_id)
            ).first()
            
            if not session:
                raise ValueError("会话不存在或不属于该用户")
            
            # 获取关联文档
            session_docs = self.db.query(SessionDocument).filter(
                SessionDocument.session_id == uuid.UUID(session_id)
            ).order_by(SessionDocument.order).all()
            
            # 构建结果
            result = []
            for doc in session_docs:
                # 获取文档详情
                paper = self.db.query(PaperAnalysis).filter(
                    PaperAnalysis.paper_id == doc.paper_id
                ).first()
                
                if paper:
                    result.append({
                        "id": str(doc.id),
                        "paper_id": str(doc.paper_id),
                        "filename": paper.filename,
                        "file_type": paper.file_type,
                        "file_size": paper.file_size,
                        "order": doc.order,
                        "added_at": doc.added_at.isoformat() if doc.added_at else None,
                        "summary": paper.summary
                    })
            
            return result
        except Exception as e:
            raise Exception(f"Failed to get session documents: {str(e)}")

    async def add_document_to_session(self, session_id: str, paper_id: str, user_id: str) -> dict:
        """向现有会话添加文档"""
        try:
            # 验证会话存在且属于该用户
            session = self.db.query(ChatSession).filter(
                ChatSession.id == uuid.UUID(session_id),
                ChatSession.user_id == uuid.UUID(user_id)
            ).first()
            
            if not session:
                raise ValueError("会话不存在或不属于该用户")
            
            # 验证文档存在
            paper = self.db.query(PaperAnalysis).filter(
                PaperAnalysis.paper_id == uuid.UUID(paper_id)
            ).first()
            
            if not paper:
                raise ValueError(f"文档ID {paper_id} 不存在")
            
            # 检查该文档是否已在会话中
            existing_doc = self.db.query(SessionDocument).filter(
                SessionDocument.session_id == uuid.UUID(session_id),
                SessionDocument.paper_id == uuid.UUID(paper_id)
            ).first()
            
            if existing_doc:
                raise ValueError("该文档已添加到会话中")
            
            # 检查会话中的文档数量是否超出限制
            doc_count = self.db.query(SessionDocument).filter(
                SessionDocument.session_id == uuid.UUID(session_id)
            ).count()
            
            if doc_count >= self.MAX_DOCUMENTS_PER_SESSION:
                raise ValueError(f"一个会话最多支持{self.MAX_DOCUMENTS_PER_SESSION}个文档")
            
            # 创建关联
            session_doc = SessionDocument(
                session_id=uuid.UUID(session_id),
                paper_id=uuid.UUID(paper_id),
                order=doc_count,  # 新文档添加到末尾
                filename=paper.filename
            )
            self.db.add(session_doc)
            
            # 如果是首个文档，更新会话的paper_id（向后兼容）
            if doc_count == 0 and not session.paper_id:
                session.paper_id = uuid.UUID(paper_id)
                session.session_type = "document"  # 更新类型为文档会话
            
            self.db.commit()
            
            # 返回添加的文档信息
            return {
                "id": str(session_doc.id),
                "session_id": session_id,
                "paper_id": paper_id,
                "filename": paper.filename,
                "order": session_doc.order
            }
        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to add document to session: {str(e)}")

    async def delete_chat_session(self, session_id: str, user_id: str) -> dict:
        """逻辑删除会话（仅标记为非活跃）"""
        try:
            # 验证会话存在且属于该用户
            session = self.db.query(ChatSession).filter(
                ChatSession.id == uuid.UUID(session_id),
                ChatSession.user_id == uuid.UUID(user_id)
            ).first()
            
            if not session:
                raise ValueError("会话不存在或不属于该用户")
            
            # 执行逻辑删除（标记为非活跃）
            session.is_active = False
            session.updated_at = datetime.utcnow()
            
            self.db.commit()
            
            return {
                "status": "success",
                "message": "会话已删除",
                "session_id": session_id
            }
        except Exception as e:
            self.db.rollback()
            raise Exception(f"删除会话失败: {str(e)}")

    async def update_chat_session_title(self, session_id: str, user_id: str, new_title: str) -> dict:
        """更新会话标题"""
        try:
            # 验证会话存在且属于该用户
            session = self.db.query(ChatSession).filter(
                ChatSession.id == uuid.UUID(session_id),
                ChatSession.user_id == uuid.UUID(user_id)
            ).first()
            
            if not session:
                raise ValueError("会话不存在或不属于该用户")
            
            # 更新标题
            session.title = new_title
            session.updated_at = datetime.utcnow()
            
            self.db.commit()
            
            return {
                "id": str(session.id),
                "title": session.title,
                "updated_at": session.updated_at.isoformat(),
                "status": "success"
            }
        except Exception as e:
            self.db.rollback()
            raise Exception(f"更新会话标题失败: {str(e)}")

    def _estimate_memory_usage(self, content_size):
        # 预估内存使用并设置处理级别
        estimated_mb = content_size * 5 / (1024 * 1024)  # 粗略估计
        return {
            "level": "high" if estimated_mb > 500 else "medium" if estimated_mb > 100 else "low",
            "estimated_mb": estimated_mb
        }

    @contextmanager
    def reliable_transaction(self):
        """提供事务重试与恢复机制"""
        retry_count = 0
        while retry_count < 3:
            try:
                yield
                self.db.commit()
                return
            except Exception as e:
                self.db.rollback()
                retry_count += 1
                if retry_count == 3:
                    raise
                time.sleep(1)  # 指数退避

    async def stream_message(self, session_id: str, message: str, user_id: str):
        """流式发送消息，逐步返回AI回复"""
        # 添加消息状态跟踪
        message_status = {"stored": False, "user_message_id": None}
        
        try:
            # 1. 保存用户消息
            current_time = datetime.utcnow()
            user_message = ChatMessage(
                session_id=uuid.UUID(session_id),
                role='user',
                content=message,
                message_type='markdown',
                created_at=current_time
            )
            self.db.add(user_message)
            self.db.commit()  # 立即提交用户消息
            message_status["user_message_id"] = str(user_message.id)
            message_status["stored"] = True
            
            # 2. 获取会话信息
            session = self.db.query(ChatSession).filter(
                ChatSession.id == uuid.UUID(session_id),
                ChatSession.user_id == uuid.UUID(user_id)
            ).first()
            
            if not session:
                yield {"error": "Chat session not found", "done": True}
                return
            
            # 3. 获取历史消息作为上下文
            try:
                history_messages = self.db.query(ChatMessage).filter(
                    ChatMessage.session_id == uuid.UUID(session_id)
                ).order_by(ChatMessage.created_at.asc()).all()
                
                context_history = "\n".join([
                    f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                    for msg in history_messages[-8:]
                ])
            except Exception as e:
                print(f"Error getting history for streaming: {str(e)}")
                context_history = ""
            
            # 4. 准备上下文
            retrieval_context = None
            if session.session_type == "document":
                # 获取相关文档IDs
                paper_ids = []
                if session.paper_id:  # 向后兼容
                    paper_ids.append(str(session.paper_id))
                
                # 从关联表获取多文档信息
                session_docs = self.db.query(SessionDocument).filter(
                    SessionDocument.session_id == uuid.UUID(session_id)
                ).order_by(SessionDocument.order).all()
                
                for doc in session_docs:
                    paper_id = str(doc.paper_id)
                    if paper_id not in paper_ids:
                        paper_ids.append(paper_id)
                
                if paper_ids:
                    if len(paper_ids) > 1:
                        retrieval_context = await self.rag_retriever.get_context_from_multiple_docs(
                            question=message,
                            paper_ids=paper_ids,
                            history=context_history
                        )
                    else:
                        retrieval_context = await self.rag_retriever.get_relevant_context(
                            question=message,
                            paper_id=paper_ids[0],
                            history=context_history
                        )
            
            # 5. 流式生成回答
            full_response = {"answer": "", "sources": [], "confidence": 0.5, "reply": []}
            
            # 收集片段
            chunks = []
            async for chunk in self.ai_manager.stream_response(
                question=message,
                context=retrieval_context,
                history=context_history
            ):
                # 更新完整响应
                if not chunk.get("error"):
                    text = chunk.get("partial_response", "")
                    full_response["answer"] += text
                    chunks.append(text)
                    
                    # 转换为前端期望的格式
                    yield {
                        "id": f"stream_{len(chunks)}_{session_id}",
                        "content": text,
                        "done": chunk.get("done", False)
                    }
                else:
                    yield {
                        "error": chunk.get("error"),
                        "done": True
                    }
                    return
            
            # 6. 在流式传输完成后保存完整响应
            # 使用_parse_response_content解析完整响应
            full_response["reply"] = self.ai_manager._parse_response_content(full_response["answer"])
            
            try:
                # 异步保存完整响应
                ai_message = await self._save_ai_response(session_id, full_response)
                if ai_message:
                    # 更新会话时间
                    session.updated_at = datetime.utcnow()
                    self.db.commit()
                    
                    yield {
                        "message_id": str(ai_message.id),
                        "saved": True,
                        "done": True
                    }
            except Exception as save_error:
                print(f"Error saving streamed response: {str(save_error)}")
                yield {
                    "error": f"Response streaming completed but failed to save: {str(save_error)}",
                    "done": True
                }
            
        except Exception as e:
            print(f"Streaming error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            yield {
                "error": str(e),
                "done": True
            }

    # 需要添加类似代码:
    app = Celery('paper_analyzer')
    
    @app.task
    def build_index_task(content, paper_id, line_mapping=None):
        # 异步构建索引
        # ...

    # 在关键操作处添加详细日志
        logger.info(f"Processing document {paper_id} with size {len(content)}")