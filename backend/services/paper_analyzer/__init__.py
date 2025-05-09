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

class PaperAnalyzerService:
    def __init__(self, db: Session):
        self.db = db
        self.ai_manager = AIManager()
        self.paper_processor = PaperProcessor()
        self.rag_retriever = RAGRetriever(db)
        self.translator = Translator()

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
                    self.db  # 显式传递db连接
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
                "is_scanned": paper_analysis.is_scanned
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

    async def ask_question(self, question: str, paper_id: str):
        """向文档提问"""
        try:
            # 验证paper_id
            if paper_id == "current_paper_id":
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
                # 首先检查数据库中是否存在这个paper_id
                paper = self.db.query(PaperAnalysis).filter(
                    PaperAnalysis.paper_id == uuid.UUID(paper_id)
                ).first()
                
                if not paper:
                    return {
                        "status": "error",
                        "message": f"Paper ID {paper_id} not found in database"
                    }
                    
                context = await self.rag_retriever.get_relevant_context(question, paper_id)
            except Exception as e:
                # 记录详细错误信息
                import traceback
                print(f"Context retrieval error for paper {paper_id}: {str(e)}")
                print(traceback.format_exc())
                return {
                    "status": "error",
                    "message": f"获取上下文失败: {str(e)}"
                }
            
            # 获取 AI 回答
            response = await self.ai_manager.get_response(question, context)
            
            # 存储问答记录
            paper_question = PaperQuestion(
                question=question,
                answer=response["answer"],  # 只存储答案文本
                paper_ids=[uuid.UUID(paper_id)]
            )
            self.db.add(paper_question)
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