# backend/services/local_glossary_manager.py    本地术语表管理器
from typing import Optional
from datetime import datetime
import logging
import traceback
from sqlalchemy.orm import Session
from models.glossary import Glossary, GlossaryEntry

# 配置日志
logger = logging.getLogger(__name__)

class LocalGlossaryManager:
    def __init__(self, db: Session):
        self.db = db

    def _normalize_lang_code(self, lang_code: str) -> str:
        """标准化语言代码"""
        return lang_code.upper() if lang_code else None

    async def search_glossaries_and_entries(
        self,
        name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        page: int = 1,
        page_size: int = 10
    ) -> dict:
        """搜索本地数据库中的术语表和词汇明细，优化分页查询"""
        try:
            # 1. 构建基础查询
            entries_query = self.db.query(
                GlossaryEntry,
                Glossary.name.label('glossary_name'),
                Glossary.source_lang,
                Glossary.target_lang,
                Glossary.created_at.label('glossary_created_at'),
                Glossary.updated_at.label('glossary_updated_at')
            ).join(Glossary)

            # 2. 添加过滤条件
            if name:
                entries_query = entries_query.filter(Glossary.name.ilike(f"%{name}%"))
            if source_lang:
                entries_query = entries_query.filter(Glossary.source_lang == self._normalize_lang_code(source_lang))
            if target_lang:
                entries_query = entries_query.filter(Glossary.target_lang == self._normalize_lang_code(target_lang))
            if start_date:
                entries_query = entries_query.filter(Glossary.created_at >= start_date)
            if end_date:
                entries_query = entries_query.filter(Glossary.created_at <= end_date)

            # 3. 计算总条目数
            total_entries = entries_query.count()
            
            # 4. 计算总页数
            total_pages = (total_entries + page_size - 1) // page_size
            
            # 5. 验证并修正页码
            valid_page = min(max(1, page), total_pages) if total_pages > 0 else 1
            
            # 6. 获取分页数据
            offset = (valid_page - 1) * page_size
            entries = entries_query.offset(offset).limit(page_size).all()

            # 7. 格式化返回数据
            result_entries = [
                {
                    "id": entry.GlossaryEntry.id,
                    "glossary_id": entry.GlossaryEntry.glossary_id,
                    "glossary_name": entry.glossary_name,
                    "source_lang": entry.source_lang,
                    "target_lang": entry.target_lang,
                    "source_term": entry.GlossaryEntry.source_term,
                    "target_term": entry.GlossaryEntry.target_term,
                    "created_at": entry.GlossaryEntry.created_at.isoformat(),
                    "glossary_created_at": entry.glossary_created_at.isoformat(),
                    "glossary_updated_at": entry.glossary_updated_at.isoformat() if entry.glossary_updated_at else None
                }
                for entry in entries
            ]

            return {
                "total": total_entries,
                "page": valid_page,
                "page_size": page_size,
                "total_pages": total_pages,
                "entries": result_entries
            }

        except Exception as e:
            logger.error(f"Error searching glossaries and entries: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def get_glossary_by_id(self, glossary_id: int) -> Optional[dict]:
        """通过ID获取术语表详细信息"""
        try:
            glossary = self.db.query(Glossary).filter(Glossary.id == glossary_id).first()
            if not glossary:
                return None

            entries = self.db.query(GlossaryEntry).filter(
                GlossaryEntry.glossary_id == glossary.id
            ).all()

            return {
                "id": glossary.id,
                "deepl_glossary_id": glossary.deepl_glossary_id,
                "name": glossary.name,
                "source_lang": glossary.source_lang,
                "target_lang": glossary.target_lang,
                "created_at": glossary.created_at.isoformat(),
                "updated_at": glossary.updated_at.isoformat() if glossary.updated_at else None,
                "entries": [
                    {
                        "id": entry.id,
                        "source_term": entry.source_term,
                        "target_term": entry.target_term,
                        "created_at": entry.created_at.isoformat()
                    }
                    for entry in entries
                ]
            }
        except Exception as e:
            logger.error(f"Error getting glossary by id: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def get_glossary_by_lang_pair(
        self,
        source_lang: str,
        target_lang: str
    ) -> Optional[dict]:
        """通过语言对获取术语表"""
        try:
            glossary = self.db.query(Glossary).filter(
                Glossary.source_lang == self._normalize_lang_code(source_lang),
                Glossary.target_lang == self._normalize_lang_code(target_lang)
            ).first()
            
            if not glossary:
                return None

            return await self.get_glossary_by_id(glossary.id)
        except Exception as e:
            logger.error(f"Error getting glossary by language pair: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    # 更新术语表条目的目标术语
    async def update_glossary_entry(
        self,
        entry_id: int,
        target_term: str
    ) -> dict:
        """更新术语表条目的目标术语"""
        try:
            entry = self.db.query(GlossaryEntry).filter(GlossaryEntry.id == entry_id).first()
            if not entry:
                raise ValueError(f"Entry with ID {entry_id} not found")
            
            # 更新目标术语
            entry.target_term = target_term
            entry.updated_at = datetime.utcnow()
            
            # 同时更新所属术语表的更新时间
            glossary = self.db.query(Glossary).filter(Glossary.id == entry.glossary_id).first()
            if glossary:
                glossary.updated_at = datetime.utcnow()
            
            self.db.commit()
            
            return {
                "id": entry.id,
                "source_term": entry.source_term,
                "target_term": entry.target_term,
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat() if entry.updated_at else None
            }
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating glossary entry: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def delete_glossary_entry(
        self,
        entry_id: int
    ) -> bool:
        """删除术语表条目"""
        try:
            entry = self.db.query(GlossaryEntry).filter(GlossaryEntry.id == entry_id).first()
            if not entry:
                raise ValueError(f"Entry with ID {entry_id} not found")
            
            # 更新所属术语表的更新时间
            glossary = self.db.query(Glossary).filter(Glossary.id == entry.glossary_id).first()
            if glossary:
                glossary.updated_at = datetime.utcnow()
            
            self.db.delete(entry)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting glossary entry: {str(e)}")
            logger.error(traceback.format_exc())
            raise
