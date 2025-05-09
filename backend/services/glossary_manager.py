# backend/services/glossary_manager.py    术语表管理器
import httpx
import os
from typing import Dict, List, Optional
import json
from datetime import datetime
import asyncio
import logging
import traceback
from sqlalchemy.orm import Session
from models.glossary import Glossary, GlossaryEntry

# 添加 logger 配置
logger = logging.getLogger(__name__)

class GlossaryManager:
    def __init__(self, db: Session):
        self.db = db
        self.api_key = os.getenv("DEEPL_API_KEY")
        self.api_type = os.getenv("DEEPL_API_TYPE", "free")
        # 更新到 v3 API
        self.base_url = "https://api-free.deepl.com/v3" if self.api_type.lower() == "free" else "https://api.deepl.com/v3"
        self.headers = {
            "Authorization": f"DeepL-Auth-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        self.cache_lock = asyncio.Lock()  # 添加缓存锁

    def _get_language_pair_cache_path(self, source_lang: str, target_lang: str) -> str:
        """获取语言对的缓存文件路径"""
        cache_filename = f"glossary_{source_lang.upper()}_{target_lang.upper()}.json"
        return os.path.join(self.cache_dir, cache_filename)

    async def create_glossary(self, glossary_payload: dict) -> dict:
        """创建新的术语表"""
        try:
            logger.info(f"Creating new glossary with name: {glossary_payload.get('name')}")
            
            # 验证 payload
            if not self.validate_glossary_payload(glossary_payload):
                raise ValueError("Invalid glossary payload")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/glossaries",
                    json=glossary_payload,
                    headers=self.headers
                )
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Successfully created glossary with ID: {result.get('glossary_id')}")
                return result

        except httpx.HTTPError as e:
            logger.error(f"HTTP error creating glossary: {str(e)}")
            logger.error(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
            raise
        except Exception as e:
            logger.error(f"Error creating glossary: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def list_glossaries(self) -> List[dict]:
        """获取所有术语表列表 (GET /v3/glossaries)"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/glossaries",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["glossaries"]

    async def get_glossary(self, glossary_id: str) -> dict:
        """获取特定术语表信息 (GET /v3/glossaries/{glossary_id})"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/glossaries/{glossary_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()

    async def get_entries(self, glossary_id: str) -> str:
        """获取术语表条目"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/glossaries/{glossary_id}/entries",
                    headers=self.headers
                )
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.error(f"Error getting glossary entries: {str(e)}")
            return ""  # 返回空字符串而不是抛出异常，这样可以继续处理

    async def update_glossary(self, glossary_id: str, payload: dict) -> dict:
        """
        更新术语表 (PATCH /v3/glossaries/{glossary_id})
        用于添加新条目或更新现有条目
        """
        if not self.validate_glossary_payload(payload):
            raise ValueError("Invalid glossary payload")

        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{self.base_url}/glossaries/{glossary_id}",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            self._cache_glossary(glossary_id, result)
            return result

    async def replace_glossary(self, glossary_id: str, payload: dict) -> dict:
        """
        替换术语表 (PUT /v3/glossaries/{glossary_id})
        完全替换现有术语表的内容
        """
        if not self.validate_glossary_payload(payload):
            raise ValueError("Invalid glossary payload")

        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.base_url}/glossaries/{glossary_id}",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            self._cache_glossary(glossary_id, result)
            return result

    async def delete_glossary(self, glossary_id: str) -> None:
        """删除术语表 (DELETE /v3/glossaries/{glossary_id})"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.base_url}/glossaries/{glossary_id}",
                    headers=self.headers
                )
                response.raise_for_status()
                logger.info(f"Successfully deleted glossary {glossary_id} from DeepL")
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error deleting glossary {glossary_id}: {str(e)}")
            logger.error(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
            raise
        except Exception as e:
            logger.error(f"Error deleting glossary {glossary_id}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def delete_dictionary(self, glossary_id: str, source_lang: str, target_lang: str) -> None:
        """删除特定语言对的字典"""
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.base_url}/glossaries/{glossary_id}/dictionaries",
                headers=self.headers,
                params={
                    "source_lang": source_lang,
                    "target_lang": target_lang
                }
            )
            response.raise_for_status()
            # 更新缓存
            cached = self.get_cached_glossary(glossary_id)
            if cached:
                cached["dictionaries"] = [d for d in cached["dictionaries"] 
                                        if not (d["source_lang"] == source_lang and 
                                               d["target_lang"] == target_lang)]
                self._cache_glossary(glossary_id, cached)

    def validate_glossary_payload(self, payload: dict) -> bool:
        """验证术语表 payload 是否符合 API 要求"""
        try:
            # 检查必需字段
            if not all(key in payload for key in ["name", "dictionaries"]):
                return False

            # 检查名称长度（最大 1024 字节）
            if len(payload["name"].encode('utf-8')) > 1024:
                return False

            # 检查字典
            for dictionary in payload["dictionaries"]:
                # 检查必需字段
                if not all(key in dictionary for key in 
                          ["source_lang", "target_lang", "entries", "entries_format"]):
                    return False

                # 检查 entries_format
                if dictionary["entries_format"] not in ["tsv", "csv"]:
                    return False

                # 检查条目格式和大小限制
                entries = dictionary["entries"].split("\n")
                seen_sources = set()  # 用于检查重复的源术语

                for entry in entries:
                    if dictionary["entries_format"] == "tsv":
                        parts = entry.split("\t")
                    else:  # csv
                        parts = entry.split(",")

                    if len(parts) != 2:
                        return False

                    source, target = parts
                    
                    # 检查重复的源术语
                    if source in seen_sources:
                        return False
                    seen_sources.add(source)

                    # 检查空条目
                    if not source.strip() or not target.strip():
                        return False

                    # 检查长度限制（每个短语最大 1024 字节）
                    if len(source.encode('utf-8')) > 1024 or len(target.encode('utf-8')) > 1024:
                        return False

                    # 检查控制字符
                    if any(ord(c) < 32 for c in source + target):
                        return False

                    # 检查首尾空白字符
                    if source != source.strip() or target != target.strip():
                        return False

                # 检查字典大小限制（10MB）
                if len(dictionary["entries"].encode('utf-8')) > 10 * 1024 * 1024:
                    return False

            return True
        except Exception as e:
            logger.error(f"Glossary validation error: {str(e)}")
            return False

    def _cache_glossary(self, glossary_id: str, data: dict):
        """缓存术语表数据"""
        cache_path = self._get_cache_path(glossary_id)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_cached_glossary(self, glossary_id: str) -> Optional[dict]:
        """获取缓存的术语表数据"""
        cache_path = self._get_cache_path(glossary_id)
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _normalize_lang_code(self, lang_code: str) -> str:
        """标准化语言代码"""
        lang_code = lang_code.upper()
        # DeepL 支持的语言代码
        supported_codes = {'ZH', 'EN', 'DE', 'FR', 'ES', 'PT', 'IT', 'NL', 'PL', 'RU', 'JA', 'ID'}
        if lang_code not in supported_codes:
            raise ValueError(f"Unsupported language code: {lang_code}")
        return lang_code

    async def get_glossary_details(self, glossary_id: str, page: int = 1, page_size: int = 10) -> dict:
        """获取术语表详细信息，包括元数据和分页条目"""
        try:
            # 1. 获取术语表基本信息
            glossary_info = await self.get_glossary(glossary_id)
            
            # 2. 获取所有字典的条目
            all_entries = []
            for dictionary in glossary_info["dictionaries"]:
                try:
                    response = await self.get_entries(
                        glossary_id,
                        dictionary["source_lang"],
                        dictionary["target_lang"]
                    )
                    
                    # 解析 JSON 响应
                    response_data = json.loads(response)
                    if "dictionaries" in response_data and response_data["dictionaries"]:
                        dict_data = response_data["dictionaries"][0]
                        entries_text = dict_data.get("entries", "")
                        
                        # 解析 TSV 格式的条目
                        for line in entries_text.split('\n'):
                            if line.strip():
                                parts = line.split('\t')
                                if len(parts) == 2:
                                    source, target = parts
                                    all_entries.append({
                                        "source": source.strip(),
                                        "target": target.strip(),
                                        "source_lang": dictionary["source_lang"],
                                        "target_lang": dictionary["target_lang"]
                                    })
                                
                except Exception as e:
                    logger.warning(f"Failed to fetch entries for glossary {glossary_id} "
                                 f"with languages {dictionary['source_lang']}->{dictionary['target_lang']}: {str(e)}")
            
            # 3. 计算分页
            total_entries = len(all_entries)
            # 确保 page_size 是整数
            page_size = int(page_size)
            # 确保 page 是整数且大于0
            page = max(1, int(page))
            
            # 计算起始和结束索引
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_entries)  # 确保不超过总条目数
            
            # 获取当前页的条目
            paginated_entries = all_entries[start_idx:end_idx]
            
            # 计算总页数
            total_pages = (total_entries + page_size - 1) // page_size
            
            # 4. 返回组合结果
            return {
                "glossary_info": glossary_info,
                "entries": paginated_entries,
                "total_entries": total_entries,
                "entries_available": bool(all_entries),
                "pagination": {
                    "current_page": page,
                    "page_size": page_size,  # 使用传入的 page_size
                    "total_pages": total_pages
                }
            }
        except Exception as e:
            logger.error(f"Error getting glossary details for {glossary_id}: {str(e)}")
            raise


    async def get_latest_glossary(self, source_lang: str, target_lang: str) -> Optional[dict]:
        """获取指定语言对最新的术语表"""
        try:
            glossaries = await self.list_glossaries()
            
            # 筛选出匹配语言对的术语表
            matching_glossaries = [
                g for g in glossaries
                if any(d["source_lang"] == source_lang.upper() and 
                      d["target_lang"] == target_lang.upper() 
                      for d in g["dictionaries"])
            ]
            
            if not matching_glossaries:
                return None
                
            # 按创建时间排序，返回最新的
            latest_glossary = max(
                matching_glossaries,
                key=lambda g: datetime.fromisoformat(g["creation_time"].replace('Z', '+00:00'))
            )
            
            return latest_glossary
        except Exception as e:
            logger.error(f"Error getting latest glossary: {str(e)}")
            return None

    async def check_glossary_size(self, glossary_id: str) -> float:
        """检查术语表大小（MB）"""
        try:
            glossary = await self.get_glossary(glossary_id)
            total_size = 0
            
            for dictionary in glossary["dictionaries"]:
                entries_text = await self.get_entries(
                    glossary_id,
                    dictionary["source_lang"],
                    dictionary["target_lang"]
                )
                total_size += len(entries_text.encode('utf-8'))
            
            return total_size / (1024 * 1024)  # 转换为 MB
        except Exception as e:
            logger.error(f"Error checking glossary size: {str(e)}")
            return 0

    async def merge_entries(self, existing_entries: str, new_entries: str) -> str:
        """合并术语表条目，去重并保持格式"""
        existing_terms = set(line.strip() for line in existing_entries.split('\n') if line.strip())
        new_terms = set(line.strip() for line in new_entries.split('\n') if line.strip())
        
        # 合并并去重
        all_terms = existing_terms.union(new_terms)
        return '\n'.join(sorted(all_terms))

# 单一术语表策略
    def _get_main_glossary_name(self, source_lang: str, target_lang: str) -> str:
        """获取主术语表的标准名称"""
        return f"Main_Glossary_{source_lang.upper()}-{target_lang.upper()}"

    async def get_or_create_main_glossary(self, source_lang: str, target_lang: str) -> dict:
        """获取或创建主术语表"""
        try:
            # 标准化语言代码
            source_lang = self._normalize_lang_code(source_lang)
            target_lang = self._normalize_lang_code(target_lang)
            
            # 从数据库查询现有术语表
            existing_glossary = self.db.query(Glossary).filter(
                Glossary.source_lang == source_lang,
                Glossary.target_lang == target_lang
            ).first()
            
            if existing_glossary:
                logger.info(f"Found existing glossary for {source_lang}-{target_lang}")
                return {
                    "glossary_id": existing_glossary.deepl_glossary_id,
                    "name": existing_glossary.name,
                    "source_lang": existing_glossary.source_lang,
                    "target_lang": existing_glossary.target_lang
                }

            # 创建新的术语表
            logger.info(f"Creating new main glossary for {source_lang}-{target_lang}")
            glossary_name = self._get_main_glossary_name(source_lang, target_lang)
            
            glossary_payload = {
                "name": glossary_name,
                "dictionaries": [{
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "entries": "placeholder\tplaceholder",
                    "entries_format": "tsv"
                }]
            }

            # 调用 DeepL API 创建术语表
            deepl_result = await self.create_glossary(glossary_payload)
            
            # 保存到数据库
            new_glossary = Glossary(
                deepl_glossary_id=deepl_result["glossary_id"],
                name=glossary_name,
                source_lang=source_lang,
                target_lang=target_lang
            )
            self.db.add(new_glossary)
            self.db.commit()
            
            return deepl_result

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error in get_or_create_main_glossary: {str(e)}")
            logger.error(f"Full error: {traceback.format_exc()}")
            raise

    async def update_main_glossary(self, source_lang: str, target_lang: str, new_terms: List[tuple]) -> dict:
        """更新主术语表，合并新旧术语"""
        try:
            # 开始事务
            transaction = self.db.begin_nested()
            
            try:
                # 标准化语言代码
                source_lang = self._normalize_lang_code(source_lang)
                target_lang = self._normalize_lang_code(target_lang)

                # 从数据库获取现有术语表
                existing_glossary = self.db.query(Glossary).filter(
                    Glossary.source_lang == source_lang,
                    Glossary.target_lang == target_lang
                ).first()

                # 获取现有术语
                existing_terms = []
                if existing_glossary:
                    existing_terms = [(entry.source_term, entry.target_term) 
                                    for entry in existing_glossary.entries]

                # 合并新旧术语
                all_terms = set(existing_terms).union(set(new_terms))
                logger.info(f"Combined {len(existing_terms)} existing terms with {len(new_terms)} new terms")

                # 验证大小
                merged_entries = '\n'.join(f"{source}\t{target}" for source, target in sorted(all_terms))
                if len(merged_entries.encode('utf-8')) > 10 * 1024 * 1024:
                    raise ValueError("Merged glossary exceeds size limit (10MB)")

                # 创建新的术语表 payload
                glossary_name = self._get_main_glossary_name(source_lang, target_lang)
                new_glossary_payload = {
                    "name": glossary_name,
                    "dictionaries": [{
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "entries": merged_entries,
                        "entries_format": "tsv"
                    }]
                }

                # 创建新的 DeepL 术语表
                new_deepl_glossary = await self.create_glossary(new_glossary_payload)

                # 更新数据库
                if existing_glossary:
                    # 删除旧的 DeepL 术语表
                    old_deepl_id = existing_glossary.deepl_glossary_id
                    try:
                        await self.delete_glossary(old_deepl_id)
                    except Exception as e:
                        logger.warning(f"Failed to delete old DeepL glossary: {str(e)}")

                    # 更新数据库中的术语表记录
                    existing_glossary.deepl_glossary_id = new_deepl_glossary["glossary_id"]
                    existing_glossary.updated_at = datetime.now()

                    # 清除旧术语
                    self.db.query(GlossaryEntry).filter(
                        GlossaryEntry.glossary_id == existing_glossary.id
                    ).delete()
                else:
                    # 创建新的数据库记录
                    existing_glossary = Glossary(
                        deepl_glossary_id=new_deepl_glossary["glossary_id"],
                        name=glossary_name,
                        source_lang=source_lang,
                        target_lang=target_lang
                    )
                    self.db.add(existing_glossary)
                    self.db.flush()  # 获取 glossary_id

                # 添加新术语
                for source, target in all_terms:
                    entry = GlossaryEntry(
                        glossary_id=existing_glossary.id,
                        source_term=source,
                        target_term=target
                    )
                    self.db.add(entry)

                self.db.commit()
                logger.info(f"Successfully updated glossary with {len(all_terms)} terms")
                return new_deepl_glossary

            except Exception as e:
                transaction.rollback()
                logger.error(f"Transaction rolled back: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error in update_main_glossary: {str(e)}")
            logger.error(f"Full error: {traceback.format_exc()}")
            self.db.rollback()
            raise

    async def cleanup_duplicate_glossaries(self, source_lang: str, target_lang: str) -> None:
        """清理同语言对的非主术语表"""
        try:
            main_glossary_name = self._get_main_glossary_name(source_lang, target_lang)
            glossaries = await self.list_glossaries()
            
            for glossary in glossaries:
                # 检查是否是同语言对的非主术语表
                if (glossary["name"] != main_glossary_name and
                    any(d["source_lang"] == source_lang.upper() and 
                        d["target_lang"] == target_lang.upper() 
                        for d in glossary["dictionaries"])):
                    await self.delete_glossary(glossary["glossary_id"])
                    
        except Exception as e:
            logger.error(f"Error in cleanup_duplicate_glossaries: {str(e)}")

    async def _get_entries_from_api(self, glossary_id: str) -> List[tuple]:
        """从 API 获取术语表条目"""
        try:
            entries_text = await self.get_entries(glossary_id)
            entries = []
            for line in entries_text.split('\n'):
                if line.strip() and line != "placeholder\tplaceholder":
                    parts = line.split('\t')
                    if len(parts) == 2:
                        entries.append(tuple(map(str.strip, parts)))
            return entries
        except Exception as e:
            logger.error(f"Failed to get entries from API: {str(e)}")
            return []

# 术语表同步管理器
class GlossarySyncManager:
    def __init__(self, glossary_manager):
        self.glossary_manager = glossary_manager
        self.sync_interval = 3600  # 1小时同步一次

    async def sync_glossaries(self):
        """同步所有术语表"""
        while True:
            try:
                # 获取所有术语表
                glossaries = await self.glossary_manager.list_glossaries()
                
                for glossary in glossaries:
                    # 更新本地缓存
                    cached_data = self.glossary_manager.get_cached_glossary(
                        glossary["glossary_id"]
                    )
                    if cached_data:
                        cached_data["last_sync"] = datetime.now().isoformat()
                        self.glossary_manager._cache_glossary(
                            glossary["glossary_id"], 
                            cached_data
                        )
                
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Glossary sync error: {str(e)}")
                await asyncio.sleep(60)  # 错误后等待1分钟再试

async def get_document_result(self, document_id: str, document_key: str) -> bytes:
    """获取翻译结果"""
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"DeepL-Auth-Key {self.api_key}"
        }
        response = await client.get(
            f"{self.api_url}/document/{document_id}/result",
            headers=headers,
            params={"document_key": document_key}
        )
        if response.status_code != 200:
            error_msg = response.text
            logger.error(f"DeepL API result retrieval error: {error_msg}")
            raise ValueError(f"DeepL API error: {error_msg}")
        return response.content



