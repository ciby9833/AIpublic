# 文件：backend/services/paper_analyzer/rag_retriever.py    实现RAG检索
from typing import List, Optional, TypedDict, Dict, Union, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import platform
from huggingface_hub import snapshot_download
import os
import torch
import uuid
from sqlalchemy.orm import Session
from datetime import datetime
from dataclasses import dataclass
import json
import logging
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class IndexConfig:
    """向量索引配置类 - 配置化管理索引参数"""
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimension: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunk_overlap: float = 0.3
    similarity_threshold: float = 0.7
    index_type: str = "hierarchical"  # "flat", "hierarchical", "ivf"
    use_gpu: bool = False
    cache_embeddings: bool = True
    batch_size: int = 32
    max_tokens_per_chunk: int = 400

@dataclass 
class RetrieversConfig:
    """检索器配置类 - 配置化管理检索参数"""
    top_k: int = 5
    max_tokens: int = 4000
    rerank_enabled: bool = True
    diversity_threshold: float = 0.8
    context_window: int = 8192
    merge_strategy: str = "weighted"  # "simple", "weighted", "hierarchical"
    relevance_threshold: float = 0.6
    max_context_length: int = 6000

class ChunkInfo(TypedDict):
    line_number: int
    content: str
    page: int
    start_pos: int
    end_pos: int
    is_scanned: bool
    similarity: float
    document_id: Optional[str]  # 添加文档ID
    document_name: Optional[str]  # 添加文档名称

class ContextInfo(TypedDict):
    chunks: List[ChunkInfo]
    total_chunks: int
    question_embedding: Optional[List[float]]
    text: Optional[str]  # 添加合并后的文本
    document_id: Optional[str]  # 文档ID
    document_name: Optional[str]  # 文档名称

class MultiDocContextInfo(TypedDict):
    documents: List[Dict[str, Any]]  # 多个文档的上下文信息
    text: str  # 合并后的文本
    chunks: List[ChunkInfo]  # 所有文档的分块信息
    total_chunks: int

class RAGRetriever:
    def __init__(self, db=None, index_config: IndexConfig = None, retrievers_config: RetrieversConfig = None):
        # 使用配置初始化
        self.db = db
        self.index_config = index_config or IndexConfig()
        self.retrievers_config = retrievers_config or RetrieversConfig()
        
        logger.info(f"[RAG_INIT] 初始化RAG检索器 - 模型: {self.index_config.model_name}, 索引类型: {self.index_config.index_type}")
        
        # 使用新的模型加载方式
        self.device = 'cuda' if self.index_config.use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"[RAG_DEVICE] 使用设备: {self.device}")
        
        try:
            self.model = SentenceTransformer(self.index_config.model_name, device=self.device)
            logger.info(f"[RAG_MODEL_LOADED] 模型加载成功 - 维度: {self.index_config.dimension}")
        except Exception as e:
            logger.error(f"[RAG_MODEL_ERROR] 模型加载失败: {str(e)}")
            # 降级到CPU模式
            self.device = 'cpu'
            self.model = SentenceTransformer(self.index_config.model_name, device=self.device)
            logger.info(f"[RAG_MODEL_FALLBACK] 降级到CPU模式加载成功")
        
        # 使用配置创建索引
        self.indexes = {}  # 存储每个文档的索引
        self.text_chunks = {}  # 存储文档的文本块
        self.document_metadata = {}  # 存储文档元数据
        
        logger.info(f"[RAG_CONFIG] 检索配置 - top_k: {self.retrievers_config.top_k}, 阈值: {self.retrievers_config.relevance_threshold}")

    def _create_index(self, dimension: int):
        """根据配置创建索引"""
        if self.index_config.index_type == "hierarchical":
            return self._create_hierarchical_index(dimension, 1000)
        elif self.index_config.index_type == "ivf":
            return self._create_ivf_index(dimension)
        else:
            # 默认平面索引
            return faiss.IndexFlatIP(dimension)

    def _create_hierarchical_index(self, dimension, data_size):
        """创建分层索引 - 优化大数据集检索"""
        if data_size > 10000:
            logger.info(f"[RAG_INDEX] 创建HNSW索引用于大数据集 - 数据量: {data_size}")
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
        else:
            logger.info(f"[RAG_INDEX] 创建LSH索引用于中等数据集 - 数据量: {data_size}")
            index = faiss.IndexLSH(dimension, 64)
        return index

    def _create_ivf_index(self, dimension: int):
        """创建IVF索引 - 用于超大数据集"""
        logger.info(f"[RAG_INDEX] 创建IVF索引")
        nlist = 100  # 聚类中心数量
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        return index

    def _split_text(self, text: str) -> List[str]:
        """使用配置化参数分割文本"""
        chunk_size = self.index_config.chunk_size
        overlap = self.index_config.chunk_overlap
        
        logger.debug(f"[RAG_SPLIT] 分割文本 - 块大小: {chunk_size}, 重叠: {overlap}")
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            # 如果不是最后一块，尝试在句子边界分割
            if end < text_length:
                # 寻找最近的句号、换行符或空格
                for i in range(min(100, chunk_size // 4)):
                    if end - i > start and text[end - i] in '。\n ':
                        end = end - i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 20:  # 过滤太短的块
                chunks.append(chunk)
            
            # 计算下一个开始位置，考虑重叠
            start = end - overlap if end < text_length else text_length
        
        logger.info(f"[RAG_SPLIT_COMPLETE] 文本分割完成 - 生成 {len(chunks)} 个文本块")
        return chunks

    async def build_index(self, content: str, paper_id: str, line_mapping: dict = None, db=None, structured_data: dict = None):
        """构建文档索引并存储到数据库 - 使用配置化管理"""
        try:
            logger.info(f"[RAG_BUILD_INDEX] 开始构建索引 - 文档ID: {paper_id}, 内容长度: {len(content)} 字符")
            
            # 使用传入的db连接，或者实例的db连接
            current_db = db or self.db
            if not current_db:
                raise ValueError("Database connection required for index building")
            
            # 使用配置化文本分割
            chunks = self._split_text(content)
            logger.info(f"[RAG_BUILD_CHUNKS] 文本分割完成 - 生成 {len(chunks)} 个文本块")
            
            # 批量生成嵌入向量
            embeddings = await self._generate_embeddings_batch(chunks)
            logger.info(f"[RAG_BUILD_EMBEDDINGS] 嵌入向量生成完成 - 形状: {embeddings.shape}")
            
            # 创建并配置索引
            dimension = embeddings.shape[1]
            index = self._create_hierarchical_index(dimension, len(chunks))
            
            # IVF索引需要训练
            if hasattr(index, 'train') and hasattr(index, 'is_trained') and not index.is_trained:
                logger.info(f"[RAG_BUILD_TRAIN] 训练IVF索引")
                index.train(embeddings)
            
            # 添加向量到索引
            index.add(embeddings)
            logger.info(f"[RAG_BUILD_ADD] 向量添加到索引完成")
            
            # 存储到新的数据结构
            self.indexes[paper_id] = index
            self.text_chunks[paper_id] = chunks
            self.document_metadata[paper_id] = {
                'line_mapping': line_mapping or {},
                'structured_data': structured_data,
                'chunk_count': len(chunks),
                'embedding_dimension': dimension,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # 存储到数据库
            await self._store_index_to_database(current_db, paper_id, chunks, embeddings, structured_data)
            logger.info(f"[RAG_BUILD_SUCCESS] 索引构建完成 - 文档ID: {paper_id}")
            
        except Exception as e:
            logger.error(f"[RAG_BUILD_ERROR] 索引构建失败: {str(e)}")
            if 'current_db' in locals() and current_db:
                current_db.rollback()
            raise Exception(f"Index building error: {str(e)}")

    async def _generate_embeddings_batch(self, chunks: List[str]) -> np.ndarray:
        """批量生成嵌入向量 - 优化性能"""
        logger.debug(f"[RAG_EMBEDDINGS] 开始批量生成嵌入向量 - 块数: {len(chunks)}")
        
        batch_size = self.index_config.batch_size
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logger.debug(f"[RAG_EMBEDDINGS_BATCH] 处理批次 {i//batch_size + 1} - 大小: {len(batch)}")
            
            with torch.no_grad():
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings).astype('float32')
        logger.debug(f"[RAG_EMBEDDINGS_COMPLETE] 嵌入向量生成完成 - 形状: {embeddings.shape}")
        return embeddings

    async def _store_index_to_database(self, db, paper_id: str, chunks: List[str], embeddings: np.ndarray, structured_data: dict = None):
        """存储索引数据到数据库 - 增强的存储策略"""
        try:
            from models.paper import PaperAnalysis
            
            # 将嵌入向量转换为列表
            embeddings_list = embeddings.tolist()
            
            # 增加事务重试逻辑
            max_retries = 3
            for retry in range(max_retries):
                try:
                    paper = db.query(PaperAnalysis).filter(
                        PaperAnalysis.paper_id == uuid.UUID(paper_id)
                    ).first()
                    
                    if paper:
                        paper.documents = chunks
                        paper.embeddings = embeddings_list
                        paper.index_built = True
                        if structured_data:
                            paper.structured_data = structured_data
                        
                        db.commit()
                        logger.info(f"[RAG_STORE_SUCCESS] 索引存储到数据库成功 - 重试次数: {retry}")
                        return
                    
                except Exception as retry_error:
                    logger.warning(f"[RAG_STORE_RETRY] 存储重试 {retry + 1}/{max_retries}: {str(retry_error)}")
                    db.rollback()
                    if retry == max_retries - 1:
                        raise
                    await asyncio.sleep(0.5 * (retry + 1))  # 指数退避
            
        except Exception as e:
            logger.error(f"[RAG_STORE_ERROR] 存储索引到数据库失败: {str(e)}")
            raise

    async def schedule_index_building(self, content: str, paper_id: str, line_mapping: dict = None, db=None, structured_data: dict = None):
        """异步调度索引构建任务"""
        try:
            # 记录索引构建请求
            print(f"Scheduling async index building for paper {paper_id}")
            
            # 方法1：使用背景任务（如果环境支持）
            if hasattr(self, 'background_tasks'):
                self.background_tasks.add_task(
                    self.build_index,
                    content=content,
                    paper_id=paper_id,
                    line_mapping=line_mapping,
                    db=db,
                    structured_data=structured_data
                )
                return {"status": "scheduled", "paper_id": paper_id}
            
            # 方法2：使用异步任务（如果有Celery）
            try:
                from celery import Celery
                try:
                    app = Celery('paper_analyzer')
                    task_id = app.send_task(
                        'services.paper_analyzer.tasks.build_index_task',
                        args=[content, paper_id, line_mapping, structured_data]
                    )
                    return {"status": "scheduled", "paper_id": paper_id, "task_id": str(task_id)}
                except Exception as celery_error:
                    print(f"Celery task creation failed: {str(celery_error)}")
                    # 降级到直接调用
            except ImportError:
                print("Celery not available, falling back to direct execution")
            
            # 降级：直接构建，但在异步函数中执行以不阻塞主请求
            import asyncio
            asyncio.create_task(
                self.build_index(
                    content=content,
                    paper_id=paper_id,
                    line_mapping=line_mapping,
                    db=db,
                    structured_data=structured_data
                )
            )
            return {"status": "started", "paper_id": paper_id}
            
        except Exception as e:
            print(f"Failed to schedule index building: {str(e)}")
            return {"status": "error", "paper_id": paper_id, "error": str(e)}

    async def get_relevant_context(self, question: str, paper_id: str, history: str = None, doc_limit: int = 5) -> dict:
        """获取相关上下文 - 使用配置化检索，增强动态阈值调整"""
        try:
            logger.info(f"[RAG_CONTEXT] 开始获取相关上下文 - 文档ID: {paper_id}, 问题长度: {len(question)} 字符")
            logger.debug(f"[RAG_CONTEXT_QUERY] 查询内容: {question[:100]}...")
            
            # 如果有历史对话，将其与问题组合
            search_query = f"{history}\n{question}" if history else question
            
            # 确保索引已加载 - 适应新的数据结构
            await self._ensure_paper_index(paper_id)
            
            # 检查索引和文本块是否存在
            if paper_id not in self.indexes or paper_id not in self.text_chunks:
                raise Exception(f"Index or text chunks not found for paper {paper_id}")
            
            index = self.indexes[paper_id]
            chunks = self.text_chunks[paper_id]
            metadata = self.document_metadata.get(paper_id, {})
            line_mapping = metadata.get('line_mapping', {})
            
            logger.debug(f"[RAG_CONTEXT_INFO] 索引信息 - 文档块数: {len(chunks)}, 请求限制: {doc_limit}")
            
            # 生成问题的嵌入向量
            with torch.no_grad():
                question_embedding = self.model.encode([search_query], convert_to_numpy=True)[0]
            question_embedding = question_embedding.reshape(1, -1).astype('float32')
            
            # 使用配置的top_k参数，但不超过请求限制
            k = min(doc_limit * 3, len(chunks), self.retrievers_config.top_k * 2)  # 增加搜索范围
            logger.debug(f"[RAG_CONTEXT_SEARCH] 执行相似度搜索 - k: {k}")
            
            # 使用 FAISS 进行相似度搜索
            distances, indices = index.search(question_embedding, k)
            
            # 记录原始搜索结果
            logger.debug(f"[RAG_CONTEXT_RAW] 原始搜索结果 - 索引数量: {len(indices[0])}, 距离范围: {distances[0].min():.3f}-{distances[0].max():.3f}")
            
            # 动态计算阈值
            base_threshold = self.retrievers_config.relevance_threshold
            similarities = []
            
            # 先计算所有相似度
            for i, idx in enumerate(indices[0]):
                idx = int(idx)
                if idx >= len(chunks):
                    continue
                
                # 获取原始距离值
                raw_distance = float(distances[0][i])
                
                # 智能相似度计算 - 根据索引类型和距离特征调整
                similarity = self._calculate_smart_similarity(index, raw_distance)
                
                similarities.append((idx, similarity))
                
                # 调试日志：记录原始距离和转换后的相似度
                if i < 3:  # 只记录前3个结果
                    logger.debug(f"[RAG_SIMILARITY_CONVERT] 块#{idx}: 原始距离={raw_distance:.3f}, 转换相似度={similarity:.3f}")
            
            # 排序并分析相似度分布
            similarities.sort(key=lambda x: x[1], reverse=True)
            logger.debug(f"[RAG_CONTEXT_SIMILARITIES] 相似度分布 - 最高: {similarities[0][1]:.3f}, 最低: {similarities[-1][1]:.3f}")
            
            # 动态调整阈值策略
            if similarities:
                max_similarity = similarities[0][1]
                min_similarity = similarities[-1][1]
                
                # 如果最高相似度都低于基础阈值，降低阈值
                if max_similarity < base_threshold:
                    # 使用最高相似度的80%作为动态阈值，但不低于0.3
                    dynamic_threshold = max(0.3, max_similarity * 0.8)
                    logger.warning(f"[RAG_CONTEXT_DYNAMIC_THRESHOLD] 最高相似度{max_similarity:.3f}低于基础阈值{base_threshold:.3f}，使用动态阈值: {dynamic_threshold:.3f}")
                else:
                    dynamic_threshold = base_threshold
                    logger.debug(f"[RAG_CONTEXT_STATIC_THRESHOLD] 使用标准阈值: {dynamic_threshold:.3f}")
            else:
                dynamic_threshold = 0.3  # 保底阈值
                logger.warning(f"[RAG_CONTEXT_FALLBACK_THRESHOLD] 无相似度结果，使用保底阈值: {dynamic_threshold:.3f}")
            
            # 应用动态阈值过滤
            relevant_chunks = []
            relevant_text = ""
            filtered_count = 0
            
            for idx, similarity in similarities:
                
                # 应用动态阈值过滤
                if similarity < dynamic_threshold:
                    filtered_count += 1
                    logger.debug(f"[RAG_CONTEXT_FILTER] 跳过低相似度块 #{idx} - 相似度: {similarity:.3f} < 阈值: {dynamic_threshold:.3f}")
                    continue
                
                # 限制结果数量
                if len(relevant_chunks) >= doc_limit:
                    logger.debug(f"[RAG_CONTEXT_LIMIT] 已达到结果数量限制: {doc_limit}")
                    break
                
                chunk = chunks[idx]
                
                # 安全获取行信息
                line_info = {}
                if isinstance(line_mapping, dict):
                    str_key = str(idx)
                    int_key = idx
                    if str_key in line_mapping:
                        line_info = line_mapping[str_key]
                    elif int_key in line_mapping:
                        line_info = line_mapping[int_key]
                
                # 构建文本
                relevant_text += f"\n\n{chunk}"
                
                relevant_chunks.append({
                    "line_number": idx,
                    "content": str(chunk),
                    "page": int(line_info.get("page", 1)),
                    "start_pos": int(line_info.get("start_pos", 0)),
                    "end_pos": int(line_info.get("end_pos", 0)),
                    "is_scanned": bool(line_info.get("is_scanned", False)),
                    "similarity": similarity,
                    "document_id": str(paper_id),
                    "document_name": metadata.get('filename', '未知文档')
                })
                
                logger.debug(f"[RAG_CONTEXT_INCLUDED] 包含块 #{idx} - 相似度: {similarity:.3f}")
            
            logger.info(f"[RAG_CONTEXT_SUCCESS] 上下文检索完成 - 返回 {len(relevant_chunks)} 个相关块（过滤掉 {filtered_count} 个低相似度块）")
            
            # 如果仍然没有结果，提供最相似的几个块作为fallback
            if len(relevant_chunks) == 0 and similarities:
                logger.warning(f"[RAG_CONTEXT_FALLBACK] 没有通过阈值的块，提供最相似的 {min(2, len(similarities))} 个块作为fallback")
                for i in range(min(2, len(similarities))):
                    idx, similarity = similarities[i]
                    chunk = chunks[idx]
                    
                    # 安全获取行信息
                    line_info = {}
                    if isinstance(line_mapping, dict):
                        str_key = str(idx)
                        int_key = idx
                        if str_key in line_mapping:
                            line_info = line_mapping[str_key]
                        elif int_key in line_mapping:
                            line_info = line_mapping[int_key]
                    
                    relevant_text += f"\n\n{chunk}"
                    
                    relevant_chunks.append({
                        "line_number": idx,
                        "content": str(chunk),
                        "page": int(line_info.get("page", 1)),
                        "start_pos": int(line_info.get("start_pos", 0)),
                        "end_pos": int(line_info.get("end_pos", 0)),
                        "is_scanned": bool(line_info.get("is_scanned", False)),
                        "similarity": similarity,
                        "document_id": str(paper_id),
                        "document_name": metadata.get('filename', '未知文档'),
                        "fallback": True  # 标记为fallback结果
                    })
                
                logger.info(f"[RAG_CONTEXT_FALLBACK_COMPLETE] Fallback完成 - 返回 {len(relevant_chunks)} 个块")
            
            return {
                "chunks": relevant_chunks,
                "total_chunks": len(relevant_chunks),
                "question_embedding": question_embedding.tolist(),
                "text": relevant_text.strip(),
                "document_id": paper_id,
                "document_name": metadata.get('filename', '未知文档'),
                "retrieval_config": {
                    "top_k": k,
                    "base_threshold": base_threshold,
                    "used_threshold": dynamic_threshold,
                    "total_available_chunks": len(chunks),
                    "filtered_count": filtered_count,
                    "max_similarity": similarities[0][1] if similarities else 0.0,
                    "min_similarity": similarities[-1][1] if similarities else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"[RAG_CONTEXT_ERROR] 上下文检索失败: {str(e)}")
            import traceback
            logger.debug(f"[RAG_CONTEXT_TRACEBACK] 错误堆栈: {traceback.format_exc()}")
            raise Exception(f"Context retrieval error: {str(e)}")

    async def _ensure_paper_index(self, paper_id: str) -> Dict:
        """确保文档索引已加载 - 适应新的数据结构"""
        # 检查内存中的索引
        if paper_id in self.indexes and paper_id in self.text_chunks:
            logger.debug(f"[RAG_ENSURE_CACHE] 使用缓存索引 - 文档ID: {paper_id}")
            return self.indexes[paper_id]
        
        # 从数据库恢复索引
        current_db = self.db
        if not current_db:
            raise ValueError("Database connection required to retrieve index")
        
        from models.paper import PaperAnalysis
        
        paper = current_db.query(PaperAnalysis).filter(
            PaperAnalysis.paper_id == uuid.UUID(paper_id)
        ).first()
        
        if not paper:
            raise Exception(f"Paper ID {paper_id} not found in database")
        
        if not paper.index_built or not paper.documents or not paper.embeddings:
            # 重新构建索引
            if paper.content:
                logger.info(f"[RAG_ENSURE_REBUILD] 重新构建索引 - 文档ID: {paper_id}")
                await self.build_index(
                    paper.content, 
                    paper_id, 
                    paper.line_mapping, 
                    current_db, 
                    paper.structured_data
                )
                return self.indexes[paper_id]
            else:
                raise Exception(f"Paper content not available for {paper_id}")
        
        try:
            # 恢复索引
            embeddings = np.array(paper.embeddings, dtype='float32')
            documents = paper.documents
            line_mapping = paper.line_mapping or {}
            
            # 创建索引
            dimension = embeddings.shape[1]
            index = self._create_hierarchical_index(dimension, len(documents))
            
            # 训练和添加向量
            if hasattr(index, 'train') and hasattr(index, 'is_trained') and not index.is_trained:
                index.train(embeddings)
            index.add(embeddings)
            
            # 存储到新的数据结构
            self.indexes[paper_id] = index
            self.text_chunks[paper_id] = documents
            self.document_metadata[paper_id] = {
                'line_mapping': line_mapping,
                'structured_data': paper.structured_data,
                'filename': paper.filename,
                'chunk_count': len(documents),
                'embedding_dimension': dimension,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            logger.info(f"[RAG_ENSURE_RESTORED] 索引从数据库恢复成功 - 文档ID: {paper_id}")
            return index
            
        except Exception as e:
            logger.error(f"[RAG_ENSURE_ERROR] 恢复索引失败: {str(e)}")
            raise Exception(f"Failed to restore index: {str(e)}")

    async def get_context_from_multiple_docs(self, question: str, paper_ids: List[str], history: str = None, doc_limit_per_paper: int = 3) -> Union[Dict, List[Dict]]:
        """从多个文档中检索相关上下文
        
        参数:
            question: 用户问题
            paper_ids: 多个文档ID列表
            history: 对话历史
            doc_limit_per_paper: 每个文档检索的分块数量限制
        
        返回:
            多文档合并的上下文信息
        """
        try:
            if not paper_ids:
                raise ValueError("No paper IDs provided")
            
            # 单文档情况，直接调用现有方法
            if len(paper_ids) == 1:
                return await self.get_relevant_context(
                    question=question,
                    paper_id=paper_ids[0],
                    history=history,
                    doc_limit=doc_limit_per_paper
                )
            
            # 多文档情况
            search_query = f"{history}\n{question}" if history else question
            
            # 生成问题的嵌入向量
            with torch.no_grad():
                question_embedding = self.model.encode([search_query], convert_to_numpy=True)[0]
            question_embedding = question_embedding.reshape(1, -1).astype('float32')
            
            # 从每个文档中检索上下文
            all_doc_contexts = []
            all_chunks = []
            combined_text = ""
            
            for paper_id in paper_ids:
                try:
                    # 获取单文档上下文
                    single_context = await self.get_relevant_context(
                        question=question,
                        paper_id=paper_id,
                        history=history,
                        doc_limit=doc_limit_per_paper
                    )
                    
                    # 构建多文档合并信息
                    if single_context and "chunks" in single_context and len(single_context["chunks"]) > 0:
                        all_doc_contexts.append({
                            "document_id": paper_id,
                            "document_name": single_context.get("document_name", "未知文档"),
                            "chunks": single_context["chunks"],
                            "text": single_context.get("text", "")
                        })
                        
                        # 添加到全局分块集合
                        all_chunks.extend(single_context["chunks"])
                        
                        # 添加到合并文本，包含文档来源
                        doc_text = single_context.get("text", "")
                        if doc_text:
                            doc_name = single_context.get("document_name", f"文档 {paper_id}")
                            combined_text += f"\n\n【文档: {doc_name}】\n{doc_text}"
                            
                except Exception as doc_err:
                    print(f"Error retrieving context from document {paper_id}: {str(doc_err)}")
                    # 继续处理其他文档，不中断整体流程
            
            # 对所有块按相似度排序
            all_chunks = sorted(all_chunks, key=lambda x: x.get("similarity", 0), reverse=True)
            
            # 如果没有获取到任何上下文，返回空结果
            if not all_chunks:
                return {
                    "chunks": [],
                    "total_chunks": 0,
                    "text": "",
                    "documents": []
                }
            
            # 返回多文档合并结果
            return {
                "chunks": all_chunks[:10],  # 限制最多返回10个最相关的块
                "total_chunks": len(all_chunks),
                "text": combined_text.strip(),
                "documents": all_doc_contexts,
                "question_embedding": question_embedding.tolist()
            }
            
        except Exception as e:
            print(f"Multiple document context retrieval error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise Exception(f"Multiple document context retrieval error: {str(e)}")
    
    async def search_across_all_papers(self, query: str, user_id: str = None, limit: int = 20) -> Dict:
        """
        在用户的所有文档中搜索相关内容
        
        参数:
            query: 搜索查询
            user_id: 用户ID，如果提供则只搜索该用户的文档
            limit: 结果数量限制
            
        返回:
            匹配结果列表，包含文档信息和匹配内容
        """
        try:
            current_db = self.db
            if not current_db:
                raise ValueError("Database connection required for search")
            
            from models.paper import PaperAnalysis
            
            # 构建查询
            paper_query = current_db.query(PaperAnalysis)
            if user_id:
                paper_query = paper_query.filter(PaperAnalysis.user_id == uuid.UUID(user_id))
            
            # 只查询已建立索引的文档
            paper_query = paper_query.filter(PaperAnalysis.index_built == True)
            
            # 获取所有文档
            papers = paper_query.all()
            
            # 如果没有文档，返回空结果
            if not papers:
                return {
                    "results": [],
                    "total": 0,
                    "query": query
                }
            
            # 生成查询的嵌入向量
            with torch.no_grad():
                query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # 在每个文档中搜索
            all_results = []
            
            for paper in papers:
                paper_id = str(paper.paper_id)
                
                try:
                    # 确保索引已加载
                    paper_data = await self._ensure_paper_index(paper_id)
                    
                    # 获取索引
                    index = paper_data.get("index")
                    documents = paper_data.get("documents", [])
                    
                    if not index or not documents:
                        continue
                    
                    # 搜索相似内容
                    k = min(5, len(documents))  # 每个文档最多5个结果
                    distances, indices = index.search(query_embedding.reshape(1, -1), k)
                    
                    # 添加结果
                    for i, idx in enumerate(indices[0]):
                        idx = int(idx)
                        if idx < len(documents):
                            similarity = float(1 - distances[0][i])
                            if similarity > 0.5:  # 只保留相似度较高的结果
                                all_results.append({
                                    "paper_id": paper_id,
                                    "filename": paper.filename,
                                    "content": documents[idx],
                                    "chunk_index": idx,
                                    "similarity": similarity
                                })
                except Exception as paper_err:
                    print(f"Error searching in document {paper_id}: {str(paper_err)}")
                    # 继续处理其他文档
            
            # 按相似度排序
            all_results = sorted(all_results, key=lambda x: x.get("similarity", 0), reverse=True)
            
            # 限制结果数量
            all_results = all_results[:limit]
            
            return {
                "results": all_results,
                "total": len(all_results),
                "query": query
            }
            
        except Exception as e:
            print(f"Cross-document search error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise Exception(f"Cross-document search error: {str(e)}")

    def _batch_search(self, index, query_vector, k, batch_size=100):
        # 需要实现分批检索逻辑
        results = []
        # 实现分批检索，合并结果
        return results

    def _calculate_smart_similarity(self, index, raw_distance):
        """智能相似度计算 - 根据索引类型和距离特征自动调整
        
        参数:
            index: FAISS索引对象
            raw_distance: 原始距离值
            
        返回:
            标准化的相似度值 (0-1之间，越大越相似)
        """
        try:
            logger.debug(f"[SIMILARITY_CALC] 原始距离: {raw_distance}, 索引类型: {type(index).__name__}")
            
            # 检查索引类型和度量类型
            index_type = type(index).__name__
            
            # 1. 内积索引 (IndexFlatIP, IndexIVFFlat with METRIC_INNER_PRODUCT)
            if hasattr(index, 'metric_type') and index.metric_type == faiss.METRIC_INNER_PRODUCT:
                # 内积越大相似度越高，但需要规范化
                # 通常内积值在[-1, 1]范围内（对于单位向量）
                similarity = max(0.0, min(1.0, (raw_distance + 1) / 2))
                logger.debug(f"[SIMILARITY_CALC] 内积索引 - 相似度: {similarity:.3f}")
                return similarity
            
            # 2. LSH索引 (IndexLSH)
            elif 'LSH' in index_type:
                # LSH返回的是汉明距离，值越小越相似
                # 对于64位LSH，最大距离是64
                max_distance = 64.0
                # 转换为相似度：距离越小，相似度越高
                if raw_distance < 0:
                    # 处理异常负值
                    logger.warning(f"[SIMILARITY_CALC] LSH异常负距离: {raw_distance}, 使用绝对值")
                    raw_distance = abs(raw_distance)
                
                similarity = max(0.0, 1.0 - (raw_distance / max_distance))
                logger.debug(f"[SIMILARITY_CALC] LSH索引 - 汉明距离: {raw_distance}, 相似度: {similarity:.3f}")
                return similarity
            
            # 3. HNSW索引 (IndexHNSWFlat)
            elif 'HNSW' in index_type:
                # HNSW默认使用L2距离，值越小越相似
                # L2距离范围通常是[0, +∞)，需要转换
                if raw_distance < 0:
                    logger.warning(f"[SIMILARITY_CALC] HNSW异常负距离: {raw_distance}, 使用绝对值")
                    raw_distance = abs(raw_distance)
                
                # 使用指数衰减函数转换：sim = exp(-distance)
                similarity = np.exp(-raw_distance)
                logger.debug(f"[SIMILARITY_CALC] HNSW索引 - L2距离: {raw_distance}, 相似度: {similarity:.3f}")
                return similarity
            
            # 4. 平面索引 (IndexFlat, IndexFlatL2)
            elif 'Flat' in index_type:
                # 检查是否为L2距离
                if hasattr(index, 'metric_type') and index.metric_type == faiss.METRIC_L2:
                    # L2距离，值越小越相似
                    if raw_distance < 0:
                        logger.warning(f"[SIMILARITY_CALC] L2异常负距离: {raw_distance}, 使用绝对值")
                        raw_distance = abs(raw_distance)
                    
                    # 使用指数衰减：sim = exp(-distance/2)
                    similarity = np.exp(-raw_distance / 2.0)
                    logger.debug(f"[SIMILARITY_CALC] Flat L2索引 - 距离: {raw_distance}, 相似度: {similarity:.3f}")
                    return similarity
                else:
                    # 默认假设为L2距离
                    if raw_distance < 0:
                        raw_distance = abs(raw_distance)
                    similarity = max(0.0, 1.0 - raw_distance / 2.0)  # 简单线性转换
                    logger.debug(f"[SIMILARITY_CALC] Flat默认索引 - 距离: {raw_distance}, 相似度: {similarity:.3f}")
                    return similarity
            
            # 5. IVF索引 (IndexIVF*)
            elif 'IVF' in index_type:
                # IVF索引默认使用L2距离
                if raw_distance < 0:
                    logger.warning(f"[SIMILARITY_CALC] IVF异常负距离: {raw_distance}, 使用绝对值")
                    raw_distance = abs(raw_distance)
                
                similarity = np.exp(-raw_distance / 2.0)
                logger.debug(f"[SIMILARITY_CALC] IVF索引 - 距离: {raw_distance}, 相似度: {similarity:.3f}")
                return similarity
            
            # 6. 未知索引类型 - 使用通用处理
            else:
                logger.warning(f"[SIMILARITY_CALC] 未知索引类型: {index_type}, 使用通用处理")
                
                # 处理异常值
                if raw_distance < 0:
                    logger.warning(f"[SIMILARITY_CALC] 通用处理异常负距离: {raw_distance}")
                    if raw_distance < -10:  # 极端异常值
                        # 可能是LSH的汉明距离，取绝对值并按LSH处理
                        raw_distance = abs(raw_distance)
                        similarity = max(0.0, 1.0 - (raw_distance / 64.0))
                    else:
                        # 小的负值，可能是数值误差，直接设为高相似度
                        similarity = 0.9
                elif raw_distance > 100:  # 极大的正值
                    # 可能是未归一化的距离
                    similarity = 0.1
                else:
                    # 常规处理：假设为L2距离
                    similarity = max(0.0, 1.0 - raw_distance / 10.0)
                
                logger.debug(f"[SIMILARITY_CALC] 通用处理 - 距离: {raw_distance}, 相似度: {similarity:.3f}")
                return similarity
            
        except Exception as e:
            logger.error(f"[SIMILARITY_CALC_ERROR] 相似度计算失败: {str(e)}, 原始距离: {raw_distance}")
            # 返回一个安全的默认值
            return 0.5

    async def build_web_index(self, web_content: dict, web_id: str, db=None) -> bool:
        """为网页内容构建向量索引 - 新增方法"""
        try:
            logger.info(f"[WEB_INDEX_BUILD] 开始构建网页索引 - URL: {web_content.get('url', '')}")
            
            # 提取文本内容和分块
            content = web_content.get("content", "")
            chunks = web_content.get("chunks", [])
            
            if not content or not chunks:
                logger.warning(f"[WEB_INDEX_EMPTY] 网页内容为空，跳过索引构建")
                return False
            
            # 使用现有的分块或重新分块
            if chunks:
                chunk_texts = [chunk.get("content", "") for chunk in chunks]
            else:
                chunk_texts = self._split_text(content)
            
            # 过滤空分块
            chunk_texts = [chunk for chunk in chunk_texts if chunk.strip()]
            
            if not chunk_texts:
                logger.warning(f"[WEB_INDEX_NO_CHUNKS] 没有有效的文本分块")
                return False
            
            logger.info(f"[WEB_INDEX_CHUNKS] 网页分块数量: {len(chunk_texts)}")
            
            # 生成embeddings
            embeddings = await self._generate_embeddings_batch(chunk_texts)
            
            if embeddings is None or embeddings.size == 0:
                logger.error(f"[WEB_INDEX_EMBEDDINGS_FAIL] Embeddings生成失败")
                return False
            
            # 存储到数据库
            await self._store_web_index_to_database(
                db, web_id, chunk_texts, embeddings, web_content
            )
            
            logger.info(f"[WEB_INDEX_SUCCESS] 网页索引构建成功 - ID: {web_id}")
            return True
            
        except Exception as e:
            logger.error(f"[WEB_INDEX_ERROR] 网页索引构建失败: {str(e)}")
            import traceback
            logger.debug(f"[WEB_INDEX_TRACEBACK] {traceback.format_exc()}")
            return False

    async def _store_web_index_to_database(self, db, web_id: str, chunks: List[str], 
                                          embeddings: np.ndarray, web_content: dict):
        """存储网页索引到数据库 - 新增方法"""
        try:
            logger.info(f"[WEB_DB_STORE] 开始存储网页索引到数据库 - ID: {web_id}")
            
            from models.paper import PaperAnalysis
            import uuid
            from datetime import datetime
            
            # 检查是否已存在
            existing_web = db.query(PaperAnalysis).filter(
                PaperAnalysis.paper_id == uuid.UUID(web_id)
            ).first()
            
            if existing_web:
                logger.info(f"[WEB_DB_UPDATE] 更新现有网页记录")
                web_record = existing_web
            else:
                logger.info(f"[WEB_DB_CREATE] 创建新网页记录")
                web_record = PaperAnalysis(
                    paper_id=uuid.UUID(web_id),
                    filename=web_content.get("title", web_content.get("url", "Unknown Web Page")),
                    file_type="web",
                    created_at=datetime.utcnow()
                )
                db.add(web_record)
            
            # 更新内容
            web_record.content = web_content.get("content", "")
            web_record.line_mapping = self._convert_web_chunks_to_line_mapping(
                web_content.get("chunks", [])
            )
            web_record.total_lines = len(chunks)
            web_record.structured_data = web_content.get("structured_data", {})
            web_record.source_url = web_content.get("url")
            web_record.summary = web_content.get("content", "")[:500]  # 前500字符作为摘要
            
            # 存储文档分块和embeddings
            documents_data = []
            embeddings_data = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                documents_data.append({
                    "chunk_index": i,
                    "content": chunk,
                    "start_pos": web_content.get("chunks", [{}])[i].get("start_pos", 0) if i < len(web_content.get("chunks", [])) else 0,
                    "end_pos": web_content.get("chunks", [{}])[i].get("end_pos", len(chunk)) if i < len(web_content.get("chunks", [])) else len(chunk),
                    "source_url": web_content.get("url", ""),
                    "chunk_type": "web"
                })
                embeddings_data.append(embedding.tolist())
            
            web_record.documents = documents_data
            web_record.embeddings = embeddings_data
            web_record.index_built = True
            web_record.updated_at = datetime.utcnow()
            
            db.commit()
            logger.info(f"[WEB_DB_SUCCESS] 网页索引存储成功")
            
        except Exception as e:
            db.rollback()
            logger.error(f"[WEB_DB_ERROR] 网页索引存储失败: {str(e)}")
            raise

    def _convert_web_chunks_to_line_mapping(self, chunks: List[dict]) -> dict:
        """将网页分块转换为line_mapping格式"""
        line_mapping = {}
        
        for i, chunk in enumerate(chunks):
            line_key = str(i + 1)
            line_mapping[line_key] = {
                "content": chunk.get("content", ""),
                "start_pos": chunk.get("start_pos", 0),
                "end_pos": chunk.get("end_pos", 0),
                "page": 1,  # 网页默认为第1页
                "is_scanned": False,
                "is_web_chunk": True,
                "source_url": chunk.get("source_url", ""),
                "chunk_index": chunk.get("chunk_index", i)
            }
        
        return line_mapping

    async def search_web_content(self, query: str, web_ids: List[str] = None, 
                                user_id: str = None, limit: int = 10) -> Dict:
        """在网页内容中搜索 - 新增方法"""
        try:
            logger.info(f"[WEB_SEARCH] 开始网页内容搜索 - 查询: {query[:50]}...")
            
            from models.paper import PaperAnalysis
            
            # 构建查询条件
            query_filter = [PaperAnalysis.file_type == "web"]
            
            if web_ids:
                query_filter.append(PaperAnalysis.paper_id.in_([uuid.UUID(wid) for wid in web_ids]))
            
            if user_id:
                query_filter.append(PaperAnalysis.user_id == uuid.UUID(user_id))
            
            # 获取符合条件的网页记录
            web_records = self.db.query(PaperAnalysis).filter(*query_filter).all()
            
            if not web_records:
                logger.warning(f"[WEB_SEARCH_NO_RECORDS] 没有找到匹配的网页记录")
                return {"results": [], "total": 0, "query": query}
            
            logger.info(f"[WEB_SEARCH_RECORDS] 找到网页记录数: {len(web_records)}")
            
            # 生成查询向量
            query_embedding = await self._generate_embeddings_batch([query])
            if query_embedding is None or query_embedding.size == 0:
                raise Exception("查询向量生成失败")
            
            query_vector = query_embedding[0]
            
            # 在每个网页中搜索
            all_results = []
            for web_record in web_records:
                try:
                    if not web_record.embeddings or not web_record.documents:
                        logger.warning(f"[WEB_SEARCH_NO_INDEX] 网页 {web_record.paper_id} 没有索引")
                        continue
                    
                    # 转换embeddings为numpy数组
                    embeddings_array = np.array(web_record.embeddings, dtype=np.float32)
                    
                    # 计算相似度
                    similarities = np.dot(embeddings_array, query_vector)
                    
                    # 获取top结果
                    top_indices = np.argsort(similarities)[::-1][:limit]
                    
                    for idx in top_indices:
                        similarity = float(similarities[idx])
                        
                        if similarity >= self.retrievers_config.relevance_threshold:
                            document = web_record.documents[idx]
                            
                            result = {
                                "web_id": str(web_record.paper_id),
                                "web_title": web_record.filename,
                                "source_url": web_record.source_url,
                                "content": document.get("content", ""),
                                "similarity": similarity,
                                "chunk_index": document.get("chunk_index", idx),
                                "start_pos": document.get("start_pos", 0),
                                "end_pos": document.get("end_pos", 0),
                                "chunk_type": "web"
                            }
                            
                            all_results.append(result)
                            
                except Exception as record_error:
                    logger.error(f"[WEB_SEARCH_RECORD_ERROR] 处理网页记录失败 {web_record.paper_id}: {str(record_error)}")
                    continue
            
            # 按相似度排序并限制结果数量
            all_results.sort(key=lambda x: x["similarity"], reverse=True)
            final_results = all_results[:limit]
            
            logger.info(f"[WEB_SEARCH_SUCCESS] 网页搜索完成 - 找到结果: {len(final_results)}")
            
            return {
                "results": final_results,
                "total": len(final_results),
                "query": query,
                "search_type": "web_content"
            }
            
        except Exception as e:
            logger.error(f"[WEB_SEARCH_ERROR] 网页搜索失败: {str(e)}")
            raise Exception(f"网页搜索失败: {str(e)}")

    async def get_web_context(self, question: str, web_id: str, history: str = None, 
                             chunk_limit: int = 5) -> dict:
        """获取网页相关上下文 - 新增方法"""
        try:
            logger.info(f"[WEB_CONTEXT] 开始获取网页上下文 - ID: {web_id}")
            
            # 使用现有的get_relevant_context方法，因为网页也存储在PaperAnalysis表中
            context = await self.get_relevant_context(
                question=question,
                paper_id=web_id,
                history=history,
                doc_limit=chunk_limit
            )
            
            # 为网页上下文添加特殊标记
            if context and "chunks" in context:
                for chunk in context["chunks"]:
                    chunk["content_type"] = "web"
                    chunk["is_web_content"] = True
            
            logger.info(f"[WEB_CONTEXT_SUCCESS] 网页上下文获取完成")
            return context
            
        except Exception as e:
            logger.error(f"[WEB_CONTEXT_ERROR] 获取网页上下文失败: {str(e)}")
            raise Exception(f"获取网页上下文失败: {str(e)}")
