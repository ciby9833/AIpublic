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
    def __init__(self, db=None):
        # 使用新的模型加载方式
        model_name = 'all-mpnet-base-v2'
        try:
            # 首先尝试从缓存加载
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
            model_path = os.path.join(cache_dir, model_name)
            
            if not os.path.exists(model_path):
                # 如果缓存不存在，下载模型
                model_path = snapshot_download(
                    repo_id=f"sentence-transformers/{model_name}",
                    cache_dir=cache_dir
                )
            
            self.model = SentenceTransformer(model_path)
            # 确保使用 CPU 进行推理
            self.model.to('cpu')
            
        except Exception as e:
            print(f"Model loading error: {str(e)}")
            raise
            
        self.paper_indices = {}  # 保留内存缓存
        self.db = db  # 添加数据库连接
        self.chunk_size = 512
        self.overlap = 50
        
        # 检查系统架构
        self.is_m1 = platform.processor() == 'arm'
        if self.is_m1:
            print("Running on M1 Mac, using CPU version of FAISS")

    def _create_index(self, dimension: int):
        """创建适合当前系统的索引"""
        if self.is_m1:
            # M1 Mac 使用 CPU 版本
            return faiss.IndexFlatL2(dimension)
        else:
            # 其他系统可以使用 GPU 版本
            return faiss.IndexFlatL2(dimension)

    def _split_text(self, text: str) -> List[str]:
        """将文本分割成重叠的块"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    async def build_index(self, content: str, paper_id: str, line_mapping: dict = None, db=None, structured_data: dict = None):
        """构建文档索引并存储到数据库"""
        try:
            # 使用传入的db连接，或者实例的db连接
            current_db = db or self.db
            if not current_db:
                raise ValueError("Database connection required for index building")
                
            # 分割文档为块
            chunks = self._split_text(content)
            
            # 生成嵌入向量
            with torch.no_grad():
                embeddings = self.model.encode(chunks, convert_to_numpy=True)
            
            # 创建适合当前系统的索引
            dimension = embeddings.shape[1]
            index = self._create_index(dimension)
            
            # 确保数据类型正确
            embeddings = embeddings.astype('float32')
            
            # 添加向量到索引
            index.add(embeddings)
            
            # 存储到内存缓存
            self.paper_indices[paper_id] = {
                'index': index,
                'documents': chunks,
                'embeddings': embeddings,
                'line_mapping': line_mapping or {},
                'structured_data': structured_data  # 添加结构化数据
            }
            
            # 将嵌入向量和文档存储到数据库
            from models.paper import PaperAnalysis
            
            # 将numpy数组转换为Python列表以便JSON序列化
            embeddings_list = embeddings.tolist()
            
            # 更新数据库
            paper = current_db.query(PaperAnalysis).filter(
                PaperAnalysis.paper_id == uuid.UUID(paper_id)
            ).first()
            
            if paper:
                paper.documents = chunks
                paper.embeddings = embeddings_list
                paper.index_built = True
                # 添加结构化数据到数据库
                if structured_data:
                    paper.structured_data = structured_data
                current_db.commit()
                
                print(f"Successfully built and stored index for paper {paper_id} with {len(chunks)} chunks")
            else:
                print(f"Warning: Paper {paper_id} not found in database")
            
        except Exception as e:
            print(f"Index building error details: {str(e)}")
            if 'current_db' in locals() and current_db:
                current_db.rollback()
            raise Exception(f"Index building error: {str(e)}")

    async def _ensure_paper_index(self, paper_id: str) -> Dict:
        """确保文档索引已加载，如果未加载则从数据库加载"""
        # 检查是否有内存缓存的索引
        if paper_id not in self.paper_indices:
            # 尝试从数据库恢复索引
            current_db = self.db
            if not current_db:
                raise ValueError("Database connection required to retrieve index")
            
            from models.paper import PaperAnalysis
            
            # 查询数据库
            paper = current_db.query(PaperAnalysis).filter(
                PaperAnalysis.paper_id == uuid.UUID(paper_id)
            ).first()
            
            if not paper:
                raise Exception(f"Paper ID {paper_id} not found in database")
                
            if not paper.index_built or not paper.documents or not paper.embeddings:
                # 尝试重建索引
                if paper.content:
                    print(f"Rebuilding index for paper {paper_id}")
                    # 重建索引
                    await self.build_index(paper.content, paper_id, paper.line_mapping, current_db)
                    # 重新查询更新后的记录
                    paper = current_db.query(PaperAnalysis).filter(
                        PaperAnalysis.paper_id == uuid.UUID(paper_id)
                    ).first()
                else:
                    raise Exception(f"Paper content not available for {paper_id}")
            
            try:
                # 恢复索引
                embeddings = np.array(paper.embeddings, dtype='float32')
                documents = paper.documents
                line_mapping = paper.line_mapping or {}
                
                # 创建索引
                dimension = embeddings.shape[1]
                index = self._create_index(dimension)
                index.add(embeddings)
                
                # 缓存到内存
                self.paper_indices[paper_id] = {
                    'index': index,
                    'documents': documents,
                    'embeddings': embeddings,
                    'line_mapping': line_mapping,
                    'filename': paper.filename,  # 添加文件名
                    'paper_id': paper_id         # 添加ID冗余信息
                }
                
                print(f"Successfully restored index for paper {paper_id} from database")
            except Exception as e:
                print(f"Error restoring index from database: {str(e)}")
                raise Exception(f"Failed to restore index: {str(e)}")
                
        return self.paper_indices[paper_id]

    async def get_relevant_context(self, question: str, paper_id: str, history: str = None, doc_limit: int = 5) -> dict:
        """获取相关上下文，支持历史对话"""
        try:
            # 如果有历史对话，将其与问题组合
            search_query = f"{history}\n{question}" if history else question
            
            # 确保索引已加载
            paper_data = await self._ensure_paper_index(paper_id)
            index = paper_data['index']
            documents = paper_data['documents']
            line_mapping = paper_data['line_mapping']
            filename = paper_data.get('filename', '未知文件')
            
            # 生成问题的嵌入向量
            with torch.no_grad():
                question_embedding = self.model.encode([search_query], convert_to_numpy=True)[0]
            question_embedding = question_embedding.reshape(1, -1).astype('float32')
            
            # 使用 FAISS 进行相似度搜索
            k = min(doc_limit, len(documents))
            distances, indices = index.search(question_embedding, k)
            
            # 组合相关段落，包含行号信息
            relevant_chunks = []
            relevant_text = ""
            
            for i, idx in enumerate(indices[0]):
                idx = int(idx)  # 确保是整数
                chunk = documents[idx] if idx < len(documents) else ""
                
                # 安全获取行信息
                line_info = {}
                if isinstance(line_mapping, dict):
                    str_key = str(idx)
                    int_key = idx
                    # 尝试不同类型的键
                    if str_key in line_mapping:
                        line_info = line_mapping[str_key]
                    elif int_key in line_mapping:
                        line_info = line_mapping[int_key]
                
                similarity = float(1 - distances[0][i]) if i < len(distances[0]) else 0.0
                
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
                    "document_id": str(paper_id),  # 添加文档ID
                    "document_name": filename      # 添加文档名称
                })
            
            return {
                "chunks": relevant_chunks,
                "total_chunks": len(relevant_chunks),
                "question_embedding": question_embedding.tolist(),
                "text": relevant_text.strip(),  # 添加合并的文本
                "document_id": paper_id,
                "document_name": filename
            }
            
        except Exception as e:
            print(f"Context retrieval error details: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise Exception(f"Context retrieval error: {str(e)}")

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
