from typing import List, Optional, TypedDict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import platform
from huggingface_hub import snapshot_download
import os
import torch

class ChunkInfo(TypedDict):
    line_number: int
    content: str
    page: int
    start_pos: int
    end_pos: int
    is_scanned: bool
    similarity: float

class ContextInfo(TypedDict):
    chunks: List[ChunkInfo]
    total_chunks: int
    question_embedding: Optional[List[float]]

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

    async def build_index(self, content: str, paper_id: str, line_mapping: dict = None, db=None):
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
                'line_mapping': line_mapping or {}
            }
            
            # 将嵌入向量和文档存储到数据库
            from models.paper import PaperAnalysis
            import uuid
            
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
                current_db.commit()
                
                print(f"Successfully built and stored index for paper {paper_id} with {len(chunks)} chunks")
            else:
                print(f"Warning: Paper {paper_id} not found in database")
            
        except Exception as e:
            print(f"Index building error details: {str(e)}")
            if 'current_db' in locals() and current_db:
                current_db.rollback()
            raise Exception(f"Index building error: {str(e)}")

    async def get_relevant_context(self, question: str, paper_id: str, db=None):
        """获取相关上下文，包含行号信息，支持从数据库恢复索引"""
        try:
            # 检查是否有内存缓存的索引
            if paper_id not in self.paper_indices:
                # 尝试从数据库恢复索引
                current_db = db or self.db
                if not current_db:
                    raise ValueError("Database connection required to retrieve index")
                
                from models.paper import PaperAnalysis
                import uuid
                import json
                
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
                        'line_mapping': line_mapping
                    }
                    
                    print(f"Successfully restored index for paper {paper_id} from database")
                except Exception as e:
                    print(f"Error restoring index from database: {str(e)}")
                    raise Exception(f"Failed to restore index: {str(e)}")

            # 使用索引检索相关上下文
            paper_data = self.paper_indices[paper_id]
            index = paper_data['index']
            documents = paper_data['documents']
            line_mapping = paper_data['line_mapping']
            
            # 生成问题的嵌入向量
            with torch.no_grad():
                question_embedding = self.model.encode([question], convert_to_numpy=True)[0]
            question_embedding = question_embedding.reshape(1, -1).astype('float32')
            
            # 使用 FAISS 进行相似度搜索
            k = min(5, len(documents))
            distances, indices = index.search(question_embedding, k)
            
            # 组合相关段落，包含行号信息
            relevant_chunks = []
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
                
                relevant_chunks.append({
                    "line_number": idx,
                    "content": str(chunk),
                    "page": int(line_info.get("page", 1)),
                    "start_pos": int(line_info.get("start_pos", 0)),
                    "end_pos": int(line_info.get("end_pos", 0)),
                    "is_scanned": bool(line_info.get("is_scanned", False)),
                    "similarity": similarity
                })
            
            return {
                "chunks": relevant_chunks,
                "total_chunks": len(relevant_chunks),
                "question_embedding": question_embedding.tolist()
            }
            
        except Exception as e:
            print(f"Context retrieval error details: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise Exception(f"Context retrieval error: {str(e)}")
