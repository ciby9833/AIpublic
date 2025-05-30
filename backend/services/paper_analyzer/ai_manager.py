import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Union, Dict, Any, Literal
import json
import re
import asyncio
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading

# 添加日志配置
logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """记忆项目数据结构"""
    content: str
    timestamp: datetime
    importance: float  # 重要性权重 0-1
    context_type: str  # "conversation", "document", "fact", "preference"
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    session_id: Optional[str] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass 
class ConversationMemory:
    """对话记忆管理"""
    short_term: deque = field(default_factory=lambda: deque(maxlen=50))  # 短期记忆
    long_term: Dict[str, MemoryItem] = field(default_factory=dict)  # 长期记忆
    session_memory: Dict[str, List[MemoryItem]] = field(default_factory=dict)  # 会话记忆
    user_preferences: Dict[str, Any] = field(default_factory=dict)  # 用户偏好
    document_insights: Dict[str, List[MemoryItem]] = field(default_factory=dict)  # 文档洞察

@dataclass
class IntentAnalysisConfig:
    """意图分析配置"""
    use_advanced_analysis: bool = True
    confidence_threshold: float = 0.7
    enable_context_awareness: bool = True
    enable_document_awareness: bool = True
    enable_multi_intent: bool = False
    cache_intent_results: bool = True
    intent_cache_ttl: int = 300  # 5分钟

class SourceInfo(TypedDict):
    line_number: int
    content: str
    page: int
    start_pos: int
    end_pos: int
    is_scanned: bool
    similarity: float
    document_id: Optional[str]
    document_name: Optional[str]

class MessageContent(TypedDict):
    type: Literal["text", "markdown", "code", "table", "image"]
    content: str
    language: Optional[str]  # For code blocks
    alt: Optional[str]  # For images
    columns: Optional[List[str]]  # For tables
    rows: Optional[List[List[Any]]]  # For tables
    metadata: Optional[Dict[str, Any]]  # Additional metadata

class AIResponse(TypedDict):
    answer: str  # Backward compatibility
    sources: List[SourceInfo]
    confidence: float
    reply: List[MessageContent]  # New rich content format

class MessageInfo(TypedDict):
    role: str
    content: str

class AdvancedMemoryManager:
    """高级记忆管理器 - 参考长期记忆项目的概念"""
    
    def __init__(self, max_short_term: int = 50, max_long_term: int = 1000):
        self.memory = ConversationMemory()
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        self.lock = threading.RLock()
        
        # 记忆重要性计算权重
        self.importance_weights = {
            "recency": 0.3,      # 时间新近性
            "frequency": 0.2,    # 访问频率
            "relevance": 0.3,    # 内容相关性
            "explicit": 0.2      # 显式重要性标记
        }
        
        logger.info(f"[MEMORY_INIT] 记忆管理器初始化 - 短期容量: {max_short_term}, 长期容量: {max_long_term}")

    def add_memory(self, content: str, context_type: str = "conversation", 
                   importance: float = 0.5, user_id: str = None, 
                   document_id: str = None, session_id: str = None) -> str:
        """添加记忆项目"""
        with self.lock:
            memory_id = f"mem_{int(time.time() * 1000)}"
            memory_item = MemoryItem(
                content=content,
                timestamp=datetime.utcnow(),
                importance=importance,
                context_type=context_type,
                user_id=user_id,
                document_id=document_id,
                session_id=session_id
            )
            
            # 添加到短期记忆
            self.memory.short_term.append(memory_item)
            
            # 根据重要性决定是否进入长期记忆
            if importance > 0.7 or context_type in ["fact", "preference"]:
                self.memory.long_term[memory_id] = memory_item
                self._manage_long_term_capacity()
            
            # 添加到会话记忆
            if session_id:
                if session_id not in self.memory.session_memory:
                    self.memory.session_memory[session_id] = []
                self.memory.session_memory[session_id].append(memory_item)
            
            # 添加到文档洞察
            if document_id:
                if document_id not in self.memory.document_insights:
                    self.memory.document_insights[document_id] = []
                self.memory.document_insights[document_id].append(memory_item)
            
            logger.debug(f"[MEMORY_ADD] 添加记忆 - 类型: {context_type}, 重要性: {importance:.2f}")
            return memory_id

    def get_relevant_memories(self, query: str, user_id: str = None, 
                            session_id: str = None, limit: int = 5) -> List[MemoryItem]:
        """获取相关记忆"""
        with self.lock:
            relevant_memories = []
            
            # 从短期记忆中查找
            for memory in list(self.memory.short_term):
                if self._is_memory_relevant(memory, query, user_id, session_id):
                    memory.access_count += 1
                    memory.last_accessed = datetime.utcnow()
                    relevant_memories.append(memory)
            
            # 从长期记忆中查找
            for memory in self.memory.long_term.values():
                if self._is_memory_relevant(memory, query, user_id, session_id):
                    memory.access_count += 1
                    memory.last_accessed = datetime.utcnow()
                    relevant_memories.append(memory)
            
            # 按重要性和相关性排序
            relevant_memories.sort(key=lambda m: self._calculate_memory_score(m, query), reverse=True)
            
            logger.debug(f"[MEMORY_RETRIEVE] 检索到 {len(relevant_memories)} 条相关记忆")
            return relevant_memories[:limit]

    def _is_memory_relevant(self, memory: MemoryItem, query: str, user_id: str = None, session_id: str = None) -> bool:
        """判断记忆是否相关"""
        # 简单的文本相似度检查
        query_words = set(query.lower().split())
        memory_words = set(memory.content.lower().split())
        
        if len(query_words & memory_words) > 0:
            return True
        
        # 用户和会话匹配
        if user_id and memory.user_id == user_id:
            return True
        
        if session_id and memory.session_id == session_id:
            return True
        
        return False

    def _calculate_memory_score(self, memory: MemoryItem, query: str) -> float:
        """计算记忆得分"""
        score = 0.0
        
        # 时间新近性
        age_hours = (datetime.utcnow() - memory.timestamp).total_seconds() / 3600
        recency_score = max(0, 1 - age_hours / (24 * 7))  # 一周内的记忆得分较高
        score += recency_score * self.importance_weights["recency"]
        
        # 访问频率
        frequency_score = min(1.0, memory.access_count / 10)
        score += frequency_score * self.importance_weights["frequency"]
        
        # 显式重要性
        score += memory.importance * self.importance_weights["explicit"]
        
        # 内容相关性（简单实现）
        query_words = set(query.lower().split())
        memory_words = set(memory.content.lower().split())
        if len(query_words) > 0:
            relevance_score = len(query_words & memory_words) / len(query_words)
            score += relevance_score * self.importance_weights["relevance"]
        
        return score

    def _manage_long_term_capacity(self):
        """管理长期记忆容量"""
        if len(self.memory.long_term) > self.max_long_term:
            # 移除最不重要的记忆
            memories_by_score = sorted(
                self.memory.long_term.items(),
                key=lambda x: self._calculate_memory_score(x[1], ""),
                reverse=False
            )
            
            to_remove = len(self.memory.long_term) - self.max_long_term
            for i in range(to_remove):
                memory_id, _ = memories_by_score[i]
                del self.memory.long_term[memory_id]
            
            logger.debug(f"[MEMORY_CLEANUP] 清理了 {to_remove} 条长期记忆")

class EnhancedIntentAnalyzer:
    """增强的意图分析器 - 结合文档感知和上下文感知"""
    
    def __init__(self, config: IntentAnalysisConfig = None):
        self.config = config or IntentAnalysisConfig()
        self.intent_cache = {}
        self.cache_timestamps = {}
        
        # 预定义意图模式
        self.intent_patterns = {
            "DOCUMENT_QUERY": [
                r"文档.*说.*什么", r"根据.*文档", r"文档.*内容", r"在.*文档.*中",
                r"这.*文档.*描述", r"文档.*提到", r"查找.*文档", r"搜索.*文档"
            ],
            "GENERAL_QUERY": [
                r"什么是", r"如何.*", r"为什么", r"告诉我", r"解释.*",
                r"介绍.*", r"说明.*", r"描述.*"  
            ],
            "COMPARISON_QUERY": [
                r"比较.*", r"对比.*", r".*区别.*", r".*差异.*", r".*优缺点.*"
            ],
            "SUMMARY_QUERY": [
                r"总结.*", r"概括.*", r"归纳.*", r"整理.*", r"梳理.*"
            ],
            "ANALYSIS_QUERY": [
                r"分析.*", r"评估.*", r"判断.*", r"评价.*", r"深入.*研究.*"
            ]
        }
        
        logger.info(f"[INTENT_INIT] 意图分析器初始化 - 高级分析: {self.config.use_advanced_analysis}")

    async def analyze_intent(self, question: str, has_documents: bool = False, 
                           history: str = None, document_info: dict = None,
                           user_context: dict = None) -> str:
        """分析用户意图 - 增强版本"""
        
        # 检查缓存
        cache_key = f"{question}_{has_documents}_{hash(history or '')}"
        if self.config.cache_intent_results and cache_key in self.intent_cache:
            cached_time = self.cache_timestamps.get(cache_key, datetime.min)
            if (datetime.utcnow() - cached_time).seconds < self.config.intent_cache_ttl:
                logger.debug(f"[INTENT_CACHE] 使用缓存结果")
                return self.intent_cache[cache_key]
        
        logger.info(f"[INTENT_ANALYZE] 开始意图分析 - 问题: {question[:50]}...")
        
        # 基础模式匹配
        base_intent = self._pattern_based_analysis(question)
        logger.debug(f"[INTENT_PATTERN] 模式匹配结果: {base_intent}")
        
        # 上下文感知分析
        if self.config.enable_context_awareness and history:
            context_intent = self._context_aware_analysis(question, history)
            logger.debug(f"[INTENT_CONTEXT] 上下文分析结果: {context_intent}")
        else:
            context_intent = base_intent
        
        # 文档感知分析
        if self.config.enable_document_awareness and has_documents:
            doc_intent = self._document_aware_analysis(question, document_info)
            logger.debug(f"[INTENT_DOCUMENT] 文档感知分析结果: {doc_intent}")
        else:
            doc_intent = context_intent
        
        # 最终意图决策
        final_intent = self._make_final_decision(base_intent, context_intent, doc_intent, has_documents)
        
        # 缓存结果
        if self.config.cache_intent_results:
            self.intent_cache[cache_key] = final_intent
            self.cache_timestamps[cache_key] = datetime.utcnow()
        
        logger.info(f"[INTENT_FINAL] 最终意图: {final_intent}")
        return final_intent

    def _pattern_based_analysis(self, question: str) -> str:
        """基于模式的意图分析"""
        question_lower = question.lower()
        
        # 逐个检查意图模式
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return intent
        
        # 默认意图
        return "GENERAL_QUERY"

    def _context_aware_analysis(self, question: str, history: str) -> str:
        """上下文感知分析"""
        if not history:
            return self._pattern_based_analysis(question)
        
        # 分析历史对话中的文档引用
        if any(keyword in history.lower() for keyword in ["文档", "根据", "内容", "材料"]):
            return "DOCUMENT_QUERY"
        
        # 分析对话连续性
        if any(keyword in question.lower() for keyword in ["还有", "另外", "继续", "进一步"]):
            return "DOCUMENT_QUERY"
        
        return self._pattern_based_analysis(question)

    def _document_aware_analysis(self, question: str, document_info: dict = None) -> str:
        """文档感知分析"""
        if not document_info:
            return self._pattern_based_analysis(question)
        
        doc_count = document_info.get("count", 0)
        
        # 多文档场景优先考虑文档查询
        if doc_count > 1:
            return "DOCUMENT_QUERY"
        
        # 单文档场景，检查问题中的文档引用
        if any(keyword in question.lower() for keyword in ["这个", "该", "此"]):
            return "DOCUMENT_QUERY"
        
        return self._pattern_based_analysis(question)

    def _make_final_decision(self, base_intent: str, context_intent: str, 
                           doc_intent: str, has_documents: bool) -> str:
        """最终意图决策"""
        
        # 权重投票机制
        intent_votes = defaultdict(float)
        intent_votes[base_intent] += 0.4
        intent_votes[context_intent] += 0.3
        intent_votes[doc_intent] += 0.3
        
        # 如果有文档但投票结果是通用查询，调整权重
        if has_documents and max(intent_votes.keys(), key=intent_votes.get) == "GENERAL_QUERY":
            if intent_votes["DOCUMENT_QUERY"] > 0.2:
                return "DOCUMENT_QUERY"
        
        # 返回得票最高的意图
        return max(intent_votes.keys(), key=intent_votes.get)

class AIManager:
    def __init__(self, enable_memory: bool = True, enable_enhanced_intent: bool = True):
        # 配置Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 生成配置
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            top_p=0.95,
            top_k=32,
            max_output_tokens=8192,
            response_mime_type="text/plain",
        )
        
        # 安全设置
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        # 集成增强功能
        self.enable_memory = enable_memory
        self.enable_enhanced_intent = enable_enhanced_intent
        
        # 初始化记忆管理器
        if enable_memory:
            self.memory_manager = AdvancedMemoryManager()
            logger.info(f"[AI_INIT] 记忆管理已启用")
        else:
            self.memory_manager = None
            logger.info(f"[AI_INIT] 记忆管理已禁用")
        
        # 初始化增强意图分析器
        if enable_enhanced_intent:
            intent_config = IntentAnalysisConfig(
                use_advanced_analysis=True,
                enable_context_awareness=True,
                enable_document_awareness=True
            )
            self.intent_analyzer = EnhancedIntentAnalyzer(intent_config)
            logger.info(f"[AI_INIT] 增强意图分析已启用")
        else:
            self.intent_analyzer = None
            logger.info(f"[AI_INIT] 使用简化意图分析")
        
        logger.info(f"[AI_MANAGER] AIManager初始化完成 - 记忆: {enable_memory}, 增强意图: {enable_enhanced_intent}")

    async def get_response(self, question: str, context: Union[str, Dict, List[Dict]], history: str = None, 
                          user_id: str = None, session_id: str = None, document_id: str = None) -> dict:
        """获取AI回答 - 集成记忆管理"""
        try:
            logger.info(f"[AI_RESPONSE] 开始生成回答 - 问题长度: {len(question)} 字符")
            
            # 记忆增强
            if self.memory_manager and user_id:
                # 获取相关记忆
                relevant_memories = self.memory_manager.get_relevant_memories(
                    query=question,
                    user_id=user_id,
                    session_id=session_id,
                    limit=3
                )
                
                if relevant_memories:
                    memory_context = "\n".join([m.content for m in relevant_memories])
                    logger.debug(f"[AI_MEMORY] 使用 {len(relevant_memories)} 条相关记忆")
                    
                    # 将记忆融入历史上下文
                    if history:
                        history = f"相关记忆：{memory_context}\n\n历史对话：{history}"
                    else:
                        history = f"相关记忆：{memory_context}"
            
            # 智能意图识别
            intent = "DOCUMENT_QUERY"  # 默认值
            if self.intent_analyzer:
                document_info = self._extract_document_info(context)
                intent = await self.intent_analyzer.analyze_intent(
                    question=question,
                    has_documents=bool(context),
                    history=history,
                    document_info=document_info
                )
            else:
                # 使用简化意图分析
                intent = await self.identify_intent(question, context, bool(context), history)
            
            logger.info(f"[AI_INTENT] 识别意图: {intent}")
            
            # 根据意图选择生成策略
            if intent == "GENERAL_QUERY":
                logger.info(f"[AI_MODE_GENERAL] 使用通用知识模式")
                prompt = f"""你是CargoPPT的AI助手。请基于你的知识回答以下问题。

历史对话：{history or '无'}

用户问题：{question}

请根据你的知识回答这个问题，不要提及任何文档。"""
                
            elif intent == "ANALYSIS_QUERY":
                logger.info(f"[AI_MODE_ANALYSIS] 使用分析查询模式")
                # 为分析查询构建专门的prompt
                base_prompt = self._build_prompt(question, context, history)
                
                # 智能分析提示 - 结合文档内容和外部知识
                analysis_instruction = """

## 📋 分析指导

**任务**: 基于提供的文档内容，结合你的知识库，进行深度分析和推理。

**分析策略**:
1. 📖 **文档内容提取**: 首先总结文档中的关键信息、概念和应用案例
2. 🧠 **知识库扩展**: 基于文档内容，调用你的专业知识进行拓展和补充
3. 🔗 **智能推理**: 将文档内容与外部知识结合，进行逻辑推理和分析
4. 🎯 **具体应用**: 针对用户提及的特定领域或应用场景，提供具体的分析和建议

**回答要求**:
- ✅ 首先基于文档内容回答
- ✅ 然后结合相关领域知识进行扩展
- ✅ 提供具体的应用场景和实施建议
- ✅ 使用结构化格式（标题、要点、表格等）
- ✅ 当文档内容不足时，明确说明并基于合理推理补充

**特别注意**: 如果用户询问特定行业应用（如物流、金融、医疗等），即使文档中没有直接提及，也要基于文档中的通用概念和你的专业知识，推理出在该行业的可能应用。"""
                
                prompt = base_prompt + analysis_instruction
            
            elif intent in ["COMPARISON_QUERY", "SUMMARY_QUERY"]:
                logger.info(f"[AI_MODE_SPECIAL] 使用特殊查询模式: {intent}")
                prompt = self._build_prompt(question, context, history)
                
                if intent == "COMPARISON_QUERY":
                    prompt += "\n请提供详细的比较分析，包括相似点、不同点和优缺点。"
                else:  # SUMMARY_QUERY
                    prompt += "\n请提供结构化的总结，包括关键要点、重要结论和实际意义。"
            
            else:  # DOCUMENT_QUERY 或默认
                logger.info(f"[AI_MODE_DOCUMENT] 使用文档查询模式")
                prompt = self._build_prompt(question, context, history)
            
            logger.debug(f"[AI_PROMPT] 提示词长度: {len(prompt)} 字符")
            
            # 生成回答
            logger.info(f"[AI_GENERATE_START] 开始调用Gemini API生成回答")
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # 检查响应
            if not response or not response.text:
                logger.error(f"[AI_GENERATE_EMPTY] Gemini返回空响应")
                return {
                    "answer": "抱歉，我无法生成回答。请重新尝试您的问题。",
                    "sources": [],
                    "confidence": 0.0,
                    "reply": [{"type": "markdown", "content": "抱歉，我无法生成回答。请重新尝试您的问题。"}]
                }
            
            answer = response.text.strip()
            logger.info(f"[AI_GENERATE_SUCCESS] 回答生成成功 - 长度: {len(answer)} 字符")
            
            # 记忆存储
            if self.memory_manager and user_id:
                # 存储问题
                self.memory_manager.add_memory(
                    content=f"用户问题: {question}",
                    context_type="conversation",
                    importance=0.6,
                    user_id=user_id,
                    session_id=session_id,
                    document_id=document_id
                )
                
                # 存储重要答案
                if len(answer) > 100:  # 只存储较长的答案
                    self.memory_manager.add_memory(
                        content=f"AI回答: {answer[:500]}...",  # 截取前500字符
                        context_type="conversation",
                        importance=0.7,
                        user_id=user_id,
                        session_id=session_id,
                        document_id=document_id
                    )
                
                logger.debug(f"[AI_MEMORY_STORE] 已存储对话记忆")
            
            # 提取来源信息
            sources = []
            confidence = 0.8
            
            if isinstance(context, dict):
                sources = context.get("chunks", [])[:5]
                confidence = 0.9
            elif isinstance(context, list):
                sources = context[:5]
                confidence = 0.9
            
            # 解析回答为富文本格式
            reply = self._parse_response_content(answer)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "reply": reply,
                "intent": intent,  # 返回识别的意图
                "memory_used": len(relevant_memories) if self.memory_manager and user_id and 'relevant_memories' in locals() else 0
            }
            
        except Exception as e:
            logger.error(f"[AI_RESPONSE_ERROR] 生成回答失败: {str(e)}")
            import traceback
            logger.debug(f"[AI_RESPONSE_TRACEBACK] 错误堆栈: {traceback.format_exc()}")
            
            return {
                "answer": f"处理您的问题时发生错误: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "reply": [{"type": "markdown", "content": f"处理您的问题时发生错误: {str(e)}"}],
                "intent": "ERROR",
                "memory_used": 0
            }

    def _extract_document_info(self, context) -> dict:
        """从上下文中提取文档信息"""
        if not context:
            return {"count": 0, "names": []}
        
        if isinstance(context, dict):
            if "documents" in context:
                # 多文档格式
                docs = context["documents"]
                return {
                    "count": len(docs),
                    "names": [doc.get("document_name", "未知文档") for doc in docs]
                }
            elif "document_name" in context:
                # 单文档格式
                return {
                    "count": 1,
                    "names": [context["document_name"]]
                }
        
        return {"count": 1, "names": ["文档"]}

    async def chat_without_context(self, message: str, history: List[Dict[str, str]] = None, 
                                 user_id: str = None, session_id: str = None) -> dict:
        """无上下文聊天 - 集成记忆管理"""
        try:
            logger.info(f"[AI_CHAT] 开始无上下文聊天 - 消息长度: {len(message)} 字符")
            
            # 记忆增强
            memory_context = ""
            if self.memory_manager and user_id:
                relevant_memories = self.memory_manager.get_relevant_memories(
                    query=message,
                    user_id=user_id,
                    session_id=session_id,
                    limit=3
                )
                
                if relevant_memories:
                    memory_context = "\n".join([f"- {m.content}" for m in relevant_memories])
                    logger.debug(f"[AI_CHAT_MEMORY] 使用 {len(relevant_memories)} 条相关记忆")
            
            # 构建提示词
            history_text = ""
            if history:
                history_items = []
                for msg in history[-8:]:  # 最近8轮对话
                    role = "用户" if msg["role"] == "user" else "助手"
                    history_items.append(f"{role}: {msg['content']}")
                history_text = "\n".join(history_items)
            
            prompt = f"""你是CargoPPT的AI助手，一个专业、友好且有帮助的AI。

{f"相关记忆：{memory_context}" if memory_context else ""}

{f"对话历史：{history_text}" if history_text else ""}

用户问题：{message}

请提供有帮助的回答。如果问题涉及专业知识，请基于你的知识给出准确的信息。"""
            
            # 生成回答
            response = self.model.generate_content( 
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            if not response or not response.text:
                logger.error(f"[AI_CHAT_EMPTY] Gemini返回空响应")
                return {
                    "answer": "抱歉，我无法理解您的问题。请重新表述。",
                            "sources": [],
                    "confidence": 0.0,
                    "reply": [{"type": "markdown", "content": "抱歉，我无法理解您的问题。请重新表述。"}]
                }
            
            answer = response.text.strip()
            logger.info(f"[AI_CHAT_SUCCESS] 聊天回答生成成功 - 长度: {len(answer)} 字符")
            
            # 记忆存储
            if self.memory_manager and user_id:
                # 存储问题
                self.memory_manager.add_memory(
                    content=f"用户问题: {message}",
                    context_type="conversation",
                    importance=0.5,
                    user_id=user_id,
                    session_id=session_id
                )
                
                # 存储答案
                if len(answer) > 50:
                    self.memory_manager.add_memory(
                        content=f"AI回答: {answer[:300]}...",
                        context_type="conversation",
                        importance=0.6,
                        user_id=user_id,
                        session_id=session_id
                    )
            
            # 解析回答
            reply = self._parse_response_content(answer)
            
            return {
                "answer": answer,
                "sources": [],
                "confidence": 0.7,
                "reply": reply,
                "memory_used": len(relevant_memories) if self.memory_manager and user_id and 'relevant_memories' in locals() else 0
            }
            
        except Exception as e:
            logger.error(f"[AI_CHAT_ERROR] 无上下文聊天失败: {str(e)}")
            return {
                "answer": f"聊天时发生错误: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "reply": [{"type": "markdown", "content": f"聊天时发生错误: {str(e)}"}],
                "memory_used": 0
            }

    async def get_multi_document_response(self, question: str, document_contexts: List[Dict], history: str = None) -> dict:
        """处理多文档查询，整合多个文档的相关内容"""
        logger.info(f"[MULTI_DOC] 开始多文档查询 - 问题: {question[:50]}..., 文档数量: {len(document_contexts)}")
        
        try:
            # 整合所有文档的上下文
            combined_context = ""
            sources = []
            
            for i, doc_context in enumerate(document_contexts):
                if doc_context and doc_context.get("context"):
                    combined_context += f"\n\n=== 文档 {i+1} ===\n{doc_context['context']}"
                    if doc_context.get("sources"):
                        sources.extend(doc_context["sources"])
            
            logger.debug(f"[MULTI_DOC_CONTEXT] 合并上下文长度: {len(combined_context)} 字符, 来源数量: {len(sources)}")
            
            # 使用合并的上下文生成回答
            response = await self.get_response(question, combined_context, history)
            
            # 更新来源信息
            if response and isinstance(response, dict):
                response["sources"] = sources[:10]  # 限制来源数量
                logger.info(f"[MULTI_DOC_SUCCESS] 多文档查询成功 - 响应长度: {len(response.get('answer', ''))} 字符")
            
            return response
            
        except Exception as e:
            logger.error(f"[MULTI_DOC_ERROR] 多文档查询失败: {str(e)}")
            return {
                "answer": f"多文档查询时发生错误: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "reply": [{"type": "markdown", "content": f"多文档查询时发生错误: {str(e)}"}]
            }

    async def analyze_structured_data(self, query: str, structured_data: dict, paper_id: str = None, is_sampled: bool = False) -> dict:
        """分析结构化数据（表格、图表等）"""
        logger.info(f"[STRUCTURED_ANALYSIS] 开始结构化数据分析 - 查询: {query[:50]}..., 数据类型: {type(structured_data)}")
        
        try:
            # 构建结构化数据的文本描述
            data_description = self._format_structured_data(structured_data)
            
            # 构建专门的提示词
            prompt = f"""基于以下结构化数据回答问题：

数据内容：
{data_description}

用户问题：{query}

请基于上述数据提供准确的分析和回答。如果数据不足以回答问题，请明确说明。"""

            logger.debug(f"[STRUCTURED_PROMPT] 结构化数据提示词长度: {len(prompt)} 字符")
            
            # 生成回答
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            if response and response.text:
                logger.info(f"[STRUCTURED_SUCCESS] 结构化数据分析成功 - 响应长度: {len(response.text)} 字符")
                
                # 解析响应内容
                parsed_content = self._parse_response_content(response.text)
                
                return {
                    "answer": response.text,
                    "sources": [{
                        "line_number": 0,
                        "content": "结构化数据分析",
                        "page": 1,
                        "start_pos": 0,
                        "end_pos": len(str(structured_data)),
                        "is_scanned": False,
                        "similarity": 1.0,
                        "document_id": paper_id,
                        "document_name": "结构化数据"
                    }] if paper_id else [],
                    "confidence": 0.8,
                    "reply": parsed_content
                }
            else:
                logger.warning(f"[STRUCTURED_NO_RESPONSE] 结构化数据分析无响应")
                return {
                    "answer": "无法分析提供的结构化数据",
                    "sources": [],
                    "confidence": 0.0,
                    "reply": [{"type": "markdown", "content": "无法分析提供的结构化数据"}]
                }
                
        except Exception as e:
            logger.error(f"[STRUCTURED_ERROR] 结构化数据分析失败: {str(e)}")
            return {
                "answer": f"结构化数据分析时发生错误: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "reply": [{"type": "markdown", "content": f"结构化数据分析时发生错误: {str(e)}"}]
            }

    def _format_structured_data(self, data: dict) -> str:
        """格式化结构化数据为文本描述"""
        logger.debug(f"[FORMAT_STRUCTURED] 开始格式化结构化数据 - 数据键: {list(data.keys()) if isinstance(data, dict) else 'Not dict'}")
        
        try:
            if not isinstance(data, dict):
                return str(data)
            
            formatted_parts = []
            
            # 处理表格数据
            if "tables" in data and data["tables"]:
                formatted_parts.append("=== 表格数据 ===")
                for i, table in enumerate(data["tables"][:5]):  # 限制表格数量
                    formatted_parts.append(f"\n表格 {i+1}:")
                    if isinstance(table, dict):
                        if "headers" in table and table["headers"]:
                            formatted_parts.append(f"列标题: {', '.join(table['headers'])}")
                        if "rows" in table and table["rows"]:
                            formatted_parts.append(f"数据行数: {len(table['rows'])}")
                            # 显示前几行数据
                            for j, row in enumerate(table["rows"][:3]):
                                formatted_parts.append(f"  行{j+1}: {row}")
                    else:
                        formatted_parts.append(f"  {str(table)[:200]}...")
            
            # 处理图表数据
            if "charts" in data and data["charts"]:
                formatted_parts.append("\n=== 图表数据 ===")
                for i, chart in enumerate(data["charts"][:3]):  # 限制图表数量
                    formatted_parts.append(f"\n图表 {i+1}:")
                    if isinstance(chart, dict):
                        chart_type = chart.get("type", "未知类型")
                        formatted_parts.append(f"  类型: {chart_type}")
                        if "data" in chart:
                            formatted_parts.append(f"  数据点数: {len(chart['data']) if isinstance(chart['data'], list) else '未知'}")
                    else:
                        formatted_parts.append(f"  {str(chart)[:200]}...")
            
            # 处理其他数据
            for key, value in data.items():
                if key not in ["tables", "charts"] and value:
                    formatted_parts.append(f"\n=== {key} ===")
                    if isinstance(value, (list, dict)):
                        formatted_parts.append(f"{str(value)[:500]}...")
                    else:
                        formatted_parts.append(str(value)[:500])
            
            result = "\n".join(formatted_parts)
            logger.debug(f"[FORMAT_STRUCTURED_SUCCESS] 结构化数据格式化完成 - 输出长度: {len(result)} 字符")
            return result
            
        except Exception as e:
            logger.error(f"[FORMAT_STRUCTURED_ERROR] 格式化结构化数据失败: {str(e)}")
            return f"数据格式化失败: {str(e)}"

    async def stream_response(self, question, context, history=None):
        """改进的流式响应，参考ChatGPT实现优化，增加chunk校验"""
        try:
            logger.info(f"[STREAM_RESPONSE] 开始流式响应 - 问题长度: {len(question)} 字符")
            logger.debug(f"[STREAM_QUESTION] 问题内容: {question[:100]}{'...' if len(question) > 100 else ''}")
            
            # 验证问题不为空
            if not question or not question.strip():
                logger.error(f"[STREAM_EMPTY_QUESTION] 问题内容为空")
                yield {"error": "问题内容不能为空", "done": True}
                return
            
            # 构建提示
            logger.info(f"[STREAM_PROMPT_BUILD] 开始构建提示词")
            prompt = self._build_prompt(question, context, history)
            logger.debug(f"[STREAM_PROMPT] 提示词长度: {len(prompt)} 字符")
            
            # 尝试使用真正的流式API
            try:
                # 使用流式生成配置
                stream_config = {
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                    "top_p": 0.8,
                    "top_k": 40
                }
                
                logger.info(f"[STREAM_NATIVE_START] 尝试使用原生流式API")
                # 尝试真正的流式生成
                response = self.model.generate_content(
                    prompt,
                    generation_config=stream_config,
                    safety_settings=self.safety_settings,
                    stream=True  # 启用流式
                )
                
                # 处理真正的流式响应，增加校验
                chunk_count = 0
                for chunk in response:
                    chunk_count += 1
                    if hasattr(chunk, 'text') and chunk.text:
                        # 校验chunk内容
                        validated_chunk = self._validate_stream_chunk(chunk.text)
                        if validated_chunk:
                            logger.debug(f"[STREAM_NATIVE_CHUNK] 原生chunk #{chunk_count} - 长度: {len(validated_chunk)} 字符")
                            yield {
                                "partial_response": validated_chunk,
                                "done": False,
                                "type": "content"
                            }
                
                # 流结束标记 - 这里不是最终确认，只是内容结束
                logger.info(f"[STREAM_NATIVE_COMPLETE] 原生流式完成 - 总chunks: {chunk_count}")
                yield {"partial_response": "", "done": True, "type": "end", "content_complete": True}
                return
                
            except Exception as stream_error:
                logger.warning(f"[STREAM_NATIVE_FAILED] 原生流式失败，降级处理: {str(stream_error)}")
                # 降级到模拟流式
                
            # 降级处理：使用同步方法
            logger.info(f"[STREAM_FALLBACK_START] 开始降级流式处理")
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # 智能分块，参考ChatGPT的实现
            if hasattr(response, 'text'):
                full_text = response.text
                logger.info(f"[STREAM_FALLBACK_SUCCESS] 降级响应成功 - 响应长度: {len(full_text)} 字符")
                # 按语义分块而不是固定字符数
                chunks = self._smart_chunk_text(full_text)
                logger.info(f"[STREAM_CHUNKS] 文本分块完成 - 总块数: {len(chunks)}")
                
                for i, chunk in enumerate(chunks):
                    # 校验每个chunk
                    validated_chunk = self._validate_stream_chunk(chunk)
                    if validated_chunk:
                        logger.debug(f"[STREAM_FALLBACK_CHUNK] 降级chunk #{i+1}/{len(chunks)} - 长度: {len(validated_chunk)} 字符")
                        yield {
                            "partial_response": validated_chunk,
                            "done": i == len(chunks) - 1,
                            "type": "content",
                            "chunk_index": i
                        }
                        # 减少延迟，提高响应速度
                        await asyncio.sleep(0.005)  # 5ms延迟，比原来的10ms更快
                
                logger.info(f"[STREAM_FALLBACK_COMPLETE] 降级流式完成")
                # 注意：这里只是内容完成，不是最终确认
                yield {"partial_response": "", "done": True, "type": "content_end", "content_complete": True}
            else:
                logger.error(f"[STREAM_NO_TEXT] 无法获取响应内容")
                yield {"error": "无法获取响应内容", "done": True}
            
        except Exception as e:
            logger.error(f"[STREAM_ERROR] 流式生成错误: {str(e)}")
            import traceback
            logger.debug(f"[STREAM_TRACEBACK] 错误堆栈: {traceback.format_exc()}")
            
            # 简化的降级处理
            try:
                logger.info(f"[STREAM_SIMPLE_FALLBACK] 尝试简单降级")
                fallback_response = self.model.generate_content(
                    f"请简洁回答：{question}",
                    generation_config={"temperature": 0.7, "max_output_tokens": 512}
                )
                
                if fallback_response and fallback_response.text:
                    logger.info(f"[STREAM_SIMPLE_SUCCESS] 简单降级成功 - 响应长度: {len(fallback_response.text)} 字符")
                    # 快速输出降级响应
                    chunks = self._smart_chunk_text(fallback_response.text)
                    for i, chunk in enumerate(chunks):
                        validated_chunk = self._validate_stream_chunk(chunk)
                        if validated_chunk:
                            logger.debug(f"[STREAM_SIMPLE_CHUNK] 简单降级chunk #{i+1} - 长度: {len(validated_chunk)} 字符")
                            yield {
                                "partial_response": validated_chunk,
                                "done": i == len(chunks) - 1,
                                "fallback_used": True,
                                "type": "content"
                            }
                            await asyncio.sleep(0.003)  # 更快的降级响应
                
                logger.info(f"[STREAM_SIMPLE_FALLBACK_COMPLETE] 简单降级完成")
                yield {"partial_response": "", "done": True, "type": "content_end", "content_complete": True}
                return
            except Exception as fallback_error:
                logger.error(f"[STREAM_SIMPLE_FAILED] 简单降级也失败: {str(fallback_error)}")
            
            yield {
                "error": "服务暂时不可用，请稍后重试",
                "done": True,
                "type": "error"
            }

    def _validate_stream_chunk(self, chunk_text: str) -> str:
        """校验流式chunk的格式和内容，增强异常处理"""
        try:
            # 基本校验：确保是字符串
            if chunk_text is None:
                logger.debug(f"[CHUNK_VALIDATE] chunk为None，返回空字符串")
                return ""
            
            if not isinstance(chunk_text, str):
                logger.warning(f"[CHUNK_VALIDATE] chunk不是字符串类型: {type(chunk_text)}")
                try:
                    # 尝试转换为字符串
                    if hasattr(chunk_text, 'text'):  # Gemini response对象
                        chunk_text = chunk_text.text
                        logger.debug(f"[CHUNK_CONVERT] 从Gemini对象提取文本")
                    elif isinstance(chunk_text, (dict, list)):
                        # 如果是字典或列表，尝试提取文本内容
                        chunk_text = self._extract_text_from_object(chunk_text)
                        logger.debug(f"[CHUNK_CONVERT] 从复杂对象提取文本")
                    else:
                        chunk_text = str(chunk_text)
                        logger.debug(f"[CHUNK_CONVERT] 强制转换为字符串")
                except Exception as convert_error:
                    logger.error(f"[CHUNK_CONVERT_ERROR] 转换chunk为字符串失败: {str(convert_error)}")
                    return ""
            
            # 长度校验：防止异常大的chunk
            if len(chunk_text) > 50000:  # 50KB限制
                logger.warning(f"[CHUNK_LARGE] chunk过大({len(chunk_text)}字符)，截断处理")
                chunk_text = chunk_text[:50000] + "..."
            
            # 内容校验：移除控制字符但保留必要的格式字符
            try:
                # 移除除了换行符、制表符、回车符外的控制字符
                cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', chunk_text)
                
                # 处理特殊的Unicode字符
                cleaned_text = self._handle_unicode_characters(cleaned_text)
                
                logger.debug(f"[CHUNK_CLEAN] chunk清理完成 - 原长度: {len(chunk_text)}, 清理后: {len(cleaned_text)}")
                
            except re.error as regex_error:
                logger.error(f"[CHUNK_REGEX_ERROR] 正则表达式处理chunk失败: {str(regex_error)}")
                # 降级处理：逐字符过滤
                cleaned_text = self._char_by_char_clean(chunk_text)
                logger.debug(f"[CHUNK_CHAR_CLEAN] 使用逐字符清理")
            
            # 检查是否包含有效内容（不只是空白字符）
            if not cleaned_text.strip():
                logger.debug(f"[CHUNK_EMPTY] chunk清理后为空")
                return ""  # 返回空字符串而不是None
            
            # 验证编码完整性
            try:
                # 尝试编码和解码，确保字符串完整
                cleaned_text.encode('utf-8').decode('utf-8')
                logger.debug(f"[CHUNK_ENCODING] chunk编码验证通过")
            except UnicodeError as unicode_error:
                logger.warning(f"[CHUNK_UNICODE_ERROR] Unicode编码错误: {str(unicode_error)}")
                # 修复编码问题
                cleaned_text = self._fix_unicode_errors(cleaned_text)
                logger.debug(f"[CHUNK_UNICODE_FIXED] Unicode错误已修复")
            
            logger.debug(f"[CHUNK_VALIDATE_SUCCESS] chunk校验成功 - 最终长度: {len(cleaned_text)} 字符")
            return cleaned_text
            
        except Exception as validation_error:
            logger.error(f"[CHUNK_VALIDATE_ERROR] Chunk校验失败: {str(validation_error)}")
            import traceback
            logger.debug(f"[CHUNK_VALIDATE_TRACEBACK] 校验错误堆栈: {traceback.format_exc()}")
            
            # 最终降级处理：返回安全的字符串
            try:
                if chunk_text is not None:
                    # 尝试最基本的字符串转换
                    safe_text = str(chunk_text)
                    # 移除明显的问题字符
                    safe_text = ''.join(char for char in safe_text if ord(char) >= 32 or char in '\n\t\r')
                    logger.debug(f"[CHUNK_SAFE_FALLBACK] 使用安全降级 - 长度: {len(safe_text[:1000])}")
                    return safe_text[:1000]  # 限制长度
                else:
                    logger.debug(f"[CHUNK_NULL_FALLBACK] chunk为null，返回空字符串")
                    return ""
            except:
                logger.error(f"[CHUNK_FINAL_FALLBACK] 最终降级也失败")
                return ""

    def _extract_text_from_object(self, obj):
        """从复杂对象中提取文本内容"""
        try:
            logger.debug(f"[TEXT_EXTRACT] 开始从对象提取文本 - 类型: {type(obj)}")
            if isinstance(obj, dict):
                # 尝试常见的文本字段
                for key in ['text', 'content', 'message', 'data', 'response']:
                    if key in obj:
                        logger.debug(f"[TEXT_EXTRACT] 从字典字段 '{key}' 提取文本")
                        return str(obj[key])
                # 如果没有找到，返回整个对象的字符串表示
                logger.debug(f"[TEXT_EXTRACT] 未找到文本字段，返回整个字典的字符串表示")
                return str(obj)
            elif isinstance(obj, list):
                # 如果是列表，尝试连接所有字符串元素
                text_parts = []
                for item in obj:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and 'text' in item:
                        text_parts.append(str(item['text']))
                result = ' '.join(text_parts) if text_parts else str(obj)
                logger.debug(f"[TEXT_EXTRACT] 从列表提取文本 - 找到 {len(text_parts)} 个文本片段")
                return result
            else:
                logger.debug(f"[TEXT_EXTRACT] 直接转换为字符串")
                return str(obj)
        except Exception as e:
            logger.error(f"[TEXT_EXTRACT_ERROR] 提取文本失败: {str(e)}")
            return str(obj)

    def _handle_unicode_characters(self, text: str) -> str:
        """处理特殊的Unicode字符"""
        try:
            original_length = len(text)
            # 处理常见的Unicode问题
            # 替换零宽字符
            text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f\ufeff]', '', text)
            
            # 规范化Unicode
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
            
            logger.debug(f"[UNICODE_HANDLE] Unicode处理完成 - 原长度: {original_length}, 处理后: {len(text)}")
            return text
        except Exception as e:
            logger.error(f"[UNICODE_HANDLE_ERROR] Unicode处理失败: {str(e)}")
            return text

    def _char_by_char_clean(self, text: str) -> str:
        """逐字符清理文本（降级方法）"""
        try:
            original_length = len(text)
            cleaned_chars = []
            for char in text:
                try:
                    # 保留可打印字符和必要的空白字符
                    if char.isprintable() or char in '\n\t\r ':
                        cleaned_chars.append(char)
                except:
                    # 如果字符检查失败，跳过该字符
                    continue
            result = ''.join(cleaned_chars)
            logger.debug(f"[CHAR_CLEAN] 逐字符清理完成 - 原长度: {original_length}, 清理后: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"[CHAR_CLEAN_ERROR] 逐字符清理失败: {str(e)}")
            return text

    def _fix_unicode_errors(self, text: str) -> str:
        """修复Unicode编码错误"""
        try:
            original_length = len(text)
            # 尝试不同的编码修复策略
            # 策略1: 使用errors='ignore'
            fixed_text = text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # 策略2: 如果还有问题，使用errors='replace'
            if not fixed_text:
                fixed_text = text.encode('utf-8', errors='replace').decode('utf-8')
                logger.debug(f"[UNICODE_FIX] 使用replace策略修复")
            else:
                logger.debug(f"[UNICODE_FIX] 使用ignore策略修复")
            
            logger.debug(f"[UNICODE_FIX] Unicode修复完成 - 原长度: {original_length}, 修复后: {len(fixed_text)}")
            return fixed_text
        except Exception as e:
            logger.error(f"[UNICODE_FIX_ERROR] Unicode修复失败: {str(e)}")
            # 最后的降级：只保留ASCII字符
            ascii_text = ''.join(char for char in text if ord(char) < 128)
            logger.debug(f"[UNICODE_FIX_ASCII] 降级为ASCII字符 - 长度: {len(ascii_text)}")
            return ascii_text

    def _smart_chunk_text(self, text: str, target_chunk_size: int = 8) -> List[str]:
        """智能文本分块，参考ChatGPT的语义分块策略"""
        if not text:
            return [""]
        
        # 优先按句子分割
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            # 句子结束标记
            if char in '.!?。！？\n':
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # 添加剩余内容
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # 如果没有明显的句子分割，按词分割
        if len(sentences) <= 1:
            words = text.split()
            chunks = []
            current_chunk = ""
            
            for word in words:
                if len(current_chunk) + len(word) + 1 <= target_chunk_size * 2:
                    current_chunk += (" " if current_chunk else "") + word
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks if chunks else [text]
        
        # 按句子组合成合适大小的块
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= target_chunk_size * 3:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]

    def _build_prompt(self, question, context, history=None, is_general_chat=False):
        """构建提示词，用于统一不同方法的提示词格式"""
        
        if is_general_chat:
            # 通用聊天模式提示词
            history_text = history or "无"
            prompt = f"""你是一个全能的AI助手，能够回答各种问题。无论是一般知识、当前事件、科学探索、文学艺术，还是技术问题，你都可以提供丰富的信息和深入的见解。

历史对话:
{history_text}

用户问题: {question}

请直接回答问题，不需要参考任何文档资料。根据问题性质灵活调整回答风格和深度。"""
        else:
            # 构建基本提示 - 文档分析模式
            prompt = "你是CargoPPT的AI助手，专门用于文档分析和问答，请根据提供的文档内容回答问题。\n\n"
            
            # 添加上下文信息
            if isinstance(context, dict) and "text" in context:
                # 如果是格式化后的上下文对象
                prompt += f"文档内容：\n{context['text']}\n\n"
            elif isinstance(context, list) and len(context) > 0:
                # 如果是上下文列表
                if all(isinstance(item, dict) for item in context):
                    # 多文档情况
                    for doc in context:
                        if "text" in doc:
                            doc_info = f"文档：{doc.get('document_name', '未知文档')}\n"
                            prompt += f"{doc_info}{doc['text']}\n\n"
                else:
                    # 纯文本列表
                    prompt += f"文档内容：\n{' '.join(context)}\n\n"
            elif isinstance(context, str):
                # 如果是字符串
                prompt += f"文档内容：\n{context}\n\n"
        
        # 添加历史对话
        if history:
            if not is_general_chat:  # 文档模式特有
                prompt += f"历史对话：\n{history}\n\n"
        
        # 添加当前问题
        prompt += f"问题：{question}\n\n"
        
        if not is_general_chat:  # 文档模式特有
            prompt += "请根据以上内容提供详细准确的回答。回答要保持严谨且基于文档内容，不要编造信息。"
        
        return prompt
#0531 新增意图识别方法  
    async def identify_intent(self, question: str, context=None, has_documents=False, history=None, document_info=None):
        """简化的意图识别，只返回GENERAL_QUERY或DOCUMENT_QUERY"""
        
        logger.info(f"[INTENT_IDENTIFY] 开始意图识别 - 问题长度: {len(question)} 字符, 有文档: {has_documents}")
        
        try:
            # 调用简化的意图分析器
            intent_result = await self.intent_analyzer.analyze_intent(
                question=question,
                has_documents=has_documents,
                history=history,
                document_info=document_info
            )
            
            logger.info(f"[INTENT_IDENTIFY_RESULT] 意图识别完成 - 结果: {intent_result}")
            return intent_result
            
        except Exception as e:
            logger.error(f"[INTENT_IDENTIFY_ERROR] 意图识别失败: {str(e)}")
            # 失败时的fallback逻辑
            if not has_documents:
                return "GENERAL_QUERY"
            else:
                # 检查简单的文档关键词
                doc_keywords = ["文档", "这篇", "总结", "摘要", "分析"]
                if any(keyword in question.lower() for keyword in doc_keywords):
                    return "DOCUMENT_QUERY"
                else:
                    return "GENERAL_QUERY"
#0531 新增无上下文流式聊天方法
    async def chat_without_context_stream(self, message: str, history: List[Dict[str, str]] = None):
        """无上下文的流式聊天，用于通用对话"""
        logger.info(f"[CHAT_STREAM] 开始无上下文流式聊天 - 消息长度: {len(message)} 字符")
        
        try:
            # 验证消息不为空
            if not message or not message.strip():
                logger.error(f"[CHAT_EMPTY_MESSAGE] 消息内容为空")
                yield {"error": "消息内容不能为空", "done": True}
                return
            
            # 构建对话历史格式
            chat_history = []
            if history and isinstance(history, list):
                logger.info(f"[CHAT_HISTORY] 处理历史消息 - 数量: {len(history)}")
                for i, msg in enumerate(history[-6:]):  # 限制历史长度
                    if isinstance(msg, dict) and "role" in msg and "content" in msg and msg["content"].strip():
                        # 转换为Gemini API所需的格式
                        role = "user" if msg["role"] == "user" else "model"
                        chat_history.append({
                            "role": role,
                            "parts": [{"text": msg["content"].strip()}]
                        })
                        logger.debug(f"[CHAT_HISTORY_ITEM] 历史消息 #{i+1} - 角色: {role}, 长度: {len(msg['content'])} 字符")
                logger.info(f"[CHAT_HISTORY_BUILT] 历史消息构建完成 - 有效消息: {len(chat_history)} 条")
            else:
                logger.info(f"[CHAT_NO_HISTORY] 无历史消息")
            
            # 尝试真正的流式生成
            try:
                stream_config = {
                    "temperature": 0.8,
                    "max_output_tokens": 1024,
                    "top_p": 0.9,
                    "top_k": 40
                }
                
                if not chat_history:
                    # 无历史记录的独立对话
                    logger.info(f"[CHAT_INDEPENDENT] 独立对话模式")
                    system_prompt = f"""你是一个强大的AI助手，能够回答各种问题。请直接回答用户问题，提供准确、有用的信息。

用户问题: {message.strip()}"""
                    
                    logger.debug(f"[CHAT_PROMPT] 系统提示词长度: {len(system_prompt)} 字符")
                    
                    response = self.model.generate_content(
                        system_prompt,
                        generation_config=stream_config,
                        safety_settings=self.safety_settings,
                        stream=True
                    )
                else:
                    # 有历史记录的连续对话
                    logger.info(f"[CHAT_CONTINUOUS] 连续对话模式")
                    chat = self.model.start_chat(history=chat_history)
                    response = chat.send_message(
                        message.strip(),
                        generation_config=stream_config,
                        safety_settings=self.safety_settings,
                        stream=True
                    )
                
                # 处理真正的流式响应，增加校验
                logger.info(f"[CHAT_NATIVE_START] 开始原生流式处理")
                chunk_count = 0
                for chunk in response:
                    chunk_count += 1
                    if hasattr(chunk, 'text') and chunk.text:
                        # 校验chunk内容
                        validated_chunk = self._validate_stream_chunk(chunk.text)
                        if validated_chunk:
                            logger.debug(f"[CHAT_NATIVE_CHUNK] 原生chunk #{chunk_count} - 长度: {len(validated_chunk)} 字符")
                            yield {
                                "partial_response": validated_chunk,
                                "done": False,
                                "type": "content"
                            }
                
                logger.info(f"[CHAT_NATIVE_COMPLETE] 原生流式完成 - 总chunks: {chunk_count}")
                yield {"partial_response": "", "done": True, "type": "end"}
                return
                
            except Exception as stream_error:
                logger.warning(f"[CHAT_NATIVE_FAILED] 原生流式失败，降级处理: {str(stream_error)}")
            
            # 降级到模拟流式
            logger.info(f"[CHAT_FALLBACK_START] 开始降级流式处理")
            if not chat_history:
                system_prompt = f"""你是一个强大的AI助手，能够回答各种问题，包括一般知识问题、深度解析问题和专业领域问题。
                
你应该:
1. 直接回答用户问题，提供准确、有用的信息
2. 使用清晰、自然的语言进行交流
3. 在适当时使用Markdown格式增强可读性
4. 当涉及专业知识时，提供深入的解析
5. 当不确定答案时，诚实表明

用户问题: {message.strip()}

请直接回答上述问题，不需要提及文档。"""
                
                logger.debug(f"[CHAT_FALLBACK_PROMPT] 降级提示词长度: {len(system_prompt)} 字符")
                
                response = self.model.generate_content(
                    system_prompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
            else:
                # 使用历史对话开始聊天
                logger.info(f"[CHAT_FALLBACK_HISTORY] 使用历史对话降级处理")
                chat = self.model.start_chat(history=chat_history)
                response = chat.send_message(
                    message.strip(),
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
            
            # 使用智能分块处理响应，增加校验
            if hasattr(response, 'text'):
                full_text = response.text
                logger.info(f"[CHAT_FALLBACK_SUCCESS] 降级响应成功 - 响应长度: {len(full_text)} 字符")
                chunks = self._smart_chunk_text(full_text, target_chunk_size=10)  # 通用聊天可以稍大一些
                logger.info(f"[CHAT_CHUNKS] 文本分块完成 - 总块数: {len(chunks)}")
                
                for i, chunk in enumerate(chunks):
                    # 校验每个chunk
                    validated_chunk = self._validate_stream_chunk(chunk)
                    if validated_chunk:
                        logger.debug(f"[CHAT_FALLBACK_CHUNK] 降级chunk #{i+1}/{len(chunks)} - 长度: {len(validated_chunk)} 字符")
                        yield {
                            "partial_response": validated_chunk,
                            "done": i == len(chunks) - 1,
                            "type": "content",
                            "chunk_index": i
                        }
                        await asyncio.sleep(0.008)  # 通用聊天稍慢一点，更自然
                
                logger.info(f"[CHAT_FALLBACK_COMPLETE] 降级流式完成")
            else:
                logger.error(f"[CHAT_NO_TEXT] 无法获取响应内容")
                yield {"error": "无法获取响应内容", "done": True}
                
        except Exception as e:
            logger.error(f"[CHAT_ERROR] 通用聊天流式生成错误: {str(e)}")
            import traceback
            logger.debug(f"[CHAT_TRACEBACK] 错误堆栈: {traceback.format_exc()}")
            
            # 简化的降级处理
            try:
                logger.info(f"[CHAT_SIMPLE_FALLBACK] 尝试简单降级")
                fallback_response = self.model.generate_content(
                    f"请回答：{message}",
                    generation_config={"temperature": 0.8, "max_output_tokens": 512}
                )
                
                if fallback_response and fallback_response.text:
                    logger.info(f"[CHAT_SIMPLE_SUCCESS] 简单降级成功 - 响应长度: {len(fallback_response.text)} 字符")
                    chunks = self._smart_chunk_text(fallback_response.text)
                    for i, chunk in enumerate(chunks):
                        validated_chunk = self._validate_stream_chunk(chunk)
                        if validated_chunk:
                            logger.debug(f"[CHAT_SIMPLE_CHUNK] 简单降级chunk #{i+1} - 长度: {len(validated_chunk)} 字符")
                            yield {
                                "partial_response": validated_chunk,
                                "done": i == len(chunks) - 1,
                                "fallback_used": True,
                                "type": "content"
                            }
                            await asyncio.sleep(0.005)
                    return
            except Exception as fallback_error:
                logger.error(f"[CHAT_SIMPLE_FAILED] 简单降级也失败: {str(fallback_error)}")
            
            # 最终错误处理
            yield {
                "error": "服务暂时不可用，请稍后重试",
                "done": True,
                "type": "error"
            }

    def _parse_response_content(self, response_text: str) -> List[MessageContent]:
        """解析AI响应内容为富文本格式
        
        参数:
            response_text: AI生成的响应文本
            
        返回:
            富文本内容列表
        """
        try:
            if not response_text or not response_text.strip():
                logger.debug(f"[PARSE_CONTENT] 响应文本为空")
                return [{"type": "markdown", "content": ""}]
            
            logger.debug(f"[PARSE_CONTENT] 开始解析响应内容 - 长度: {len(response_text)} 字符")
            
            # 检测代码块
            code_pattern = r'```(\w+)?\n?(.*?)\n?```'
            code_matches = re.findall(code_pattern, response_text, re.DOTALL)
            
            if code_matches:
                logger.debug(f"[PARSE_CONTENT] 检测到 {len(code_matches)} 个代码块")
                
                # 分割文本并处理代码块
                parts = []
                current_pos = 0
                
                for match in re.finditer(code_pattern, response_text, re.DOTALL):
                    # 添加代码块之前的文本
                    before_text = response_text[current_pos:match.start()].strip()
                    if before_text:
                        parts.append({"type": "markdown", "content": before_text})
                    
                    # 添加代码块
                    language = match.group(1) or "text"
                    code_content = match.group(2).strip()
                    parts.append({
                        "type": "code",
                        "content": code_content,
                        "language": language
                    })
                    
                    current_pos = match.end()
                
                # 添加剩余文本
                remaining_text = response_text[current_pos:].strip()
                if remaining_text:
                    parts.append({"type": "markdown", "content": remaining_text})
                
                return parts
            
            # 检测表格（简单的管道分隔表格）
            table_pattern = r'\|.*?\|.*?\n(?:\|.*?\|.*?\n)+'
            table_matches = re.findall(table_pattern, response_text, re.MULTILINE)
            
            if table_matches:
                logger.debug(f"[PARSE_CONTENT] 检测到 {len(table_matches)} 个表格")
                
                # 简化处理：暂时作为markdown返回
                return [{"type": "markdown", "content": response_text}]
            
            # 默认作为markdown处理
            logger.debug(f"[PARSE_CONTENT] 作为markdown内容处理")
            return [{"type": "markdown", "content": response_text}]
            
        except Exception as e:
            logger.error(f"[PARSE_CONTENT_ERROR] 内容解析失败: {str(e)}")
            # 降级处理：返回原始文本
            return [{"type": "markdown", "content": response_text or ""}]


#0531 新增网页分析方法
    async def analyze_web_content(self, question: str, web_context: dict, history: str = None) -> dict:
        """分析网页内容 - 新增方法"""
        try:
            logger.info(f"[AI_WEB_ANALYZE] 开始分析网页内容")
            
            # 构建网页专用prompt
            web_prompt = self._build_web_prompt(question, web_context, history)
            
            # 调用Gemini分析
            response = await self.model.generate_content_async(web_prompt)
            response_text = response.text
            
            # 提取网页信息
            web_info = self._extract_web_info(web_context)
            
            # 解析响应内容
            parsed_content = self._parse_response_content(response_text)
            
            result = {
                "answer": response_text,
                "sources": web_context.get("chunks", [])[:5],  # 限制来源数量
                "confidence": 0.85,  # 网页分析通常置信度较高
                "reply": parsed_content,
                "web_info": web_info,
                "analysis_type": "web_content"
            }
            
            logger.info(f"[AI_WEB_ANALYZE_SUCCESS] 网页内容分析完成")
            return result
            
        except Exception as e:
            logger.error(f"[AI_WEB_ANALYZE_ERROR] 网页内容分析失败: {str(e)}")
            raise Exception(f"网页内容分析失败: {str(e)}")

    def _build_web_prompt(self, question: str, web_context: dict, history: str = None) -> str:
        """构建网页分析专用prompt"""
        try:
            # 提取网页基本信息
            web_info = self._extract_web_info(web_context)
            
            # 构建上下文文本
            context_text = ""
            if web_context.get("chunks"):
                for i, chunk in enumerate(web_context["chunks"][:8]):  # 最多8个片段
                    chunk_content = chunk.get("content", "")
                    source_url = chunk.get("source_url", "")
                    context_text += f"【网页片段 {i+1}】\n{chunk_content}\n来源: {source_url}\n\n"
            
            # 构建历史对话上下文
            history_context = ""
            if history:
                history_context = f"\n\n## 📖 对话历史\n{history}\n"
            
            # 网页分析专用prompt
            prompt = f"""# 🌐 网页内容智能分析助手

## 📋 任务说明
你是一个专业的网页内容分析专家，需要基于提供的网页内容回答用户问题。

## 🔍 网页信息
- **标题**: {web_info.get('title', '未知')}
- **URL**: {web_info.get('url', '未知')}
- **内容类型**: 网页文档
- **分析时间**: {web_info.get('analyzed_at', '未知')}

## 📄 网页内容片段
{context_text}

{history_context}

## ❓ 用户问题
{question}

## 📋 分析要求
1. **🎯 准确回答**: 基于网页内容准确回答用户问题
2. **🔗 引用来源**: 引用具体的网页片段和URL
3. **📊 结构化输出**: 使用清晰的格式组织答案
4. **🌐 网页特性**: 考虑网页内容的实时性和链接性
5. **💡 深度分析**: 提供有价值的洞察和总结

## 🎨 回答格式
请使用以下格式回答：

### 📝 **问题解答**
[基于网页内容的直接回答]

### 🔍 **详细分析**
[深入分析和解释]

### 🔗 **相关链接**
[如果有相关链接，列出来]

### 💡 **总结建议**
[基于网页内容的总结和建议]

请开始你的分析："""

            logger.debug(f"[AI_WEB_PROMPT] 网页分析prompt构建完成")
            return prompt
            
        except Exception as e:
            logger.error(f"[AI_WEB_PROMPT_ERROR] 网页prompt构建失败: {str(e)}")
            # 返回基础prompt
            return f"请基于以下网页内容回答问题：\n\n{question}\n\n网页内容：\n{web_context}"

    def _extract_web_info(self, web_context: dict) -> dict:
        """提取网页信息"""
        try:
            web_info = {
                "title": "未知网页",
                "url": "未知",
                "analyzed_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "chunk_count": 0,
                "content_length": 0
            }
            
            # 从chunks中提取信息
            if web_context.get("chunks"):
                chunks = web_context["chunks"]
                web_info["chunk_count"] = len(chunks)
                
                # 提取URL和标题
                for chunk in chunks:
                    if chunk.get("source_url"):
                        web_info["url"] = chunk["source_url"]
                        break
                
                # 计算总内容长度
                total_length = sum(len(chunk.get("content", "")) for chunk in chunks)
                web_info["content_length"] = total_length
            
            # 从文档信息中提取标题
            if web_context.get("document_name"):
                web_info["title"] = web_context["document_name"]
            elif web_context.get("text"):
                # 从内容中提取标题（取前50个字符）
                content = web_context["text"][:50]
                web_info["title"] = content.split('\n')[0] if content else "未知网页"
            
            return web_info
            
        except Exception as e:
            logger.error(f"[AI_WEB_INFO_ERROR] 提取网页信息失败: {str(e)}")
            return {"title": "未知网页", "url": "未知", "analyzed_at": "未知"}

    async def analyze_web_vs_document(self, question: str, web_context: dict, 
                                     doc_context: dict, history: str = None) -> dict:
        """网页与文档对比分析 - 新增方法"""
        try:
            logger.info(f"[AI_WEB_DOC_COMPARE] 开始网页与文档对比分析")
            
            # 构建对比分析prompt
            compare_prompt = self._build_web_doc_compare_prompt(
                question, web_context, doc_context, history
            )
            
            # 调用Gemini分析
            response = await self.model.generate_content_async(compare_prompt)
            response_text = response.text
            
            # 提取信息
            web_info = self._extract_web_info(web_context)
            doc_info = self._extract_document_info(doc_context)
            
            # 解析响应内容
            parsed_content = self._parse_response_content(response_text)
            
            # 合并来源
            combined_sources = []
            if web_context.get("chunks"):
                for chunk in web_context["chunks"][:3]:
                    chunk_copy = chunk.copy()
                    chunk_copy["source_type"] = "web"
                    combined_sources.append(chunk_copy)
            
            if doc_context.get("chunks"):
                for chunk in doc_context["chunks"][:3]:
                    chunk_copy = chunk.copy()
                    chunk_copy["source_type"] = "document"
                    combined_sources.append(chunk_copy)
            
            result = {
                "answer": response_text,
                "sources": combined_sources,
                "confidence": 0.9,  # 对比分析置信度更高
                "reply": parsed_content,
                "web_info": web_info,
                "doc_info": doc_info,
                "analysis_type": "web_document_comparison"
            }
            
            logger.info(f"[AI_WEB_DOC_COMPARE_SUCCESS] 网页与文档对比分析完成")
            return result
            
        except Exception as e:
            logger.error(f"[AI_WEB_DOC_COMPARE_ERROR] 网页与文档对比分析失败: {str(e)}")
            raise Exception(f"网页与文档对比分析失败: {str(e)}")

    def _build_web_doc_compare_prompt(self, question: str, web_context: dict, 
                                     doc_context: dict, history: str = None) -> str:
        """构建网页与文档对比分析prompt"""
        try:
            # 提取信息
            web_info = self._extract_web_info(web_context)
            doc_info = self._extract_document_info(doc_context)
            
            # 构建网页内容
            web_content = ""
            if web_context.get("chunks"):
                for i, chunk in enumerate(web_context["chunks"][:5]):
                    web_content += f"【网页片段 {i+1}】\n{chunk.get('content', '')}\n\n"
            
            # 构建文档内容
            doc_content = ""
            if doc_context.get("chunks"):
                for i, chunk in enumerate(doc_context["chunks"][:5]):
                    doc_content += f"【文档片段 {i+1}】\n{chunk.get('content', '')}\n\n"
            
            # 历史对话
            history_context = f"\n\n## 📖 对话历史\n{history}\n" if history else ""
            
            prompt = f"""# 🔄 网页与文档对比分析助手

## 📋 任务说明
你需要对比分析网页内容和文档内容，为用户提供综合性的答案。

## 🌐 网页信息
- **标题**: {web_info.get('title', '未知')}
- **URL**: {web_info.get('url', '未知')}
- **类型**: 网页内容

### 网页内容：
{web_content}

## 📄 文档信息  
- **文档名**: {doc_info.get('title', '未知')}
- **类型**: {doc_info.get('type', '文档')}

### 文档内容：
{doc_content}

{history_context}

## ❓ 用户问题
{question}

## 📋 对比分析要求
1. **🔍 内容对比**: 比较网页和文档中的相关信息
2. **✅ 一致性分析**: 找出信息的一致性和差异性
3. **⏰ 时效性考量**: 考虑网页内容的实时性优势
4. **📊 权威性评估**: 评估不同来源的可靠性
5. **💡 综合建议**: 提供基于两种来源的综合建议

## 🎨 回答格式
### 📝 **综合回答**
[基于网页和文档的综合答案]

### 🔍 **对比分析**
| 对比维度 | 网页内容 | 文档内容 | 分析说明 |
|---------|---------|---------|---------|
| 信息一致性 | ... | ... | ... |
| 详细程度 | ... | ... | ... |
| 时效性 | ... | ... | ... |

### 🌐 **网页优势**
- [网页内容的独特价值]

### 📄 **文档优势**  
- [文档内容的独特价值]

### 💡 **综合建议**
[基于两种来源的最终建议]

请开始你的对比分析："""

            logger.debug(f"[AI_WEB_DOC_PROMPT] 对比分析prompt构建完成")
            return prompt
            
        except Exception as e:
            logger.error(f"[AI_WEB_DOC_PROMPT_ERROR] 对比prompt构建失败: {str(e)}")
            return f"请对比分析以下网页和文档内容来回答问题：\n\n{question}"

    async def identify_web_intent(self, question: str, web_context: dict = None) -> str:
        """识别网页相关的意图 - 新增方法"""
        try:
            logger.info(f"[AI_WEB_INTENT] 开始识别网页相关意图")
            
            # 网页特定的意图模式
            web_patterns = {
                "WEB_LINK_ANALYSIS": [
                    "链接", "超链接", "跳转", "相关链接", "参考链接", "外部链接"
                ],
                "WEB_CONTENT_EXTRACT": [
                    "提取", "摘要", "总结", "关键信息", "主要内容"
                ],
                "WEB_STRUCTURE_ANALYSIS": [
                    "结构", "布局", "组织", "章节", "目录", "导航"
                ],
                "WEB_COMPARISON": [
                    "对比", "比较", "差异", "相同", "不同", "异同"
                ],
                "WEB_REAL_TIME": [
                    "最新", "实时", "当前", "现在", "最近", "更新"
                ]
            }
        
            question_lower = question.lower()
            
            # 模式匹配
            for intent, patterns in web_patterns.items():
                if any(pattern in question_lower for pattern in patterns):
                    logger.info(f"[AI_WEB_INTENT_FOUND] 识别到网页意图: {intent}")
                    return intent
            
            # 如果有网页上下文，默认为网页查询
            if web_context and web_context.get("chunks"):
                logger.info(f"[AI_WEB_INTENT_DEFAULT] 默认网页查询意图")
                return "WEB_QUERY"
            
            logger.info(f"[AI_WEB_INTENT_GENERAL] 通用意图")
            return "GENERAL_QUERY"
            
        except Exception as e:
            logger.error(f"[AI_WEB_INTENT_ERROR] 网页意图识别失败: {str(e)}")
            return "GENERAL_QUERY"