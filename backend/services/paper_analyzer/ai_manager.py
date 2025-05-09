import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import TypedDict, List, Optional

class SourceInfo(TypedDict):
    line_number: int
    content: str
    page: int
    start_pos: int
    end_pos: int
    is_scanned: bool
    similarity: float

class AIResponse(TypedDict):
    answer: str
    sources: List[SourceInfo]
    confidence: float

class AIManager:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        # 配置 Gemini API
        genai.configure(api_key=self.api_key)
        
        # 使用正确的模型名称
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    async def get_response(self, question: str, context: str) -> dict:
        try:
            if not context:
                return {
                    "answer": "抱歉，我无法找到相关的上下文信息来回答这个问题。",
                    "sources": [],
                    "confidence": 0.0
                }

            # 构建提示词
            prompt = f"""你是一个专业的论文分析助手。请基于以下上下文回答问题。

上下文：
{context}

问题：
{question}

要求：
1. 回答要准确、专业，直接基于上下文中的信息
2. 如果上下文中的信息不足以回答问题，请明确说明
3. 如果问题与上下文无关，请说明并建议用户提供更多相关信息
4. 回答要简洁明了，避免冗长的解释
5. 如果上下文中包含多个相关段落，请综合这些信息给出完整回答
6. 如果发现上下文中的信息有矛盾，请指出并说明

请提供你的回答："""
            
            # 使用新的 API 调用方式
            response = self.model.generate_content(prompt)
            
            # 检查响应
            if not response or not response.text:
                return {
                    "answer": "抱歉，我无法生成有效的回答。请尝试重新提问或检查文档内容。",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # 返回结构化数据，确保所有数值都是 Python 原生类型
            return {
                "answer": str(response.text),
                "sources": [
                    {
                        "line_number": int(source.get("line_number", 0)),
                        "content": str(source.get("content", "")),
                        "page": int(source.get("page", 1)),
                        "start_pos": int(source.get("start_pos", 0)),
                        "end_pos": int(source.get("end_pos", 0)),
                        "is_scanned": bool(source.get("is_scanned", False)),
                        "similarity": float(source.get("similarity", 0.0))
                    }
                    for source in context.get("chunks", [])
                ],
                "confidence": float(getattr(response, 'confidence', 0.8))
            }
            
        except Exception as e:
            print(f"AI response error details: {str(e)}")
            return {
                "answer": f"抱歉，生成回答时出现错误：{str(e)}",
                "sources": [],
                "confidence": 0.0
            }
