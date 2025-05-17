import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Union, Dict, Any, Literal
import json
import re
import asyncio

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
        
        # 配置模型参数
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # 添加安全设置
        self.safety_settings = {
            "harassment": "block_medium_and_above",
            "hate_speech": "block_medium_and_above", 
            "sexually_explicit": "block_medium_and_above",
            "dangerous_content": "block_medium_and_above"
        }

    async def get_response(self, question: str, context: Union[str, Dict, List[Dict]], history: str = None) -> dict:
        """
        从文档上下文获取回答
        
        参数:
            question: 用户问题
            context: 文档上下文，可以是字符串、单个文档上下文字典或多个文档上下文列表
            history: 对话历史
        """
        try:
            # 使用统一的提示词构建方法
            prompt = self._build_prompt(question, context, history)
            
            # 发送到模型生成回答
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # 检查响应
            if not response or not response.text:
                return {
                    "answer": "抱歉，我无法生成有效的回答。请尝试重新提问或检查文档内容。",
                    "sources": [],
                    "confidence": 0.0,
                    "reply": [
                        {
                            "type": "markdown",
                            "content": "抱歉，我无法生成有效的回答。请尝试重新提问或检查文档内容。"
                        }
                    ]
                }
            
            # 解析响应内容，识别不同类型的内容
            reply_content = self._parse_response_content(response.text)
            
            # 返回结构化数据，同时兼容旧格式和新格式
            response_data = {
                "answer": str(response.text),  # 保持兼容性
                "sources": [
                    {
                        "line_number": int(source.get("line_number", 0)),
                        "content": str(source.get("content", "")),
                        "page": int(source.get("page", 1)),
                        "start_pos": int(source.get("start_pos", 0)),
                        "end_pos": int(source.get("end_pos", 0)),
                        "is_scanned": bool(source.get("is_scanned", False)),
                        "similarity": float(source.get("similarity", 0.0)),
                        "document_id": source.get("document_id", None),
                        "document_name": source.get("document_name", None)
                    }
                    for source in sources
                ],
                "confidence": float(getattr(response, 'confidence', 0.8)),
                "reply": reply_content  # 确保这是通过 _parse_response_content 转换的结构
            }
            
            # 计算响应大小
            response_size = len(json.dumps(response_data, ensure_ascii=False))
            print(f"Generated AI response with size: {response_size} bytes")
            if response_size > 50000:  # 50KB threshold for tracking large responses
                print(f"WARNING: Large response generated!")
            
            return response_data
            
        except Exception as e:
            print(f"AI response error details: {str(e)}")
            error_message = f"抱歉，生成回答时出现错误：{str(e)}"
            return {
                "answer": error_message,
                "sources": [],
                "confidence": 0.0,
                "reply": [{"type": "markdown", "content": error_message}]
            }

    def _parse_response_content(self, text: str) -> List[MessageContent]:
        """解析响应内容，识别不同类型的内容块"""
        result = []
        
        # 通过正则表达式识别不同类型的内容
        # 新增：专门处理Mermaid图表模式
        mermaid_pattern = r'```mermaid\n([\s\S]*?)\n```'
        # 原有代码块模式
        code_pattern = r'```(\w*)\n([\s\S]*?)\n```'
        # 表格模式
        table_pattern = r'\n\|(.+?)\|\n\|([-:]+\|)+\n((?:\|.+?\|\n)+)'
        
        # 临时存储已处理的内容范围
        processed_ranges = []
        
        # 首先处理Mermaid图表（优先级更高）
        for match in re.finditer(mermaid_pattern, text):
            diagram_content = match.group(1)
            start, end = match.span()
            processed_ranges.append((start, end))
            
            # 检测是否为心智图或流程图
            diagram_type = "flowchart"  # 默认为流程图
            if diagram_content.strip().startswith("mindmap"):
                diagram_type = "mindmap"  # 心智图类型
            
            result.append({
                "type": "code",  # 正确，与前端匹配
                "content": diagram_content.strip(),
                "language": "mermaid",  # 正确，与前端匹配
                "metadata": {  # 可以保留 metadata，前端有对应字段
                    "diagram_type": diagram_type
                }
            })
        
        # 处理其他代码块
        for match in re.finditer(code_pattern, text):
            language, code = match.groups()
            start, end = match.span()
            
            # 检查这个区域是否已经作为Mermaid图表处理过
            skip = False
            for s, e in processed_ranges:
                if (start >= s and start < e) or (end > s and end <= e):
                    skip = True
                    break
            
            if skip:
                continue
                
            # 不是Mermaid图表，也不是已处理过的区域
            processed_ranges.append((start, end))
            
            # 额外检查：如果是mermaid代码但没被前面的规则捕获
            if language.lower() == 'mermaid' and not code.strip().startswith("mindmap") and "graph" not in code.strip().lower():
                # 这是一个不符合预期的mermaid代码，可能需要特殊处理
                result.append({
                    "type": "code",
                    "content": code.strip(),
                    "language": "mermaid",
                    "metadata": {
                        "diagram_type": "unknown"
                    }
                })
            else:
                # 普通代码块
                result.append({
                    "type": "code",
                    "content": code.strip(),
                    "language": language or "text"  # 确保始终有语言类型
                })
        
        # 处理表格 - 保持原逻辑
        for match in re.finditer(table_pattern, text):
            header = match.group(1)
            rows_text = match.group(3)
            start, end = match.span()
            
            # 检查是否与已处理的范围重叠
            skip = False
            for s, e in processed_ranges:
                if start < e and end > s:  # 范围重叠
                    skip = True
                    break
            
            if skip:
                continue
                
            processed_ranges.append((start, end))
            
            # 解析表格标题和行
            columns = [col.strip() for col in header.split('|') if col.strip()]
            
            # 解析表格行
            table_rows = []
            for row in rows_text.strip().split('\n'):
                cells = [cell.strip() for cell in row.split('|')[1:-1]]  # 去掉首尾的 |
                if cells:
                    table_rows.append(cells)
            
            result.append({
                "type": "table",
                "content": match.group(0).strip(),
                "columns": columns,
                "rows": table_rows
            })
        
        # 处理剩余文本 - 保持原逻辑
        remaining_text = text
        # 按照起始位置排序处理过的范围
        processed_ranges.sort()
        
        last_end = 0
        for start, end in processed_ranges:
            if start > last_end:
                # 添加前面的文本段
                text_segment = remaining_text[last_end:start].strip()
                if text_segment:
                    result.append({
                        "type": "markdown",  # 而不是 "text"，统一使用 markdown
                        "content": text_segment
                    })
            last_end = end
        
        # 添加最后一段
        if last_end < len(remaining_text):
            text_segment = remaining_text[last_end:].strip()
            if text_segment:
                result.append({
                    "type": "markdown",  # 而不是 "text"，统一使用 markdown
                    "content": text_segment
                })
        
        # 如果没有识别出任何特殊格式，将整个文本作为markdown
        if not result:
            result.append({
                "type": "markdown",
                "content": text
            })
        
        return result

    async def chat_without_context(self, message: str, history: List[Dict[str, str]] = None) -> dict:
        """
        处理无文档的纯AI对话
        
        参数:
            message: 用户消息
            history: 历史消息列表，每个消息为 {"role": "user|assistant", "content": "消息内容"}
        """
        try:
            # 准备历史消息
            chat_history = []
            if history and isinstance(history, list):
                for msg in history:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        # 转换为Gemini API所需的格式
                        role = "user" if msg["role"] == "user" else "model"
                        chat_history.append({
                            "role": role,
                            "parts": [{"text": msg["content"]}]
                        })
            
            # 添加当前消息
            chat_history.append({
                "role": "user",
                "parts": [{"text": message}]
            })
            
            # 如果没有历史，添加系统指令
            if not history or len(history) == 0:
                system_prompt = """你是一个专业的AI助手，能够回答各种问题并提供帮助。你的回答应当：
1. 准确、专业、有帮助性
2. 简洁明了，避免冗长的解释
3. 有礼貌、有耐心
4. 如果不确定或不知道答案，诚实说明
5. 避免编造信息或提供误导性回答
6. 不讨论政治敏感话题

在回答中，你可以根据内容性质使用不同的格式：
- 使用普通文本回答简单问题
- 使用Markdown格式增强可读性
- 对于代码示例，使用代码块并标明语言
- 对于表格数据，使用Markdown表格格式

请用自然、专业的语气回答用户问题。"""
                
                # 使用系统提示初始化对话
                response = self.model.generate_content(
                    system_prompt + "\n\n" + message,
                    generation_config=self.generation_config
                )
            else:
                # 使用历史对话继续对话
                chat = self.model.start_chat(history=chat_history[:-1])
                response = chat.send_message(
                    message,
                    generation_config=self.generation_config
                )
            
            # 检查响应
            if not response or not response.text:
                return {
                    "answer": "抱歉，我无法生成有效的回答。请尝试重新提问。",
                    "confidence": 0.0,
                    "sources": [],
                    "reply": [
                        {
                            "type": "markdown",
                            "content": "抱歉，我无法生成有效的回答。请尝试重新提问。"
                        }
                    ]
                }
            
            # 解析回复内容，识别不同类型的内容
            reply_content = self._parse_response_content(response.text)
            
            # 返回结构化数据，兼容旧格式并提供新格式
            return {
                "answer": str(response.text),  # 兼容旧格式
                "confidence": float(getattr(response, 'confidence', 0.8)),
                "sources": [],  # 无文档对话没有来源
                "reply": reply_content  # 新的富内容格式
            }
            
        except Exception as e:
            print(f"AI chat error details: {str(e)}")
            error_message = f"抱歉，生成回答时出现错误：{str(e)}"
            return {
                "answer": error_message,
                "sources": [],
                "confidence": 0.0,
                "reply": [{"type": "markdown", "content": error_message}]
            }

    async def get_multi_document_response(self, question: str, document_contexts: List[Dict], history: str = None) -> dict:
        """
        从多个文档上下文获取统一回答
        
        参数:
            question: 用户问题
            document_contexts: 多个文档的上下文列表，每个文档包含id、名称和上下文
            history: 对话历史
        """
        # 处理并合并多文档上下文，然后调用get_response
        return await self.get_response(question, document_contexts, history)

    async def analyze_structured_data(self, query: str, structured_data: dict, paper_id: str = None, is_sampled: bool = False) -> dict:
        """
        分析Excel等结构化数据，执行统计、筛选等操作
        """
        if is_sampled:
            print(f"Processing sampled structured data for query: {query}")
            # 添加采样警告
            sampling_warning = "\n\n⚠️ **注意**: 由于数据量较大，分析基于数据采样。结果可能反映数据的总体趋势，但不保证精确的统计值。"
        
        try:
            if not structured_data:
                return {
                    "answer": "未找到结构化数据可供分析",
                    "sources": [],
                    "confidence": 0.0,
                    "reply": [{"type": "markdown", "content": "未找到结构化数据可供分析"}]
                }
            
            # 提取sheet名称和元数据名称
            sheet_names = []
            metadata_keys = []
            for key in structured_data.keys():
                if key.endswith("_metadata"):
                    metadata_keys.append(key)
                else:
                    sheet_names.append(key)
            
            # 构建提示词
            sheets_info = []
            # 添加总体概览
            overview = f"Excel文件包含 {len(sheet_names)} 个工作表: {', '.join(sheet_names)}"
            sheets_info.append(overview)
            
            for sheet_name in sheet_names:
                metadata_key = f"{sheet_name}_metadata"
                data = structured_data.get(sheet_name, [])
                metadata = structured_data.get(metadata_key, {})
                
                if not isinstance(data, list) or not data:
                    sheets_info.append(f"工作表: {sheet_name} [空表格或格式错误]")
                    continue
                
                # 获取元数据信息
                total_rows = metadata.get("total_rows", len(data))
                total_columns = metadata.get("total_columns", len(data[0].keys()) if data and data[0] else 0)
                data_range = metadata.get("data_range", f"1-{total_rows}")
                columns = metadata.get("columns", list(data[0].keys()) if data and data[0] else [])
                
                # 添加表格基本信息
                sheets_info.append(f"工作表: {sheet_name}")
                sheets_info.append(f"- 行数: {total_rows}")
                sheets_info.append(f"- 列数: {total_columns}")
                sheets_info.append(f"- 数据范围: {data_range}")
                sheets_info.append(f"- 列名: {', '.join(columns)}")
                
                # 添加列详细信息
                column_metadata = metadata.get("column_metadata", {})
                column_details = []
                
                for col, col_meta in column_metadata.items():
                    col_type = col_meta.get("type", "未知")
                    col_info = f"  - {col} ({col_type})"
                    
                    # 根据列类型添加特定统计信息
                    if col_type == "numeric":
                        min_val = col_meta.get("min", "N/A")
                        max_val = col_meta.get("max", "N/A")
                        avg_val = col_meta.get("avg", "N/A")
                        if min_val != "N/A":
                            col_info += f", 最小值: {min_val}, 最大值: {max_val}, 平均值: {round(avg_val, 2)}"
                    
                    # 添加数据完整性信息
                    non_empty = col_meta.get("non_empty_count", 0)
                    empty = col_meta.get("empty_count", 0)
                    if non_empty > 0 or empty > 0:
                        col_info += f", 非空值: {non_empty}, 空值: {empty}"
                    
                    # 文本类型添加唯一值数量
                    if col_type == "text" and "unique_count" in col_meta:
                        col_info += f", 唯一值数量: {col_meta['unique_count']}"
                    
                    column_details.append(col_info)
                
                if column_details:
                    sheets_info.append("列详细信息:")
                    sheets_info.extend(column_details)
                
                # 添加示例数据 - 只显示前3行作为示例
                sample_size = min(3, len(data))
                if sample_size > 0:
                    sheets_info.append(f"\n{sheet_name} 数据示例 (仅显示前 {sample_size} 行，实际共 {total_rows} 行):")
                    
                    # 创建示例数据的表格形式，便于阅读
                    sample_table = "| " + " | ".join(columns) + " |\n"
                    sample_table += "| " + " | ".join(["---"] * len(columns)) + " |\n"
                    
                    for i in range(sample_size):
                        row = data[i]
                        sample_table += "| " + " | ".join([str(row.get(col, "")) for col in columns]) + " |\n"
                    
                    sheets_info.append(sample_table)
                    
                    # 明确标注还有更多数据
                    if total_rows > sample_size:
                        sheets_info.append(f"... 还有 {total_rows - sample_size} 行数据未显示（共 {total_rows} 行） ...")
            
            sheets_summary = "\n\n".join(sheets_info)
            
            # 检测查询类型，添加特定指令
            query_type = "general"
            if any(term in query.lower() for term in ["统计", "计算", "求和", "平均", "总数", "合计"]):
                query_type = "aggregation"
            elif any(term in query.lower() for term in ["筛选", "过滤", "符合条件", "查找"]):
                query_type = "filtering"
            elif any(term in query.lower() for term in ["排序", "排列", "从大到小", "从小到大"]):
                query_type = "sorting"
            
            # 根据查询类型添加特定指令
            query_instructions = ""
            if query_type == "aggregation":
                query_instructions = """
特别提示 - 数值计算:
1. 请确保对整个数据集进行统计，而不仅是示例数据
2. 计算总和/平均值等统计量时，请考虑所有行
3. 明确指出你统计的行数范围 (例如: "基于全部231行数据计算...")
4. 如果遇到非数值字段，请先进行类型转换或跳过
"""
            elif query_type == "filtering":
                query_instructions = """
特别提示 - 数据筛选:
1. 筛选时请处理全部数据行，不要只筛选示例数据
2. 明确说明符合筛选条件的行数和总行数
3. 如果结果较多，可以只展示前10行，但说明总共找到多少行
4. 筛选结果为空时，请检查是否有数据格式问题或条件过严
"""
            elif query_type == "sorting":
                query_instructions = """
特别提示 - 数据排序:
1. 排序时请考虑全部数据行，而不仅是示例数据
2. 明确说明排序依据和顺序方向(升序/降序)
3. 结果较多时，可以只展示排序后的前10行，但说明总行数
"""
            
            # 强化提示词，确保AI理解需要分析全部数据
            prompt = f"""你是Excel数据分析专家。请根据以下Excel文件的结构化数据，回答用户的查询。

Excel文件结构详情:
{sheets_summary}

用户查询:
{query}

{query_instructions}

重要说明:
- 以上只是数据的概述和示例，你将接收到完整的数据（全部{len(sheet_names)}个工作表，所有行和列）
- 分析时必须处理每个工作表的所有行数据，不要仅分析示例中展示的几行
- 请基于元数据中提供的行数、列数和数据类型信息来理解数据规模
- 统计数据时，应明确说明使用了多少行数据进行计算

请按照以下指南回答:
1. 开始回答前，明确指出你要分析的是哪个工作表的哪些列，以及总共多少行数据
2. 直接给出数据分析结果，优先使用表格形式展示
3. 简洁说明分析思路和处理步骤
4. 确保结果准确，数值计算要精确
5. 不要输出完整代码，除非用户明确要求

你的回答应该简洁、专业，聚焦于用户实际需要的信息。
"""
            
            # 将结构化数据转换为JSON字符串
            data_json = json.dumps(structured_data, ensure_ascii=False)
            
            # 智能处理大型Excel文件
            full_prompt = ""
            max_data_length = 80000  # 适当增加允许的数据长度
            
            if len(data_json) <= max_data_length:
                # 如果数据不是很大，直接发送完整数据
                full_prompt = f"{prompt}\n\n完整数据:\n{data_json}"
            else:
                # 智能筛选相关工作表
                selected_sheets = []
                query_keywords = query.lower().split()
                
                # 1. 检查查询是否明确提到某个工作表
                for sheet_name in sheet_names:
                    if sheet_name.lower() in query.lower():
                        selected_sheets.append(sheet_name)
                        # 也添加对应的元数据
                        metadata_key = f"{sheet_name}_metadata"
                        if metadata_key in metadata_keys:
                            selected_sheets.append(metadata_key)
                
                # 2. 如果没有直接提到表名，查看列名匹配
                if not selected_sheets:
                    for sheet_name in sheet_names:
                        metadata_key = f"{sheet_name}_metadata"
                        metadata = structured_data.get(metadata_key, {})
                        columns = metadata.get("columns", [])
                        
                        # 检查列名是否与查询相关
                        if any(keyword in ' '.join([str(c).lower() for c in columns]) for keyword in query_keywords):
                            selected_sheets.append(sheet_name)
                            selected_sheets.append(metadata_key)
                
                # 3. 如果仍未找到相关表，选择最大的前1-2个表
                if not selected_sheets:
                    # 按行数排序工作表
                    sheet_sizes = []
                    for sheet_name in sheet_names:
                        metadata_key = f"{sheet_name}_metadata"
                        metadata = structured_data.get(metadata_key, {})
                        rows = metadata.get("total_rows", 0)
                        sheet_sizes.append((sheet_name, rows))
                    
                    # 按行数降序排序
                    sheet_sizes.sort(key=lambda x: x[1], reverse=True)
                    
                    # 选择1-2个最大的表
                    top_sheets = min(2, len(sheet_sizes))
                    for i in range(top_sheets):
                        selected_sheets.append(sheet_sizes[i][0])
                        metadata_key = f"{sheet_sizes[i][0]}_metadata"
                        if metadata_key in metadata_keys:
                            selected_sheets.append(metadata_key)
                
                # 创建筛选后的数据集
                filtered_data = {}
                for key in selected_sheets:
                    if key in structured_data:
                        filtered_data[key] = structured_data[key]
                
                # 如果查询类型是聚合计算，为所有工作表保留元数据
                if query_type == "aggregation":
                    for key in metadata_keys:
                        if key not in filtered_data:
                            filtered_data[key] = structured_data[key]
                
                # 添加表概述
                sheets_overview = f"Excel文件包含以下工作表: {', '.join(sheet_names)}"
                data_json_subset = json.dumps(filtered_data, ensure_ascii=False)
                
                # 检查筛选后的数据是否仍然过大
                if len(data_json_subset) > max_data_length:
                    # 如果还是太大，进一步处理
                    primary_sheet = selected_sheets[0] if selected_sheets else sheet_names[0]
                    primary_sheet_metadata = f"{primary_sheet}_metadata"
                    
                    # 创建极简数据集，只保留主要工作表及其元数据
                    essential_data = {}
                    
                    if primary_sheet in structured_data:
                        sheet_data = structured_data[primary_sheet]
                        metadata = structured_data.get(primary_sheet_metadata, {})
                        rows_count = metadata.get("total_rows", len(sheet_data) if isinstance(sheet_data, list) else 0)
                        
                        # 如果数据是列表且过大，只保留部分行
                        if isinstance(sheet_data, list) and len(sheet_data) > 500:
                            # 智能采样：保留前200行、中间100行和后200行
                            sampled_data = sheet_data[:200]
                            
                            # 如果有超过400行，添加中间100行
                            if len(sheet_data) > 400:
                                mid_point = len(sheet_data) // 2
                                sampled_data.extend(sheet_data[mid_point-50:mid_point+50])
                            
                            # 添加后200行
                            sampled_data.extend(sheet_data[-200:])
                            
                            essential_data[primary_sheet] = sampled_data
                            sheets_overview += f"\n注意: 工作表 {primary_sheet} 包含 {rows_count} 行数据，但由于数据量大，仅提供部分行用于示例。"
                            sheets_overview += f"\n采样方式: 提供前200行 + 中间100行 + 后200行，确保你能看到数据的头部、中部和尾部。"
                        else:
                            essential_data[primary_sheet] = sheet_data
                    
                    # 始终保留元数据
                    if primary_sheet_metadata in structured_data:
                        essential_data[primary_sheet_metadata] = structured_data[primary_sheet_metadata]
                    
                    # 为其他表只保留元数据
                    for sheet in sheet_names:
                        if sheet != primary_sheet:
                            meta_key = f"{sheet}_metadata"
                            if meta_key in structured_data:
                                essential_data[meta_key] = structured_data[meta_key]
                    
                    data_json_subset = json.dumps(essential_data, ensure_ascii=False)
                
                # 添加适当的上下文说明
                full_prompt = f"{prompt}\n\n{sheets_overview}\n\n部分数据:\n{data_json_subset}"
            
            # 使用模型生成回答
            response = self.model.generate_content(
                full_prompt,
                generation_config=self.generation_config
            )
            
            # 解析回复
            if response and response.text:
                reply_content = self._parse_response_content(response.text)
                
                if is_sampled:
                    # 添加采样警告到响应中
                    response_text = response.text + sampling_warning
                    reply_content = self._parse_response_content(response_text)
                
                return {
                    "answer": response.text,
                    "sources": [],  # 结构化数据分析没有具体的来源引用
                    "confidence": float(getattr(response, 'confidence', 0.8)),
                    "reply": reply_content,
                    "is_structured_analysis": True,
                    "is_sampled_data": is_sampled
                }
            else:
                return {
                    "answer": "无法生成有效分析结果",
                    "sources": [],
                    "confidence": 0.0,
                    "reply": [{"type": "markdown", "content": "无法生成有效分析结果"}]
                }
        
        except Exception as e:
            print(f"Structured data analysis error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {
                "answer": f"分析结构化数据时出错: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "reply": [{"type": "markdown", "content": f"分析结构化数据时出错: {str(e)}"}]
            }

    async def stream_response(self, question, context, history=None):
        try:
            # 构建提示
            prompt = self._build_prompt(question, context, history)
            
            # 使用新的API名称替换旧方法调用
            response = await self.model.generate_content_async(
                prompt,
                generation_config=self.generation_config
            )
            
            # 处理返回结果
            # 如果新API不支持流式处理，可能需要模拟流式输出
            if hasattr(response, 'text'):
                # 整个回答一次返回的情况
                full_text = response.text
                # 每5个字符模拟一次流式输出
                for i in range(0, len(full_text), 5):
                    chunk = full_text[i:i+5]
                    yield {
                        "partial_response": chunk,
                        "done": i+5 >= len(full_text)
                    }
                    await asyncio.sleep(0.01)  # 模拟延迟
            else:
                # 尝试按新API处理流式输出
                async for chunk in response:
                    if hasattr(chunk, 'text'):
                        yield {
                            "partial_response": chunk.text,
                            "done": False
                        }
                
                yield {"partial_response": "", "done": True}
            
            
        except Exception as e:
            print(f"Stream generation error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            yield {"error": str(e), "done": True}

    def _build_prompt(self, question, context, history=None):
        """构建提示词，用于统一不同方法的提示词格式"""
        
        # 构建基本提示
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
            prompt += f"历史对话：\n{history}\n\n"
        
        # 添加当前问题
        prompt += f"问题：{question}\n\n"
        prompt += "请根据以上内容提供详细准确的回答。回答要保持严谨且基于文档内容，不要编造信息。"
        
        return prompt