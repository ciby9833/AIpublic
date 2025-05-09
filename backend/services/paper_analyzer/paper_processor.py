import PyPDF2
from io import BytesIO
import docx
import pptx
import pandas as pd
import markdown
import chardet
from typing import Union
import google.generativeai as genai
from google.genai import types
import io
import os
from dotenv import load_dotenv
from PIL import Image
import base64
from pdf2image import convert_from_bytes
import tempfile

class PaperProcessor:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    async def process(self, file_content: bytes, filename: str) -> dict:
        """处理不同类型的文档文件，返回包含行号信息的内容"""
        try:
            extension = filename.lower().split('.')[-1]
            
            # 根据文件类型选择处理方法
            if extension == 'pdf':
                content, line_mapping = await self._process_pdf(file_content)
            elif extension in ['docx', 'doc']:
                content, line_mapping = await self._process_word(file_content)
            elif extension in ['pptx', 'ppt']:
                content, line_mapping = await self._process_powerpoint(file_content)
            elif extension in ['xlsx', 'xls']:
                content, line_mapping = await self._process_excel(file_content)
            elif extension == 'txt':
                content, line_mapping = await self._process_text(file_content)
            elif extension == 'md':
                content, line_mapping = await self._process_markdown(file_content)
            else:
                raise ValueError(f"Unsupported file type: {extension}")
            
            return {
                "content": content,
                "line_mapping": line_mapping,
                "total_lines": len(line_mapping)
            }
                
        except Exception as e:
            raise Exception(f"Paper processing error: {str(e)}")

    async def _process_pdf(self, file_content: bytes) -> tuple:
        """处理 PDF 文件，支持扫描件"""
        try:
            # 1. 首先尝试使用 PyPDF2 提取文本
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # 检查是否为扫描件
            is_scanned = False
            for page in pdf_reader.pages:
                if not page.extract_text().strip():
                    is_scanned = True
                    break
            
            if is_scanned:
                # 2. 如果是扫描件，使用 Gemini API 处理
                return await self._process_scanned_pdf(file_content)
            else:
                # 3. 如果是普通 PDF，使用原有逻辑
                return await self._process_normal_pdf(pdf_reader)
                
        except Exception as e:
            raise Exception(f"PDF processing error: {str(e)}")

    async def _process_scanned_pdf(self, file_content: bytes) -> tuple:
        """使用 Gemini API 处理扫描件 PDF"""
        try:
            # 1. 将 PDF 转换为图片
            images = convert_from_bytes(file_content)
            
            # 2. 处理每一页
            all_content = []
            line_mapping = {}
            current_line = 1
            
            for page_num, image in enumerate(images, 1):
                # 将图片转换为字节数据
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # 3. 调用 Gemini API 处理图片
                doc_data = {
                    "mime_type": "image/jpeg",
                    "data": img_byte_arr
                }
                
                prompt = f"""请提取这个图片中的文本内容（第{page_num}页），包括：
                1. 正文内容
                2. 表格内容
                3. 图片中的文字
                4. 页眉页脚
                请保持原文的段落结构和格式。"""
                
                response = self.model.generate_content(
                    contents=[doc_data, prompt]
                )
                
                if response and response.text:
                    page_content = response.text
                    all_content.append(page_content)
                    
                    # 处理行号映射
                    lines = page_content.split('\n')
                    for line in lines:
                        if line.strip():
                            line_mapping[current_line] = {
                                "content": line,
                                "page": page_num,
                                "start_pos": len('\n'.join(all_content)) - len(line) - 1,
                                "end_pos": len('\n'.join(all_content)) - 1,
                                "is_scanned": True
                            }
                            current_line += 1
            
            content = '\n'.join(all_content)
            return content, line_mapping
            
        except Exception as e:
            raise Exception(f"Scanned PDF processing error: {str(e)}")

    async def _process_normal_pdf(self, pdf_reader: PyPDF2.PdfReader) -> tuple:
        """处理普通 PDF 文件（原有逻辑）"""
        content = ""
        line_mapping = {}
        current_line = 1
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            lines = page_text.split('\n')
            
            for line in lines:
                if line.strip():
                    content += line + "\n"
                    line_mapping[current_line] = {
                        "content": line,
                        "page": page_num,
                        "start_pos": len(content) - len(line) - 1,
                        "end_pos": len(content) - 1,
                        "is_scanned": False  # 标记为非扫描件
                    }
                    current_line += 1
                    
        return content, line_mapping

    async def _process_word(self, file_content: bytes) -> tuple:
        """处理 Word 文件，返回内容和行号映射"""
        try:
            doc_file = BytesIO(file_content)
            doc = docx.Document(doc_file)
            
            content = ""
            line_mapping = {}
            current_line = 1
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content += paragraph.text + "\n"
                    line_mapping[current_line] = {
                        "content": paragraph.text,
                        "page": 1,
                        "start_pos": len(content) - len(paragraph.text) - 1,
                        "end_pos": len(content) - 1,
                        "is_scanned": False  # 添加这个字段
                    }
                    current_line += 1
            
            # 处理表格
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text for cell in row.cells)
                    if row_text.strip():
                        content += row_text + "\n"
                        line_mapping[current_line] = {
                            "content": row_text,
                            "page": 1,
                            "start_pos": len(content) - len(row_text) - 1,
                            "end_pos": len(content) - 1,
                            "is_scanned": False  # 添加这个字段
                        }
                        current_line += 1
                    
            return content, line_mapping
            
        except Exception as e:
            raise Exception(f"Word processing error: {str(e)}")

    async def _process_powerpoint(self, file_content: bytes) -> tuple:
        """处理 PowerPoint 文件，返回内容和行号映射"""
        ppt_file = BytesIO(file_content)
        ppt = pptx.Presentation(ppt_file)
        
        content = ""
        line_mapping = {}
        current_line = 1
        
        for slide in ppt.slides:
            # 处理幻灯片标题
            if slide.shapes.title:
                content += f"Slide Title: {slide.shapes.title.text}\n"
            
            # 处理幻灯片内容
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    content += shape.text + "\n"
                    lines = shape.text.split('\n')
                    
                    for line in lines:
                        if line.strip():
                            line_mapping[current_line] = {
                                "content": line,
                                "page": 1,
                                "start_pos": len(content) - len(line) - 1,
                                "end_pos": len(content) - 1
                            }
                            current_line += 1
        return content, line_mapping

    async def _process_excel(self, file_content: bytes) -> tuple:
        """处理 Excel 文件，返回内容和行号映射"""
        try:
            excel_file = BytesIO(file_content)
            
            # 读取所有工作表
            content = ""
            line_mapping = {}
            current_line = 1
            excel = pd.ExcelFile(excel_file)
            
            for sheet_name in excel.sheet_names:
                df = pd.read_excel(excel, sheet_name=sheet_name)
                content += f"\nSheet: {sheet_name}\n"
                content += df.to_string(index=False) + "\n"
                lines = df.to_string(index=False).split('\n')
                
                for line in lines:
                    if line.strip():
                        line_mapping[current_line] = {
                            "content": line,
                            "page": 1,
                            "start_pos": len(content) - len(line) - 1,
                            "end_pos": len(content) - 1,
                            "is_scanned": False  # Excel 文件不是扫描件
                        }
                        current_line += 1
                        
            return content, line_mapping
            
        except Exception as e:
            raise Exception(f"Excel processing error: {str(e)}")

    async def _process_text(self, file_content: bytes) -> tuple:
        """处理文本文件，返回内容和行号映射"""
        content = file_content.decode('utf-8')
        lines = content.split('\n')
        
        line_mapping = {}
        for i, line in enumerate(lines, 1):
            if line.strip():
                line_mapping[i] = {
                    "content": line,
                    "start_pos": content.find(line),
                    "end_pos": content.find(line) + len(line)
                }
        
        return content, line_mapping

    async def _process_markdown(self, file_content: bytes) -> tuple:
        """处理 Markdown 文件，返回内容和行号映射"""
        # 首先获取文本内容
        content, line_mapping = await self._process_text(file_content)
        
        # 将 Markdown 转换为纯文本
        html = markdown.markdown(content)
        # 这里可以添加 HTML 到纯文本的转换逻辑
        # 简单实现：移除 HTML 标签
        import re
        content = re.sub(r'<[^>]+>', '', html)
        lines = content.split('\n')
        current_line = len(line_mapping) + 1
        
        for line in lines:
            if line.strip():
                line_mapping[current_line] = {
                    "content": line,
                    "page": 1,
                    "start_pos": len(content) - len(line) - 1,
                    "end_pos": len(content) - 1
                }
                current_line += 1
        return content, line_mapping
