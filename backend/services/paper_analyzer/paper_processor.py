import PyPDF2
from io import BytesIO
import docx
import pptx
import pandas as pd
import markdown
import chardet
from typing import Union

class PaperProcessor:
    async def process(self, file_content: bytes, filename: str) -> str:
        """处理不同类型的文档文件"""
        try:
            # 获取文件扩展名
            extension = filename.lower().split('.')[-1]
            
            # 根据文件类型选择处理方法
            if extension == 'pdf':
                return await self._process_pdf(file_content)
            elif extension in ['docx', 'doc']:
                return await self._process_word(file_content)
            elif extension in ['pptx', 'ppt']:
                return await self._process_powerpoint(file_content)
            elif extension in ['xlsx', 'xls']:
                return await self._process_excel(file_content)
            elif extension == 'txt':
                return await self._process_text(file_content)
            elif extension == 'md':
                return await self._process_markdown(file_content)
            else:
                raise ValueError(f"Unsupported file type: {extension}")
                
        except Exception as e:
            raise Exception(f"Paper processing error: {str(e)}")

    async def _process_pdf(self, file_content: bytes) -> str:
        """处理 PDF 文件"""
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    async def _process_word(self, file_content: bytes) -> str:
        """处理 Word 文件"""
        doc_file = BytesIO(file_content)
        doc = docx.Document(doc_file)
        
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
            
        # 处理表格
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text for cell in row.cells)
                text += row_text + "\n"
        return text

    async def _process_powerpoint(self, file_content: bytes) -> str:
        """处理 PowerPoint 文件"""
        ppt_file = BytesIO(file_content)
        ppt = pptx.Presentation(ppt_file)
        
        text = ""
        for slide in ppt.slides:
            # 处理幻灯片标题
            if slide.shapes.title:
                text += f"Slide Title: {slide.shapes.title.text}\n"
            
            # 处理幻灯片内容
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text += shape.text + "\n"
        return text

    async def _process_excel(self, file_content: bytes) -> str:
        """处理 Excel 文件"""
        excel_file = BytesIO(file_content)
        
        # 读取所有工作表
        text = ""
        excel = pd.ExcelFile(excel_file)
        for sheet_name in excel.sheet_names:
            df = pd.read_excel(excel, sheet_name=sheet_name)
            text += f"\nSheet: {sheet_name}\n"
            text += df.to_string(index=False) + "\n"
        return text

    async def _process_text(self, file_content: bytes) -> str:
        """处理文本文件"""
        # 检测文件编码
        encoding = chardet.detect(file_content)['encoding']
        if not encoding:
            encoding = 'utf-8'
            
        try:
            return file_content.decode(encoding)
        except UnicodeDecodeError:
            # 如果检测到的编码失败，尝试使用 utf-8
            return file_content.decode('utf-8', errors='ignore')

    async def _process_markdown(self, file_content: bytes) -> str:
        """处理 Markdown 文件"""
        # 首先获取文本内容
        text = await self._process_text(file_content)
        
        # 将 Markdown 转换为纯文本
        html = markdown.markdown(text)
        # 这里可以添加 HTML 到纯文本的转换逻辑
        # 简单实现：移除 HTML 标签
        import re
        text = re.sub(r'<[^>]+>', '', html)
        return text
