# backend/services/paper_analyzer/web_processor.py
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, parse_qs
import json
from bs4 import BeautifulSoup
import re
import hashlib
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

@dataclass
class WebProcessingConfig:
    """网页处理配置"""
    # 基础配置
    max_page_size_mb: int = 10
    timeout_seconds: int = 30
    max_concurrent_requests: int = 5
    
    # 内容提取配置
    extract_text: bool = True
    extract_links: bool = True
    extract_images: bool = False
    extract_tables: bool = True
    extract_code: bool = True
    extract_metadata: bool = True
    
    # 过滤配置
    min_text_length: int = 100
    max_text_length: int = 1000000
    ignore_navigation: bool = True
    ignore_footer: bool = True
    ignore_ads: bool = True
    
    # 分块配置
    chunk_size: int = 512
    chunk_overlap: int = 50
    smart_chunking: bool = True
    preserve_structure: bool = True

class WebProcessor:
    """网页内容处理器 - 整合PageIndex能力"""
    
    def __init__(self, config: WebProcessingConfig = None):
        self.config = config or WebProcessingConfig()
        self.session = None
        
        logger.info(f"[WEB_PROCESSOR_INIT] 网页处理器初始化完成")
        logger.debug(f"[WEB_PROCESSOR_CONFIG] 配置: 超时={self.config.timeout_seconds}s, 最大并发={self.config.max_concurrent_requests}")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'CargoPPT WebProcessor/1.0 (Document Analysis Bot)'
            }
        )
        
        logger.debug(f"[WEB_PROCESSOR_SESSION] HTTP会话创建成功")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
            logger.debug(f"[WEB_PROCESSOR_SESSION_CLOSED] HTTP会话已关闭")

    async def process_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """处理单个URL - 主要接口方法"""
        try:
            logger.info(f"[WEB_PROCESS_START] 开始处理URL - {url}")
            
            # 验证URL
            if not self._is_valid_url(url):
                raise ValueError(f"无效的URL: {url}")
            
            # 获取网页内容
            html_content, response_info = await self._fetch_url(url)
            
            # 解析网页内容
            parsed_content = await self._parse_html_content(html_content, url)
            
            # 提取结构化数据
            structured_data = await self._extract_structured_data(parsed_content, url)
            
            # 生成分块内容
            chunks = await self._generate_chunks(parsed_content['text'], url)
            
            # 构建结果
            result = {
                "url": url,
                "title": parsed_content.get('title', ''),
                "content": parsed_content['text'],
                "structured_data": structured_data,
                "chunks": chunks,
                "metadata": {
                    "processed_at": datetime.utcnow().isoformat(),
                    "content_length": len(parsed_content['text']),
                    "chunk_count": len(chunks),
                    "response_info": response_info,
                    "extraction_stats": parsed_content.get('stats', {})
                },
                "source_type": "web"
            }
            
            logger.info(f"[WEB_PROCESS_SUCCESS] URL处理成功 - 内容长度: {len(parsed_content['text'])}, 分块数: {len(chunks)}")
            return result
            
        except Exception as e:
            logger.error(f"[WEB_PROCESS_ERROR] URL处理失败: {url} - 错误: {str(e)}")
            raise Exception(f"处理URL失败: {str(e)}")

    async def process_multiple_urls(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量处理多个URL"""
        logger.info(f"[WEB_BATCH_START] 开始批量处理URL - 数量: {len(urls)}")
        
        results = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def process_single_url(url: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.process_url(url)
                except Exception as e:
                    logger.error(f"[WEB_BATCH_ERROR] 处理URL失败: {url} - {str(e)}")
                    return {
                        "url": url,
                        "error": str(e),
                        "content": "",
                        "chunks": [],
                        "metadata": {"error": True}
                    }
        
        # 并发处理所有URL
        tasks = [process_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤出成功的结果
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[WEB_BATCH_EXCEPTION] URL处理异常: {urls[i]} - {str(result)}")
            elif not result.get("error"):
                successful_results.append(result)
        
        logger.info(f"[WEB_BATCH_COMPLETE] 批量处理完成 - 成功: {len(successful_results)}/{len(urls)}")
        return successful_results

    async def _fetch_url(self, url: str) -> Tuple[str, Dict]:
        """获取URL内容"""
        try:
            logger.debug(f"[WEB_FETCH] 开始获取URL内容: {url}")
            
            async with self.session.get(url) as response:
                response_info = {
                    "status_code": response.status,
                    "content_type": response.headers.get('content-type', ''),
                    "content_length": response.headers.get('content-length', 0),
                    "final_url": str(response.url)
                }
                
                # 检查响应状态
                if response.status != 200:
                    raise Exception(f"HTTP错误: {response.status}")
                
                # 检查内容类型
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    raise Exception(f"不支持的内容类型: {content_type}")
                
                # 获取内容
                content = await response.text()
                
                # 检查内容大小
                content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
                if content_size_mb > self.config.max_page_size_mb:
                    raise Exception(f"页面过大: {content_size_mb:.2f}MB > {self.config.max_page_size_mb}MB")
                
                logger.debug(f"[WEB_FETCH_SUCCESS] URL内容获取成功 - 大小: {content_size_mb:.2f}MB")
                return content, response_info
                
        except asyncio.TimeoutError:
            logger.error(f"[WEB_FETCH_TIMEOUT] URL获取超时: {url}")
            raise Exception(f"请求超时: {url}")
        except Exception as e:
            logger.error(f"[WEB_FETCH_ERROR] URL获取失败: {url} - {str(e)}")
            raise

    async def _parse_html_content(self, html_content: str, url: str) -> Dict[str, Any]:
        """解析HTML内容"""
        try:
            logger.debug(f"[WEB_PARSE] 开始解析HTML内容 - URL: {url}")
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 移除不需要的元素
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                if self.config.ignore_navigation or element.name in ['nav', 'header', 'footer', 'aside']:
                    element.decompose()
            
            # 提取标题
            title = ""
            if soup.title:
                title = soup.title.string.strip() if soup.title.string else ""
            if not title:
                h1 = soup.find('h1')
                title = h1.get_text().strip() if h1 else ""
            
            # 提取主要文本内容
            text_content = self._extract_text_content(soup)
            
            # 提取链接
            links = []
            if self.config.extract_links:
                links = self._extract_links(soup, url)
            
            # 提取表格
            tables = []
            if self.config.extract_tables:
                tables = self._extract_tables(soup)
            
            # 提取代码块
            code_blocks = []
            if self.config.extract_code:
                code_blocks = self._extract_code_blocks(soup)
            
            # 提取图片信息
            images = []
            if self.config.extract_images:
                images = self._extract_images(soup, url)
            
            # 统计信息
            stats = {
                "text_length": len(text_content),
                "links_count": len(links),
                "tables_count": len(tables),
                "code_blocks_count": len(code_blocks),
                "images_count": len(images)
            }
            
            result = {
                "title": title,
                "text": text_content,
                "links": links,
                "tables": tables,
                "code_blocks": code_blocks,
                "images": images,
                "stats": stats
            }
            
            logger.debug(f"[WEB_PARSE_SUCCESS] HTML解析成功 - 文本长度: {len(text_content)}, 链接: {len(links)}, 表格: {len(tables)}")
            return result
            
        except Exception as e:
            logger.error(f"[WEB_PARSE_ERROR] HTML解析失败: {str(e)}")
            raise

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """提取文本内容"""
        try:
            # 优先提取article内容
            article = soup.find('article')
            if article:
                content_element = article
            else:
                # 查找主要内容区域
                main_selectors = ['main', '[role="main"]', '.content', '.article', '.post', '#content']
                content_element = None
                
                for selector in main_selectors:
                    element = soup.select_one(selector)
                    if element:
                        content_element = element
                        break
                
                if not content_element:
                    content_element = soup
            
            # 提取文本
            text = content_element.get_text(separator='\n', strip=True)
            
            # 清理文本
            text = self._clean_text(text)
            
            # 检查文本长度
            if len(text) < self.config.min_text_length:
                logger.warning(f"[WEB_TEXT_SHORT] 提取的文本过短: {len(text)} < {self.config.min_text_length}")
            
            if len(text) > self.config.max_text_length:
                logger.warning(f"[WEB_TEXT_LONG] 提取的文本过长，将截断: {len(text)} > {self.config.max_text_length}")
                text = text[:self.config.max_text_length]
            
            return text
            
        except Exception as e:
            logger.error(f"[WEB_TEXT_EXTRACT_ERROR] 文本提取失败: {str(e)}")
            return ""

    def _clean_text(self, text: str) -> str:
        """清理文本内容"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除多余的换行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\u2e80-\u2eff.,!?;:()[\]{}"\'`~@#$%^&*+=|\\<>/\-_]', '', text)
        
        return text.strip()

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """提取链接信息"""
        links = []
        try:
            for a_tag in soup.find_all('a', href=True):
                href = a_tag.get('href')
                text = a_tag.get_text().strip()
                
                if href and text:
                    # 转换为绝对URL
                    absolute_url = urljoin(base_url, href)
                    
                    links.append({
                        "url": absolute_url,
                        "text": text,
                        "title": a_tag.get('title', ''),
                        "rel": a_tag.get('rel', [])
                    })
            
            logger.debug(f"[WEB_LINKS] 提取链接数量: {len(links)}")
            return links[:50]  # 限制链接数量
            
        except Exception as e:
            logger.error(f"[WEB_LINKS_ERROR] 链接提取失败: {str(e)}")
            return []

    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """提取表格数据"""
        tables = []
        try:
            for table in soup.find_all('table'):
                table_data = {
                    "headers": [],
                    "rows": [],
                    "caption": ""
                }
                
                # 提取表格标题
                caption = table.find('caption')
                if caption:
                    table_data["caption"] = caption.get_text().strip()
                
                # 提取表头
                thead = table.find('thead')
                if thead:
                    header_row = thead.find('tr')
                    if header_row:
                        table_data["headers"] = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
                
                # 提取表格行
                tbody = table.find('tbody') or table
                for row in tbody.find_all('tr'):
                    cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                    if cells:
                        table_data["rows"].append(cells)
                
                if table_data["rows"]:
                    tables.append(table_data)
            
            logger.debug(f"[WEB_TABLES] 提取表格数量: {len(tables)}")
            return tables
            
        except Exception as e:
            logger.error(f"[WEB_TABLES_ERROR] 表格提取失败: {str(e)}")
            return []

    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """提取代码块"""
        code_blocks = []
        try:
            # 提取pre和code标签
            for element in soup.find_all(['pre', 'code']):
                code_text = element.get_text()
                if code_text.strip():
                    code_blocks.append({
                        "code": code_text,
                        "language": element.get('class', []),
                        "tag": element.name
                    })
            
            logger.debug(f"[WEB_CODE] 提取代码块数量: {len(code_blocks)}")
            return code_blocks
            
        except Exception as e:
            logger.error(f"[WEB_CODE_ERROR] 代码块提取失败: {str(e)}")
            return []

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """提取图片信息"""
        images = []
        try:
            for img in soup.find_all('img'):
                src = img.get('src')
                if src:
                    absolute_url = urljoin(base_url, src)
                    images.append({
                        "url": absolute_url,
                        "alt": img.get('alt', ''),
                        "title": img.get('title', ''),
                        "width": img.get('width'),
                        "height": img.get('height')
                    })
            
            logger.debug(f"[WEB_IMAGES] 提取图片数量: {len(images)}")
            return images[:20]  # 限制图片数量
            
        except Exception as e:
            logger.error(f"[WEB_IMAGES_ERROR] 图片提取失败: {str(e)}")
            return []

    async def _extract_structured_data(self, parsed_content: Dict, url: str) -> Dict[str, Any]:
        """提取结构化数据"""
        try:
            structured_data = {
                "url": url,
                "title": parsed_content.get('title', ''),
                "content_summary": parsed_content['text'][:500] if parsed_content['text'] else '',
                "links": parsed_content.get('links', [])[:10],  # 前10个链接
                "tables": parsed_content.get('tables', []),
                "code_blocks": parsed_content.get('code_blocks', []),
                "statistics": parsed_content.get('stats', {}),
                "extracted_at": datetime.utcnow().isoformat()
            }
            
            logger.debug(f"[WEB_STRUCTURED] 结构化数据提取完成")
            return structured_data
            
        except Exception as e:
            logger.error(f"[WEB_STRUCTURED_ERROR] 结构化数据提取失败: {str(e)}")
            return {}

    async def _generate_chunks(self, text: str, url: str) -> List[Dict[str, Any]]:
        """生成文本分块"""
        try:
            if not text or len(text) < self.config.min_text_length:
                logger.warning(f"[WEB_CHUNKS_SKIP] 文本过短，跳过分块: {len(text) if text else 0}")
                return []
            
            chunks = []
            chunk_size = self.config.chunk_size
            overlap = self.config.chunk_overlap
            
            # 智能分块：按段落分割
            if self.config.smart_chunking:
                paragraphs = text.split('\n\n')
                current_chunk = ""
                chunk_start = 0
                
                for paragraph in paragraphs:
                    if len(current_chunk) + len(paragraph) <= chunk_size:
                        current_chunk += paragraph + '\n\n'
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                "content": current_chunk.strip(),
                                "start_pos": chunk_start,
                                "end_pos": chunk_start + len(current_chunk),
                                "chunk_index": len(chunks),
                                "source_url": url
                            })
                            
                            # 计算重叠
                            if self.config.chunk_overlap > 0:
                                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                                current_chunk = overlap_text + paragraph + '\n\n'
                                chunk_start = chunk_start + len(current_chunk) - len(overlap_text)
                            else:
                                current_chunk = paragraph + '\n\n'
                                chunk_start = chunk_start + len(current_chunk)
                        else:
                            current_chunk = paragraph + '\n\n'
                
                # 添加最后一个块
                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "start_pos": chunk_start,
                        "end_pos": chunk_start + len(current_chunk),
                        "chunk_index": len(chunks),
                        "source_url": url
                    })
            else:
                # 简单分块
                for i in range(0, len(text), chunk_size - overlap):
                    chunk_text = text[i:i + chunk_size]
                    if chunk_text.strip():
                        chunks.append({
                            "content": chunk_text.strip(),
                            "start_pos": i,
                            "end_pos": i + len(chunk_text),
                            "chunk_index": len(chunks),
                            "source_url": url
                        })
            
            logger.debug(f"[WEB_CHUNKS_SUCCESS] 文本分块完成 - 原文长度: {len(text)}, 分块数: {len(chunks)}")
            return chunks
            
        except Exception as e:
            logger.error(f"[WEB_CHUNKS_ERROR] 文本分块失败: {str(e)}")
            return []

    def _is_valid_url(self, url: str) -> bool:
        """验证URL有效性"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _calculate_content_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.md5(content.encode('utf-8')).hexdigest() 