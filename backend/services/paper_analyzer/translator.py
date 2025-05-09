import aiohttp
from .config import DEEPL_API_KEY, DEEPL_API_URL, SUPPORTED_LANGUAGES

class Translator:
    def __init__(self):
        self.api_key = DEEPL_API_KEY
        self.api_url = DEEPL_API_URL
        self.supported_languages = SUPPORTED_LANGUAGES
        self.max_chars = 5000  # DeepL API 单次请求最大字符数

    async def translate_text(self, text: str, target_lang: str) -> str:
        """翻译文本到目标语言，自动处理长文本"""
        if not text:
            return ""
            
        if target_lang not in self.supported_languages:
            raise ValueError(f"Unsupported target language: {target_lang}")

        try:
            # 如果文本长度小于最大限制，直接翻译
            if len(text) <= self.max_chars:
                return await self._translate_chunk(text, target_lang)
                
            # 处理长文本：按段落分割
            paragraphs = text.split('\n')
            chunks = []
            current_chunk = ""
            
            # 按段落分组并保持在字符限制内
            for paragraph in paragraphs:
                # 如果当前段落本身超过限制，需要拆分
                if len(paragraph) > self.max_chars:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    
                    # 将长段落分成小块
                    for i in range(0, len(paragraph), self.max_chars - 100):
                        sub_para = paragraph[i:i + self.max_chars - 100]
                        chunks.append(sub_para)
                else:
                    # 检查添加此段落是否会超出限制
                    if len(current_chunk) + len(paragraph) + 1 > self.max_chars:
                        chunks.append(current_chunk)
                        current_chunk = paragraph
                    else:
                        if current_chunk:
                            current_chunk += '\n' + paragraph
                        else:
                            current_chunk = paragraph
            
            # 添加最后一个块
            if current_chunk:
                chunks.append(current_chunk)
                
            # 翻译所有块并组合结果
            translated_chunks = []
            for chunk in chunks:
                translated = await self._translate_chunk(chunk, target_lang)
                translated_chunks.append(translated)
                
            return '\n'.join(translated_chunks)
                
        except Exception as e:
            raise Exception(f"Translation error: {str(e)}")
            
    async def _translate_chunk(self, text: str, target_lang: str) -> str:
        """翻译单个文本块"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                data={
                    "auth_key": self.api_key,
                    "text": text,
                    "target_lang": target_lang
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Translation failed: {error_text}")
                    
                result = await response.json()
                return result["translations"][0]["text"]

    def get_supported_languages(self) -> dict:
        """获取支持的语言列表"""
        return self.supported_languages
