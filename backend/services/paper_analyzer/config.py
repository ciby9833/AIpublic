import os
from dotenv import load_dotenv

load_dotenv()

# 直接导出配置变量，而不是使用类
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
DEEPL_API_URL = "https://api.deepl.com/v2/translate"  # 或使用 pro 版本 URL

# 支持的语言列表
SUPPORTED_LANGUAGES = {
    "ID": "Indonesian",
    "EN-US": "English (American)",
    "ZH": "Chinese",
    "BG": "Bulgarian",
    "CS": "Czech",
    "DA": "Danish",
    "DE": "German",
    "EL": "Greek",
    "EN-GB": "English (British)",
    "ES": "Spanish",
    "ET": "Estonian",
    "FI": "Finnish",
    "FR": "French",
    "HU": "Hungarian",
    "IT": "Italian",
    "JA": "Japanese",
    "KO": "Korean",
    "LT": "Lithuanian",
    "LV": "Latvian",
    "NB": "Norwegian",
    "NL": "Dutch",
    "PL": "Polish",
    "PT-BR": "Portuguese (Brazilian)",
    "PT-PT": "Portuguese (European)",
    "RO": "Romanian",
    "RU": "Russian",
    "SK": "Slovak",
    "SL": "Slovenian",
    "SV": "Swedish",
    "TR": "Turkish",
    "UK": "Ukrainian" 
}

# 文件配置
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.xlsx', '.txt', '.md'}

# 向量检索配置
TOP_K_RESULTS = 3
SIMILARITY_THRESHOLD = 0.7
