import { API_BASE_URL } from '../config/env';
import { apiRequest } from './auth';

export interface TranslationResponse {
  success: boolean;
  error?: string;
}

// 修改API调用方式
const createApiUrl = (path: string) => {
  // 如果是生产环境，直接使用相对路径
  // 如果是开发环境，使用完整的API_BASE_URL
  return import.meta.env.PROD ? path : `${API_BASE_URL}${path}`;
};

// 翻译文档
export const translateDocument = async (
  file: File, 
  sourceLang: string, 
  targetLang: string, 
  useGlossary: boolean = false
): Promise<Response> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('source_lang', sourceLang);
  formData.append('target_lang', targetLang);
  formData.append('use_glossary', useGlossary.toString());

  return apiRequest(`${API_BASE_URL}/api/translate/upload`, {
    method: 'POST',
    body: formData
  });
};

// 检查翻译状态
export const checkTranslationStatus = async (
  documentId: string, 
  documentKey: string
): Promise<Response> => {
  return apiRequest(`${API_BASE_URL}/api/translate/${documentId}/status`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: new URLSearchParams({ document_key: documentKey })
  });
};

// 下载翻译结果
export const downloadTranslationResult = async (
  documentId: string, 
  documentKey: string
): Promise<Response> => {
  return apiRequest(`${API_BASE_URL}/api/translate/${documentId}/result`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: new URLSearchParams({ document_key: documentKey })
  });
};

// 文本翻译相关
export const translateText = async (text: string, targetLang: string) => {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('target_lang', targetLang);

  const response = await fetch(createApiUrl('/api/translate/text'), {
    method: 'POST',
    body: formData,
  });
  return response;
};

// 添加多语言翻译接口定义 ai接口
interface MultilingualTranslationResponse {
  status: string;
  translations: {
    detected_language: string;
    english: string;
    chinese: string;
    indonesian: string;
  };
}

// 添加多语言翻译方法
export const translateMultilingual = async (text: string): Promise<MultilingualTranslationResponse> => {
  const formData = new FormData();
  formData.append('text', text);

  const response = await fetch(createApiUrl('/api/translate/multilingual'), {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail?.message || 'Translation failed');
  }

  return response.json();
};

// 术语表相关
interface GlossarySearchResponse {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  entries: Array<{
    id: number;
    glossary_id: number;
    glossary_name: string;
    source_lang: string;
    target_lang: string;
    source_term: string;
    target_term: string;
    created_at: string;
    glossary_created_at: string;
    glossary_updated_at: string | null;
  }>;
}

interface SearchGlossariesParams {
  page?: number;
  pageSize?: number;
  name?: string;
  startDate?: string;
  endDate?: string;
  sourceLang?: string;
  targetLang?: string;
}

// 用户管理相关API
export const userApi = {
  // 获取用户列表
  getUsers: async (): Promise<Response> => {
    return apiRequest(`${API_BASE_URL}/api/users`);
  },

  // 获取用户详情
  getUserById: async (userId: string): Promise<Response> => {
    return apiRequest(`${API_BASE_URL}/api/users/${userId}`);
  },

  // 更新用户信息
  updateUser: async (userId: string, userData: any): Promise<Response> => {
    return apiRequest(`${API_BASE_URL}/api/users/${userId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(userData)
    });
  }
};

// 词汇表相关API
export const glossaryApi = {
  // 搜索词汇表
  searchGlossary: async (query: string): Promise<Response> => {
    return apiRequest(`${API_BASE_URL}/api/glossary/search?q=${encodeURIComponent(query)}`);
  },

  // 高级搜索词汇表
  searchGlossaries: async (params: SearchGlossariesParams): Promise<GlossarySearchResponse> => {
    const searchParams = new URLSearchParams();
    
    if (params.page) searchParams.append('page', params.page.toString());
    if (params.pageSize) searchParams.append('page_size', params.pageSize.toString());
    if (params.name) searchParams.append('name', params.name);
    if (params.startDate) searchParams.append('start_date', params.startDate);
    if (params.endDate) searchParams.append('end_date', params.endDate);
    if (params.sourceLang) searchParams.append('source_lang', params.sourceLang);
    if (params.targetLang) searchParams.append('target_lang', params.targetLang);

    const response = await apiRequest(`${API_BASE_URL}/api/glossary/search?${searchParams.toString()}`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  },

  // 获取词汇表列表
  getGlossaryList: async (): Promise<Response> => {
    return apiRequest(`${API_BASE_URL}/api/glossary`);
  },

  // 获取词汇表列表（别名）
  getGlossaries: async (): Promise<Response> => {
    return apiRequest(`${API_BASE_URL}/api/glossary`);
  },

  // 获取词汇表详情
  getGlossaryDetails: async (glossaryId: string, page: number = 1, pageSize: number = 10): Promise<Response> => {
    return apiRequest(`${API_BASE_URL}/api/glossary/${glossaryId}?page=${page}&page_size=${pageSize}`);
  },

  // 创建词汇表
  createGlossary: async (glossaryData: any): Promise<Response> => {
    return apiRequest(`${API_BASE_URL}/api/glossary`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(glossaryData)
    });
  },

  // 删除词汇表
  deleteGlossary: async (glossaryId: string): Promise<Response> => {
    return apiRequest(`${API_BASE_URL}/api/glossary/${glossaryId}`, {
      method: 'DELETE'
    });
  },

  // 添加词汇表条目
  addGlossaryEntry: async (entry: any): Promise<Response> => {
    return apiRequest(`${API_BASE_URL}/api/glossary/entries`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(entry)
    });
  },

  // 更新词汇表条目
  updateGlossaryEntry: async (entryId: string, targetTerm: string): Promise<Response> => {
    return apiRequest(`${API_BASE_URL}/api/glossary/entries/${entryId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ target_term: targetTerm })
    });
  },

  // 删除词汇表条目
  deleteGlossaryEntry: async (entryId: string): Promise<Response> => {
    return apiRequest(`${API_BASE_URL}/api/glossary/entries/${entryId}`, {
      method: 'DELETE'
    });
  }
};

// 通用API请求函数（向后兼容）
export const makeAuthenticatedRequest = apiRequest;
