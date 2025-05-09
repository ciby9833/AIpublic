// frontend/src/services/paperAnalyzer.ts  文档阅读
import { API_BASE_URL } from '../config/env';
import { ChatResponse } from '../types/chat';

export interface LineMapping {
  [key: string]: {
    content: string;
    page?: number;
    start_pos?: number;
    end_pos?: number;
  }
}

export interface PaperAnalysisResult {
  status: string;
  message: string;
  paper_id?: string;  // 添加paper_id字段
  content?: string;   // 添加content字段
  line_mapping?: LineMapping;  // 添加行号映射
  total_lines?: number;         // 添加总行数
}

export interface QuestionResponse {
  status: string;
  response: string | ChatResponse;
}

export interface QuestionHistory {
  question: string;
  answer: string;
  created_at: string;
}

export interface Language {
  code: string;
  name: string;
}

export const paperAnalyzerApi = {
  // 分析论文
  analyzePaper: async (file: File): Promise<PaperAnalysisResult> => {
    try {
      // 验证文件类型
      const allowedTypes = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'application/vnd.ms-powerpoint',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
        'text/plain',
        'text/markdown'
      ];

      if (!allowedTypes.includes(file.type)) {
        throw new Error('Unsupported file type');
      }

      const formData = new FormData();
      formData.append('file', file);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000); // 5分钟超时

      const response = await fetch(
        `${API_BASE_URL}/api/paper/analyze`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
          },
          body: formData,
          credentials: 'include',
          signal: controller.signal
        }
      );

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      // 添加数据验证
      if (!result.content) {
        console.warn('No content in response');
      }
      
      if (!result.line_mapping || Object.keys(result.line_mapping).length === 0) {
        console.warn('No line mapping in response');
      }

      return result;
    } catch (error: unknown) {  // 明确指定 error 类型为 unknown
      if (error instanceof Error) {  // 类型守卫
        if (error.name === 'AbortError') {
          throw new Error('分析超时，请稍后重试');
        }
        console.error('Failed to analyze paper:', error);
        throw error;
      }
      // 处理未知错误
      console.error('Unknown error occurred:', error);
      throw new Error('文档分析过程中发生未知错误');
    }
  },

  // 获取文档内容
  getDocumentContent: async (paperId: string): Promise<{
    content: string;
    line_mapping: LineMapping;
    total_lines: number;
  }> => {
    if (!paperId) {
      throw new Error('Paper ID is required');
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/paper/${paperId}/content`,
        {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
          },
          credentials: 'include'
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to get document content:', error);
      throw error;
    }
  },

  // 提问
  askQuestion: async (question: string, paperId: string): Promise<QuestionResponse> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/paper/ask`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ question, paper_id: paperId }),
          credentials: 'include'
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to ask question:', error);
      throw error;
    }
  },

  // 获取问答历史
  getQuestionHistory: async (paperId: string): Promise<QuestionHistory[]> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/paper/${paperId}/questions`,
        {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
          },
          credentials: 'include'
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.history;
    } catch (error) {
      console.error('Failed to get question history:', error);
      throw error;
    }
  },

  // 获取支持的语言列表
  getSupportedLanguages: async (): Promise<Record<string, string>> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/paper/supported-languages`,
        {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
          },
          credentials: 'include'
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.languages;
    } catch (error) {
      console.error('Failed to get supported languages:', error);
      throw error;
    }
  },

  // 翻译论文
  translatePaper: async (paperId: string, targetLang: string): Promise<string> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/paper/${paperId}/translate`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
            'Content-Type': 'application/json',
          },
          credentials: 'include',
          body: JSON.stringify(targetLang)  // 直接发送语言代码字符串
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.content;
    } catch (error) {
      console.error('Failed to translate paper:', error);
      throw error;
    }
  },

  async downloadTranslation(paperId: string, targetLang: string, format: string) {
    const response = await fetch(`${API_BASE_URL}/api/paper/${paperId}/download`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify({ target_lang: targetLang, format }),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail?.message || 'Download failed');
    }
    
    return response.blob();
  },
};