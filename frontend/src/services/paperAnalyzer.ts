// frontend/src/services/paperAnalyzer.ts  文档阅读
import { API_BASE_URL } from '../config/env';

export interface PaperAnalysisResult {
  status: string;
  message: string;
  paper_id?: string;  // 添加paper_id字段
  content?: string;   // 添加content字段
}

export interface QuestionResponse {
  status: string;
  response: string;
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

      const response = await fetch(
        `${API_BASE_URL}/api/paper/analyze`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
          },
          body: formData,
          credentials: 'include'
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to analyze paper:', error);
      throw error;
    }
  },

  // 获取文档内容
  getDocumentContent: async (paperId: string): Promise<string> => {
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

      const data = await response.json();
      return data.content;
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