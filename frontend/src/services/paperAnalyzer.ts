// frontend/src/services/paperAnalyzer.ts  文档阅读
import { API_BASE_URL } from '../config/env';
import { 
  ChatResponse, 
  ChatMessage, 
  ChatSession, 
  CreateSessionRequest, 
  DocumentRequest,
  SessionDocument
} from '../types/chat';

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
  analyzePaper: async (file: File, sessionId?: string): Promise<PaperAnalysisResult> => {
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
      
      // 如果提供了会话ID，添加到请求中
      if (sessionId) {
        formData.append('session_id', sessionId);
      }

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
        // 检查是否是会话文档数量限制错误
        if (errorData.detail?.code === "SESSION_DOCUMENT_LIMIT_REACHED") {
          throw new Error(errorData.detail.message || "一个会话最多支持10个文档，请创建新会话");
        }
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
    } catch (error: unknown) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('分析超时，请稍后重试');
        }
        console.error('Failed to analyze paper:', error);
        throw error;
      }
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

  // Get all chat sessions for a paper
  getChatSessions: async (paperId: string): Promise<ChatSession[]> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/paper/${paperId}/chat-sessions`,
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
      console.error('Failed to get chat sessions:', error);
      throw error;
    }
  },

  // Create a new chat session
  createChatSession: async (params: CreateSessionRequest): Promise<ChatSession> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chat-sessions`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(params),
          credentials: 'include'
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to create chat session:', error);
      throw error;
    }
  },

  // 兼容旧接口（单文档会话）
  createSingleDocumentSession: async (paperId: string, title?: string): Promise<ChatSession> => {
    return await paperAnalyzerApi.createChatSession({
      paper_ids: [paperId],
      title,
      is_ai_only: false
    });
  },

  // 创建纯AI对话会话
  createAiOnlySession: async (title: string): Promise<ChatSession> => {
    return paperAnalyzerApi.createChatSession({
      title: title,
      is_ai_only: true, // 显式设置为AI-only会话
      paper_ids: [] // 空数组表示无相关文档
    });
  },

  // Get chat history for a session
  getChatHistory: async (sessionId: string, limit: number = 20, beforeId?: string): Promise<{messages: any[], has_more: boolean}> => {
    try {
      let url = `${API_BASE_URL}/api/chat-sessions/${sessionId}/messages?limit=${limit}`;
      if (beforeId) {
        url += `&before_id=${beforeId}`;
      }

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch messages: Status ${response.status}`);
      }

      // 直接返回后端的响应结构
      return await response.json();
    } catch (error) {
      console.error('Failed to get chat history:', error);
      return { messages: [], has_more: false };
    }
  },

  // Send a message to a chat session
  sendMessage: async (sessionId: string, message: string): Promise<any> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}/messages`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message }),
          credentials: 'include'
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to send message: Status ${response.status}`);
      }

      // Return the complete response - includes both user_message and ai_message
      return await response.json();
    } catch (error) {
      console.error('Failed to send message:', error);
      throw error;
    }
  },

  // Get all chat sessions
  getAllChatSessions: async (): Promise<ChatSession[]> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chat-sessions`,
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
      console.error('Failed to get chat sessions:', error);
      throw error;
    }
  },

  // Get a specific chat session
  getChatSession: async (sessionId: string): Promise<ChatSession> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}`,
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
      console.error('Failed to get chat session:', error);
      throw error;
    }
  },

  // 获取会话的文档
  getSessionDocuments: async (sessionId: string): Promise<any> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}/documents`,
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
      console.error('Failed to get session documents:', error);
      throw error;
    }
  },

  // 新增方法：向会话添加文档
  addDocumentToSession: async (sessionId: string, paperId: string): Promise<SessionDocument> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}/documents`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ paper_id: paperId }),
          credentials: 'include'
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.detail?.message || `HTTP error! status: ${response.status}`;
        throw new Error(errorMessage);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to add document to session:', error);
      throw error;
    }
  },

  // 新增方法：从会话移除文档
  removeDocumentFromSession: async (sessionId: string, documentId: string): Promise<any> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}/documents/${documentId}`,
        {
          method: 'DELETE',
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
      console.error('Failed to remove document from session:', error);
      throw error;
    }
  },

  // 更新会话标题
  updateSessionTitle: async (sessionId: string, title: string): Promise<any> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}/title`,
        {
          method: 'PATCH',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ title }),
          credentials: 'include'
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to update session title:', error);
      throw error;
    }
  },

  // 删除会话
  deleteSession: async (sessionId: string): Promise<any> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}`,
        {
          method: 'DELETE',
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
      console.error('Failed to delete session:', error);
      throw error;
    }
  },
};