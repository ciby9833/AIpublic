// frontend/src/services/aiChatService.ts
import { API_BASE_URL } from '../config/env';
import { 
  ChatResponse, 
  ChatMessage, 
  ChatSession, 
  CreateSessionRequest, 
  DocumentRequest,
  SessionDocument
} from '../types/chat';

export interface MessageResponse {
  user_message: ChatMessage;
  ai_message: ChatMessage;
}

export interface MessagesResponse {
  messages: ChatMessage[];
  has_more: boolean;
}

// 文档分析结果接口
export interface DocumentAnalysisResult {
  status: string;
  paper_id: string;
  content: string;
  line_mapping: Record<string, any>;
  total_lines: number;
  is_scanned: boolean;
  has_structured_data: boolean;
}

export const aiChatService = {
  // 1. 文档上传与分析
  uploadDocument: async (file: File, sessionId?: string): Promise<DocumentAnalysisResult> => {
    try {
      console.log(`[AI_CHAT] Uploading document: ${file.name}, size: ${file.size} bytes`);
      
      const formData = new FormData();
      formData.append('file', file);
      
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
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log(`[AI_CHAT] Document uploaded successfully, paper_id: ${result.paper_id}`);
      
      return result;
    } catch (error: unknown) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('上传超时，请稍后重试');
        }
        console.error('[AI_CHAT] Document upload failed:', error);
        throw error;
      }
      throw new Error('文档上传过程中发生未知错误');
    }
  },

  // 2. 会话管理
  // 2.1 创建新会话
  createSession: async (params: CreateSessionRequest): Promise<ChatSession> => {
    try {
      console.log(`[AI_CHAT] Creating new session with ${params.paper_ids?.length || 0} documents`);
      
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

      const result = await response.json();
      console.log(`[AI_CHAT] Session created successfully, id: ${result.id}`);
      
      return result;
    } catch (error) {
      console.error('[AI_CHAT] Failed to create session:', error);
      throw error;
    }
  },
  
  // 2.1.1 创建纯AI对话会话 (无文档)
  createAiOnlySession: async (title?: string): Promise<ChatSession> => {
    const sessionTitle = title || `AI对话 ${new Date().toLocaleString()}`;
    return aiChatService.createSession({
      title: sessionTitle,
      is_ai_only: true,
      paper_ids: []
    });
  },
  
  // 2.1.2 创建文档对话会话 (单文档或多文档)
  createDocumentSession: async (paperIds: string[], title?: string): Promise<ChatSession> => {
    if (!paperIds.length) {
      throw new Error('至少需要一个文档ID');
    }
    
    return aiChatService.createSession({
      title: title || `文档对话 ${new Date().toLocaleString()}`,
      is_ai_only: false,
      paper_ids: paperIds
    });
  },

  // 2.2 获取所有会话
  getAllSessions: async (): Promise<ChatSession[]> => {
    try {
      console.log('[AI_CHAT] Fetching all sessions');
      
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

      const result = await response.json();
      console.log(`[AI_CHAT] Fetched ${result.length} sessions`);
      
      return result;
    } catch (error) {
      console.error('[AI_CHAT] Failed to get sessions:', error);
      throw error;
    }
  },

  // 2.3 获取会话详情
  getSessionDetails: async (sessionId: string): Promise<ChatSession> => {
    try {
      console.log(`[AI_CHAT] Fetching session details for id: ${sessionId}`);
      
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

      const result = await response.json();
      console.log(`[AI_CHAT] Session details fetched successfully for ${result.title}`);
      
      return result;
    } catch (error) {
      console.error('[AI_CHAT] Failed to get session details:', error);
      throw error;
    }
  },
  
  // 2.4 更新会话标题
  updateSessionTitle: async (sessionId: string, newTitle: string): Promise<any> => {
    try {
      console.log(`[AI_CHAT] Updating session ${sessionId} title to "${newTitle}"`);
      
      const response = await fetch(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}/title`,
        {
          method: 'PATCH',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ title: newTitle }),
          credentials: 'include'
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log(`[AI_CHAT] Session title updated successfully`);
      
      return result;
    } catch (error) {
      console.error('[AI_CHAT] Failed to update session title:', error);
      throw error;
    }
  },
  
  // 2.5 删除会话
  deleteSession: async (sessionId: string): Promise<any> => {
    try {
      console.log(`[AI_CHAT] Deleting session: ${sessionId}`);
      
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

      const result = await response.json();
      console.log(`[AI_CHAT] Session deleted successfully`);
      
      return result;
    } catch (error) {
      console.error('[AI_CHAT] Failed to delete session:', error);
      throw error;
    }
  },

  // 3. 消息发送与接收
  // 3.1 发送消息
  streamMessage: async (sessionId: string, message: string, callbacks: {
    onChunk: (chunk: any) => void;
    onComplete: (finalResponse: any) => void;
    onError: (error: any) => void;
  }) => {
    console.log(`[AI_CHAT] Streaming message to session ${sessionId}, length: ${message.length} chars`);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}/stream`,
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
        console.error(`[AI_CHAT] Error streaming message: Status ${response.status}`);
        callbacks.onError(new Error(`Failed to stream message: Status ${response.status}`));
        return;
      }

      // 使用流式读取响应
      const reader = response.body?.getReader();
      if (!reader) {
        callbacks.onError(new Error('Stream reader not available'));
        return;
      }

      // 用于存储部分数据
      let partialData = '';
      let decoder = new TextDecoder();
      let finalResponse = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        // 解码数据块
        const chunk = decoder.decode(value, { stream: true });
        partialData += chunk;

        // 处理完整的JSON对象
        let startIndex = 0;
        let endIndex;
        
        while ((endIndex = partialData.indexOf('\n', startIndex)) !== -1) {
          const jsonStr = partialData.substring(startIndex, endIndex).trim();
          if (jsonStr) {
            try {
              const jsonData = JSON.parse(jsonStr);
              callbacks.onChunk(jsonData);
              
              // 如果收到完成标志，保存最终响应
              if (jsonData.complete || jsonData.done) {
                finalResponse = jsonData;
              }
            } catch (e) {
              console.error('Error parsing JSON chunk:', e, jsonStr);
            }
          }
          startIndex = endIndex + 1;
        }
        
        // 保留未处理的部分数据
        partialData = partialData.substring(startIndex);
      }

      // 处理剩余数据
      if (partialData.trim()) {
        try {
          const jsonData = JSON.parse(partialData.trim());
          callbacks.onChunk(jsonData);
          
          if (jsonData.complete || jsonData.done) {
            finalResponse = jsonData;
          }
        } catch (e) {
          console.error('Error parsing final JSON chunk:', e);
        }
      }

      // 调用完成回调
      if (finalResponse) {
        callbacks.onComplete(finalResponse);
      } else {
        callbacks.onComplete({});
      }
      
      console.log(`[AI_CHAT] Stream completed for session ${sessionId}`);
    } catch (error) {
      console.error('[AI_CHAT] Failed to stream message:', error);
      callbacks.onError(error);
    }
  },
  
  // 3.2 获取会话历史消息
  getMessages: async (sessionId: string, limit: number = 20, beforeId?: string): Promise<MessagesResponse> => {
    try {
      console.log(`[AI_CHAT] Fetching messages for session ${sessionId}, limit: ${limit}, beforeId: ${beforeId || 'none'}`);
      
      let url = `${API_BASE_URL}/api/chat-sessions/${sessionId}/messages?limit=${limit}`;
      if (beforeId) {
        url += `&before_id=${beforeId}`;
      }

      const response = await fetch(
        url,
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

      const result = await response.json();
      console.log(`[AI_CHAT] Fetched ${result.messages?.length || 0} messages, has_more: ${result.has_more}`);
      
      return result;
    } catch (error) {
      console.error('[AI_CHAT] Failed to fetch messages:', error);
      return { messages: [], has_more: false };
    }
  },

  // 4. 文档管理
  // 4.1 向会话添加文档
  addDocumentToSession: async (sessionId: string, paperId: string): Promise<SessionDocument> => {
    try {
      console.log(`[AI_CHAT] Adding document ${paperId} to session ${sessionId}`);
      
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
        // 检查会话文档数量限制错误
        if (errorData.detail?.code === "SESSION_DOCUMENT_LIMIT_REACHED") {
          throw new Error("一个会话最多支持10个文档，请创建新会话");
        }
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log(`[AI_CHAT] Document added successfully to session`);
      
      return result;
    } catch (error) {
      console.error('[AI_CHAT] Failed to add document to session:', error);
      throw error;
    }
  },
  
  // 4.2 获取会话文档列表
  getSessionDocuments: async (sessionId: string): Promise<{ documents: SessionDocument[] }> => {
    try {
      console.log(`[AI_CHAT] Fetching documents for session ${sessionId}`);
      
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

      const result = await response.json();
      console.log(`[AI_CHAT] Fetched ${result.documents?.length || 0} documents for session`);
      
      return result;
    } catch (error) {
      console.error('[AI_CHAT] Failed to fetch session documents:', error);
      throw error;
    }
  },
  
  // 4.3 从会话中删除文档
  removeDocumentFromSession: async (sessionId: string, documentId: string): Promise<any> => {
    try {
      console.log(`[AI_CHAT] Removing document ${documentId} from session ${sessionId}`);
      
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

      const result = await response.json();
      console.log(`[AI_CHAT] Document removed successfully from session`);
      
      return result;
    } catch (error) {
      console.error('[AI_CHAT] Failed to remove document from session:', error);
      throw error;
    }
  },

  // 5. 文档问答
  // 5.1 向文档提问
  askDocumentQuestion: async (question: string, paperId: string): Promise<any> => {
    try {
      console.log(`[AI_CHAT] Asking question to document ${paperId}`);
      
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

      const result = await response.json();
      console.log(`[AI_CHAT] Question answered successfully`);
      
      return result;
    } catch (error) {
      console.error('[AI_CHAT] Failed to ask document question:', error);
      throw error;
    }
  },
  
  // 5.2 获取问答历史
  getQuestionHistory: async (paperId: string): Promise<any[]> => {
    try {
      console.log(`[AI_CHAT] Fetching question history for document ${paperId}`);
      
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
      console.log(`[AI_CHAT] Fetched question history successfully`);
      
      return data.history || [];
    } catch (error) {
      console.error('[AI_CHAT] Failed to fetch question history:', error);
      throw error;
    }
  },
  
  // 获取文档内容
  getDocumentContent: async (paperId: string): Promise<{ content: string }> => {
    try {
      console.log(`[AI_CHAT] Fetching document content for ${paperId}`);
      
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

      const result = await response.json();
      console.log(`[AI_CHAT] Document content fetched successfully`);
      
      return result;
    } catch (error) {
      console.error('[AI_CHAT] Failed to get document content:', error);
      throw error;
    }
  }
};