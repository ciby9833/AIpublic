// frontend/src/services/paperAnalyzer.ts  文档阅读
import { API_BASE_URL } from '../config/env';
import { apiRequest } from './auth';
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

      const response = await apiRequest(
        `${API_BASE_URL}/api/paper/analyze`,
        {
          method: 'POST',
          body: formData,
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
      const response = await apiRequest(
        `${API_BASE_URL}/api/paper/${paperId}/content`,
        {
          method: 'GET'
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

  // 提问 - 已废弃，请使用会话模式
  askQuestion: async (question: string, paperId: string): Promise<QuestionResponse> => {
    throw new Error('单次问答模式已废弃，请使用会话模式');
  },

  // 获取问答历史 - 已废弃，请使用会话模式
  getQuestionHistory: async (paperId: string): Promise<QuestionHistory[]> => {
    throw new Error('问答历史API已废弃，请使用会话模式');
  },

  // 获取支持的语言列表
  getSupportedLanguages: async (): Promise<Record<string, string>> => {
    try {
      const response = await apiRequest(
        `${API_BASE_URL}/api/paper/supported-languages`,
        {
          method: 'GET'
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
      const response = await apiRequest(
        `${API_BASE_URL}/api/paper/${paperId}/translate`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
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
    const response = await apiRequest(`${API_BASE_URL}/api/paper/${paperId}/download`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
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
      const response = await apiRequest(
        `${API_BASE_URL}/api/paper/${paperId}/chat-sessions`,
        {
          method: 'GET'
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
      const response = await apiRequest(
        `${API_BASE_URL}/api/chat-sessions`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(params)
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
      is_ai_only: true, // 确保设置为true
      paper_ids: [] // 空数组表示无文档
    });
  },

  // Get chat history for a session
  getChatHistory: async (sessionId: string, limit: number = 20, beforeId?: string): Promise<{messages: any[], has_more: boolean}> => {
    console.log(`[CHAT_API] Getting chat history for session ${sessionId}, limit: ${limit}, beforeId: ${beforeId || 'none'}`);
    try {
      let url = `${API_BASE_URL}/api/chat-sessions/${sessionId}/messages?limit=${limit}`;
      if (beforeId) {
        url += `&before_id=${beforeId}`;
      }

      console.log(`[CHAT_API] Sending GET request to: ${url}`);
      const response = await apiRequest(url, {
        method: 'GET'
      });

      if (!response.ok) {
        console.error(`[CHAT_API] Error fetching messages: Status ${response.status}`);
        throw new Error(`Failed to fetch messages: Status ${response.status}`);
      }

      // 直接返回后端的响应结构
      const result = await response.json();
      console.log(`[CHAT_API] Received ${result.messages?.length || 0} messages, has_more: ${result.has_more}`);
      return result;
    } catch (error) {
      console.error('[CHAT_API] Failed to get chat history:', error);
      return { messages: [], has_more: false };
    }
  },

  // Get all chat sessions
  getAllChatSessions: async (): Promise<ChatSession[]> => {
    try {
      const response = await apiRequest(
        `${API_BASE_URL}/api/chat-sessions`,
        {
          method: 'GET'
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
      const response = await apiRequest(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}`,
        {
          method: 'GET'
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
      const response = await apiRequest(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}/documents`,
        {
          method: 'GET'
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
      const response = await apiRequest(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}/documents`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ paper_id: paperId })
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
      const response = await apiRequest(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}/documents/${documentId}`,
        {
          method: 'DELETE'
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
      const response = await apiRequest(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}/title`,
        {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ title })
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
      const response = await apiRequest(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}`,
        {
          method: 'DELETE'
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

  // streamMessage改为唯一的消息发送方法，参考ChatGPT实现
  streamMessage: async (sessionId: string, message: string, callbacks: {
    onChunk: (chunk: any) => void;
    onComplete: (finalResponse: any) => void;
    onError: (error: any) => void;
  }) => {
    console.log(`[CHAT_API] Streaming message to session ${sessionId}, length: ${message.length} chars`);
    
    // 尝试使用SSE，如果不支持则降级到ReadableStream
    const useSSE = typeof EventSource !== 'undefined';
    
    if (useSSE) {
      // 使用SSE实现，参考ChatGPT
      return paperAnalyzerApi.streamMessageWithSSE(sessionId, message, callbacks);
    } else {
      // 降级到ReadableStream实现
      return paperAnalyzerApi.streamMessageWithStream(sessionId, message, callbacks);
    }
  },

  // SSE优先的流式消息实现
  streamMessageWithSSE: async (sessionId: string, message: string, callbacks: {
    onChunk: (chunk: any) => void;
    onComplete: (finalResponse: any) => void;
    onError: (error: any) => void;
  }) => {
    try {
      // 直接使用后端的stream端点，它返回SSE格式的数据
      const response = await apiRequest(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}/stream`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message })
        }
      );

      if (!response.ok) {
        console.error(`[CHAT_API] Error streaming message: Status ${response.status}`);
        callbacks.onError(new Error(`Failed to stream message: Status ${response.status}`));
        return;
      }

      // 处理SSE流式响应
      const reader = response.body?.getReader();
      if (!reader) {
        callbacks.onError(new Error('Stream reader not available'));
        return;
      }

      let finalResponse: any = null;
      let decoder = new TextDecoder();
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          // 解码数据块
          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;

          // 处理SSE格式的数据 (data: {json}\n\n)
          let lines = buffer.split('\n');
          buffer = lines.pop() || ''; // 保留最后一个可能不完整的行

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6); // 移除 'data: ' 前缀
              
              if (data === '[DONE]') {
                // 流结束
                console.log('[SSE] Stream ended with [DONE] signal');
                if (finalResponse) {
                  callbacks.onComplete(finalResponse);
                }
                return;
              }

              try {
                const jsonData = JSON.parse(data);
                console.log('[SSE] Received data:', jsonData);
                
                // 处理不同类型的消息
                if (jsonData.content && !jsonData.done && !jsonData.saved && !jsonData.content_complete) {
                  // 流式内容块 - 正在接收AI回复的内容
                  console.log('[SSE] Content chunk received:', jsonData.content.length, 'chars');
                  callbacks.onChunk({
                    delta: jsonData.content,
                    done: false,
                    message_id: jsonData.id || jsonData.message_id
                  });
                } else if (jsonData.message_id && jsonData.saved === true && jsonData.done === true) {
                  // 最终保存确认 - 这是关键的确认信号，表示消息已成功保存到数据库
                  console.log('[SSE] Backend save confirmation received:', jsonData.message_id);
                  finalResponse = jsonData;
                  callbacks.onChunk({
                    delta: '',
                    done: true,
                    message_id: jsonData.message_id,
                    saved: true,
                    sources: jsonData.sources || [],
                    confidence: jsonData.confidence || 0.0,
                    reply: jsonData.reply || []
                  });
                  // 立即调用完成回调
                  callbacks.onComplete(finalResponse);
                  return;
                } else if (jsonData.error) {
                  // 错误处理
                  console.error('[SSE] Error received:', jsonData.error);
                  callbacks.onError(new Error(jsonData.error));
                  return;
                } else if ((jsonData.done === true || jsonData.content_complete === true) && !jsonData.saved && !jsonData.error) {
                  // 内容流结束但没有保存确认 - 继续等待保存确认
                  console.log('[SSE] Content stream done, waiting for save confirmation...');
                  callbacks.onChunk({
                    delta: '',
                    done: false, // 设置为false，表示还在等待最终确认
                    message_id: jsonData.id || jsonData.message_id,
                    saved: false,
                    content_complete: true
                  });
                  // 不要在这里调用onComplete，继续等待保存确认
                } else if (jsonData.partial_response) {
                  // 处理 partial_response 格式的内容
                  console.log('[SSE] Partial response received:', jsonData.partial_response.length, 'chars');
                  callbacks.onChunk({
                    delta: jsonData.partial_response,
                    done: jsonData.done || false,
                    message_id: jsonData.id || jsonData.message_id
                  });
                } else {
                  // 其他类型的消息，记录但不处理
                  console.log('[SSE] Other message type:', jsonData);
                }
                
              } catch (e) {
                console.error('[SSE] Error parsing data:', e, data);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }

      // 如果循环结束但没有收到最终确认，使用最后的响应
      if (finalResponse) {
        console.log('[SSE] Stream ended, calling onComplete with final response');
        callbacks.onComplete(finalResponse);
      } else {
        console.warn('[SSE] Stream ended without final response');
        callbacks.onComplete({});
      }

      // 返回清理函数
      return () => {
        reader.cancel();
      };
      
    } catch (error) {
      console.error('[CHAT_API] Failed to setup SSE stream:', error);
      callbacks.onError(error);
    }
  },

  // 保留原有的ReadableStream实现作为降级方案
  streamMessageWithStream: async (sessionId: string, message: string, callbacks: {
    onChunk: (chunk: any) => void;
    onComplete: (finalResponse: any) => void;
    onError: (error: any) => void;
  }) => {
    try {
      const response = await apiRequest(
        `${API_BASE_URL}/api/chat-sessions/${sessionId}/stream`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message })
        }
      );

      if (!response.ok) {
        console.error(`[CHAT_API] Error streaming message: Status ${response.status}`);
        callbacks.onError(new Error(`Failed to stream message: Status ${response.status}`));
        return;
      }

      // Get reader from response body stream
      const reader = response.body?.getReader();
      if (!reader) {
        callbacks.onError(new Error('Stream reader not available'));
        return;
      }

      // Used to store partial chunks
      let partialData = '';
      let decoder = new TextDecoder();
      let finalResponse = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        // Decode and process the chunk
        const chunk = decoder.decode(value, { stream: true });
        partialData += chunk;

        // Process any complete JSON objects in the stream
        let startIndex = 0;
        let endIndex;
        
        while ((endIndex = partialData.indexOf('\n', startIndex)) !== -1) {
          const jsonStr = partialData.substring(startIndex, endIndex).trim();
          if (jsonStr) {
            try {
              const jsonData = JSON.parse(jsonStr);
              console.log('[Stream] Received data:', jsonData);
              
              // 处理不同类型的消息 - 与SSE实现保持一致
              if (jsonData.content && !jsonData.done && !jsonData.saved) {
                // 流式内容块 - 正在接收AI回复的内容
                console.log('[Stream] Content chunk received:', jsonData.content.length, 'chars');
                callbacks.onChunk({
                  delta: jsonData.content,
                  done: false,
                  message_id: jsonData.message_id
                });
              } else if (jsonData.message_id && jsonData.saved === true && jsonData.done === true) {
                // 最终保存确认 - 这是关键的确认信号，表示消息已成功保存到数据库
                console.log('[Stream] Backend save confirmation received:', jsonData.message_id);
                finalResponse = jsonData;
                callbacks.onChunk({
                  delta: '',
                  done: true,
                  message_id: jsonData.message_id,
                  saved: true,
                  sources: jsonData.sources || [],
                  confidence: jsonData.confidence || 0.0,
                  reply: jsonData.reply || []
                });
                // 立即调用完成回调
                callbacks.onComplete(finalResponse);
                return;
              } else if (jsonData.error) {
                // 错误处理
                console.error('[Stream] Error received:', jsonData.error);
                callbacks.onError(new Error(jsonData.error));
                return;
              } else if (jsonData.done === true && !jsonData.saved) {
                // 流结束但没有保存确认 - 这种情况需要等待保存确认
                console.log('[Stream] Stream done without save confirmation, waiting for save...');
                callbacks.onChunk({
                  delta: '',
                  done: false, // 设置为false，等待保存确认
                  message_id: jsonData.message_id,
                  saved: false
                });
                // 不要在这里调用onComplete，等待保存确认
              } else {
                // 其他类型的消息，记录但不处理
                console.log('[Stream] Other message type:', jsonData);
              }
              
            } catch (e) {
              console.error('[Stream] Error parsing JSON chunk:', e, jsonStr);
            }
          }
          startIndex = endIndex + 1;
        }
        
        // Keep any remaining partial data
        partialData = partialData.substring(startIndex);
      }

      // Process any remaining data
      if (partialData.trim()) {
        try {
          const jsonData = JSON.parse(partialData.trim());
          console.log('[Stream] Processing remaining data:', jsonData);
          
          // 使用与主循环相同的处理逻辑
          if (jsonData.content && !jsonData.done && !jsonData.saved) {
            callbacks.onChunk({
              delta: jsonData.content,
              done: false,
              message_id: jsonData.message_id
            });
          } else if (jsonData.message_id && jsonData.saved === true && jsonData.done === true) {
            finalResponse = jsonData;
            callbacks.onChunk({
              delta: '',
              done: true,
              message_id: jsonData.message_id,
              saved: true,
              sources: jsonData.sources || [],
              confidence: jsonData.confidence || 0.0,
              reply: jsonData.reply || []
            });
            callbacks.onComplete(finalResponse);
            return;
          } else if (jsonData.error) {
            callbacks.onError(new Error(jsonData.error));
            return;
          } else if (jsonData.done === true && !jsonData.saved) {
            callbacks.onChunk({
              delta: '',
              done: false,
              message_id: jsonData.message_id,
              saved: false
            });
          }
          
        } catch (e) {
          console.error('[Stream] Error parsing final JSON chunk:', e);
        }
      }

      // Call onComplete with the final complete response
      if (finalResponse) {
        console.log('[Stream] Stream ended, calling onComplete with final response');
        callbacks.onComplete(finalResponse);
      } else {
        console.warn('[Stream] Stream ended without final response');
        callbacks.onComplete({});
      }
      
      console.log(`[CHAT_API] Stream completed for session ${sessionId}`);
    } catch (error) {
      console.error('[CHAT_API] Failed to stream message:', error);
      callbacks.onError(error);
    }
  },
};