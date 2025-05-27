// frontend/src/pages/PaperAnalyzer/index.tsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, Upload, Button, Input, message, List, Layout, Spin, Select, Tooltip, Dropdown, Menu, Switch, Tag } from 'antd';
import { 
  InboxOutlined, 
  SendOutlined, 
  FileTextOutlined, 
  TranslationOutlined,
  PaperClipOutlined,
  DeleteOutlined,
  RightOutlined,
  LeftOutlined,
  DownloadOutlined,
  FileWordOutlined,
  FilePdfOutlined,
  FileMarkdownOutlined,
  CloseCircleOutlined,
  CheckCircleOutlined,
  MessageOutlined,
  PlusOutlined,
  EditOutlined,
  HistoryOutlined
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { paperAnalyzerApi } from '../../services/paperAnalyzer';
import './styles.css';
import type { MenuProps } from 'antd';
import ChatMessage from './components/index';
import { ApiChatMessage as ChatMessageType, ChatSession, StreamingState } from '../../types/chat';
import { Modal, Drawer, Empty, Popover, Input as AntInput } from 'antd';
import { useNavigate } from 'react-router-dom';
import { authApi } from '../../services/auth';

const { TextArea } = Input;
const { Sider, Content } = Layout;
const { Option } = Select;

interface LineInfo {
  content: string;
  page?: number;
  start_pos?: number;
  end_pos?: number;
}

interface LineMapping {
  [key: string]: LineInfo;
}

const PaperAnalyzer: React.FC = () => {
  const { t } = useTranslation();
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [responses, setResponses] = useState<ChatMessageType[]>([]);
  const [documentContent, setDocumentContent] = useState<string>('');
  const [currentPaperId, setCurrentPaperId] = useState<string>('');
  const [collapsed, setCollapsed] = useState(false);
  const [translatedContent, setTranslatedContent] = useState<string>('');
  const [selectedLanguage, setSelectedLanguage] = useState<string>('');
  const [languages, setLanguages] = useState<Record<string, string>>({});
  const [translationLoading, setTranslationLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const chatMessagesRef = useRef<HTMLDivElement>(null);
  const [sending, setSending] = useState(false);
  const [lineMapping, setLineMapping] = useState<LineMapping>({});
  const [totalLines, setTotalLines] = useState<number>(0);
  const [analyzing, setAnalyzing] = useState(false);
  const [visibleLines, setVisibleLines] = useState<{start: number, end: number}>({start: 0, end: 50});
  const [syncScrolling, setSyncScrolling] = useState<boolean>(true);
  const originalContentRef = useRef<HTMLDivElement>(null);
  const translatedContentRef = useRef<HTMLDivElement>(null);
  const [siderWidth, setSiderWidth] = useState(800);
  const [isResizing, setIsResizing] = useState(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(0);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [showSessions, setShowSessions] = useState<boolean>(false);
  const [sessionTitle, setSessionTitle] = useState<string>('');
  const [editingTitle, setEditingTitle] = useState<boolean>(false);
  const [creatingSession, setCreatingSession] = useState<boolean>(false);
  const [sessionDocuments, setSessionDocuments] = useState<any[]>([]);
  const [showDocumentsList, setShowDocumentsList] = useState<boolean>(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [hasMoreMessages, setHasMoreMessages] = useState(false);
  const [lastMessageId, setLastMessageId] = useState<string | null>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const [streamingState, setStreamingState] = useState<StreamingState>({
    isStreaming: false,
    partialMessage: ''
  });
  const [isComposing, setIsComposing] = useState(false);

  // 检测是否为移动设备并重定向
  useEffect(() => {
    const checkDeviceAndRedirect = () => {
      const isMobile = window.innerWidth <= 768;
      if (isMobile) {
        navigate('/mobile-chat');
      }
    };
    
    checkDeviceAndRedirect();
    window.addEventListener('resize', checkDeviceAndRedirect);
    
    return () => window.removeEventListener('resize', checkDeviceAndRedirect);
  }, [navigate]);

  // 检查登录状态
  useEffect(() => {
    const checkAuthStatus = () => {
      const accessToken = localStorage.getItem('access_token');
      const expiresAt = localStorage.getItem('expires_at');
      
      if (!accessToken || !expiresAt) {
        console.log('无有效登录信息，将退出登录');
        message.error(t('paperAnalyzer.loginExpired'));
        authApi.logout();
        return;
      }
      
      const now = Date.now() / 1000;
      const expiresTime = Number(expiresAt);
      
      if (now >= expiresTime) {
        console.log('登录已过期，将退出登录');
        message.error(t('paperAnalyzer.loginExpired'));
        authApi.logout();
      }
    };
    
    checkAuthStatus();
    const interval = setInterval(checkAuthStatus, 60000);
    
    return () => clearInterval(interval);
  }, [navigate]);

  // 1. 首先改进scrollToBottom函数，使其更可靠
  const scrollToBottom = (delay = 100) => {
    setTimeout(() => {
      if (chatMessagesRef.current) {
        const scrollHeight = chatMessagesRef.current.scrollHeight;
        const height = chatMessagesRef.current.clientHeight;
        const maxScrollTop = scrollHeight - height;
        chatMessagesRef.current.scrollTop = maxScrollTop > 0 ? maxScrollTop : 0;
        console.log('Scrolling to bottom:', maxScrollTop);
      }
    }, delay);
  };

  // 处理文件拖放
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      handleFileUpload(file);
    }
  };

  // 处理拖拽悬停
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  // 处理文件上传
  const handleFileUpload = async (file: File) => {
    try {
      setAnalyzing(true);
      setLoading(true);
      setSelectedFile(file);
      
      // 1. 文件上传不需要添加临时消息，通过正常的流式API处理

      // 2. 确保有会话ID
      let sessionId = currentSessionId;
      if (!sessionId) {
        const sessionTitle = `${t('paperAnalyzer.documentSession')} ${file.name}`;
        try {
          const session = await handleCreateSession(sessionTitle);
          if (!session || !session.id) {
            throw new Error(t('paperAnalyzer.createSessionFailed'));
          }
          sessionId = session.id;
        } catch (error) {
          console.error("创建会话失败:", error);
          message.error(t('paperAnalyzer.createSessionFailed'));
          setAnalyzing(false);
          setLoading(false);
          return;
        }
      }

      // 3. 发起文件分析请求
      const result = await paperAnalyzerApi.analyzePaper(file, sessionId);
      
      if (result.status === 'success' && result.paper_id) {
        setCurrentPaperId(result.paper_id);
        if (result.content) {
          setDocumentContent(result.content);
          setLineMapping({});
          setTotalLines(result.total_lines || result.content.split('\n').length);
        } else {
          const contentData = await paperAnalyzerApi.getDocumentContent(result.paper_id);
          setDocumentContent(contentData.content);
          setLineMapping(contentData.line_mapping || {});
          setTotalLines(contentData.total_lines || 0);
        }
        message.success(t('paperAnalyzer.documentAnalysisComplete'));
        
        // 4. 处理文档添加到会话
        const wasAddedToSession = result && 'added_to_session' in result 
          ? Boolean(result.added_to_session) 
          : false;
        
        if (sessionId && !wasAddedToSession) {
          try {
            await paperAnalyzerApi.addDocumentToSession(sessionId, result.paper_id);
            console.log(`文档已添加到当前会话: ${sessionId}`);
            
            // 确保这里的代码是异步的，并且等待结果
            await fetchSessionDocuments(sessionId);
            
            // 额外添加一个直接检查而不依赖状态更新的逻辑
            try {
              const sessionDocs = await paperAnalyzerApi.getSessionDocuments(sessionId);
              if (sessionDocs && sessionDocs.documents) {
                setSessionDocuments(sessionDocs.documents);
                console.log(`会话文档数量更新: ${sessionDocs.documents.length}`);
              }
            } catch (error) {
              console.error("获取会话文档失败:", error);
            }
          } catch (error: any) {
            console.error('添加文档到会话失败:', error);
            
            if (error.message && error.message.includes("一个会话最多支持10个文档")) {
              message.error({
                content: t('paperAnalyzer.sessionDocumentLimitReached'),
                duration: 5
              });
              
              Modal.confirm({
                title: t('paperAnalyzer.documentLimitTitle'),
                content: t('paperAnalyzer.documentLimitContent'),
                okText: t('paperAnalyzer.createNewSession'),
                cancelText: t('paperAnalyzer.cancel'),
                onOk: async () => {
                  await handleNewChat();
                  // 安全地处理异步操作
                }
              });
            } else {
              message.error(t('paperAnalyzer.addDocumentFailed'));
            }
          }
        } else if (wasAddedToSession) {
          // 如果文档自动添加到会话，也需要刷新文档列表
          await fetchSessionDocuments(sessionId);
        }

        // 5. 添加分析完成的消息并持久化到后端
        // 先添加到本地状态
        try {
          // 使用流式API发送消息
          await paperAnalyzerApi.streamMessage(
            sessionId,
            t('paperAnalyzer.fileAnalysisComplete', { filename: file.name }),
            {
              onChunk: (chunk) => {
                if (chunk.delta) {
                  setStreamingState(prev => ({
                    isStreaming: true,
                    partialMessage: prev.partialMessage + chunk.delta,
                    messageId: chunk.message_id || prev.messageId
                  }));
                  scrollToBottom();
                }
              },
              onComplete: async () => {
                // 重新获取消息历史
                const result = await paperAnalyzerApi.getChatHistory(sessionId, 20);
                if (result && result.messages && Array.isArray(result.messages)) {
                  setResponses(result.messages);
                  setHasMoreMessages(result.has_more || false);
                  if (result.messages.length > 0) {
                    setLastMessageId(result.messages[0].id || null);
                  }
                }
              },
              onError: (error) => {
                console.error("保存分析完成消息失败:", error);
              }
            }
          );
        } catch (error) {
          console.error("发送消息失败:", error);
          message.error('提问失败');
          setStreamingState({
            isStreaming: false,
            partialMessage: ''
          });
          setSending(false);
        }
              } else {
          message.error(result.message || t('paperAnalyzer.documentAnalysisFailed'));
        }
      } catch (error: any) {
        console.error('Analysis error:', error);
        
        if (error.message && error.message.includes("一个会话最多支持10个文档")) {
          message.error({
            content: t('paperAnalyzer.sessionDocumentLimitReached'),
            duration: 5
          });
          
          Modal.confirm({
            title: t('paperAnalyzer.documentLimitTitle'),
            content: t('paperAnalyzer.documentLimitContent'),
            okText: t('paperAnalyzer.createNewSession'),
            cancelText: t('paperAnalyzer.cancel'),
            onOk: handleNewChat
          });
        } else {
          message.error(t('paperAnalyzer.documentAnalysisFailed'));
        }
    } finally {
      setAnalyzing(false);
      setLoading(false);
    }
  };

  // 2. 在handleAsk函数中发送消息后立即滚动
  const handleAsk = async () => {
    if (!question.trim() && !selectedFile) {
      message.warning(t('paperAnalyzer.questionRequired'));
      return;
    }

    try {
      // If a file is selected but not uploaded, upload it first
      if (selectedFile && !currentPaperId) {
        await handleFileUpload(selectedFile);
        if (!currentPaperId) {
          setSending(false);
          return;
        }
      }
      
      // Save question content
      const questionText = question;
      
      // Clear input and file selection
      setQuestion('');
      setSelectedFile(null);
      
      // Set sending state to disable send button
      setSending(true);
      // 强制滚动到底部显示思考状态
      scrollToBottom(10); // 更短的延迟确保快速响应
      
      // If no session ID exists, create a session first
      let sessionId = currentSessionId;
      if (!sessionId) {
        try {
          const sessionTitle = selectedFile 
            ? `${t('paperAnalyzer.documentSession')} ${selectedFile.name}`
            : `${t('paperAnalyzer.aiOnlySession')} ${new Date().toLocaleString()}`;
          
          const session = await handleCreateSession(sessionTitle);
          
          if (!session || !session.id) {
            message.error(t('paperAnalyzer.createSessionFailed'));
            setSending(false);
            return;
          }
          
          sessionId = session.id;
          setCurrentSessionId(sessionId);
        } catch (error) {
          console.error("创建会话失败:", error);
          message.error(t('paperAnalyzer.createSessionFailed'));
          setSending(false);
          return;
        }
      }
      
      try {
        // 1. 立即添加用户消息到消息列表
        const userMessage: ChatMessageType = {
          id: `user-${Date.now()}`,
          role: 'user',
          content: questionText,
          created_at: new Date().toISOString()
        };
        
        setResponses(prev => [...prev, userMessage]);
        scrollToBottom(10);
        
        // 2. Start streaming state for AI response
        setStreamingState({
          isStreaming: true,
          partialMessage: ''
        });
        
        // 3. Use streaming API
        await paperAnalyzerApi.streamMessage(
          sessionId,
          questionText,
          {
            onChunk: (chunk) => {
              if (chunk.delta) {
                setStreamingState(prev => ({
                  isStreaming: true,
                  partialMessage: prev.partialMessage + chunk.delta,
                  messageId: chunk.message_id || prev.messageId
                }));
                scrollToBottom();
              }
            },
            onComplete: async (finalResponse) => {
              // Reset streaming state
              setStreamingState({
                isStreaming: false,
                partialMessage: ''
              });
              
              // 获取完整的消息列表并完全替换当前的responses
              const updatedMessages = await paperAnalyzerApi.getChatHistory(sessionId, 20);
              if (updatedMessages && updatedMessages.messages) {
                setResponses(updatedMessages.messages);
                setHasMoreMessages(updatedMessages.has_more || false);
                if (updatedMessages.messages.length > 0) {
                  setLastMessageId(updatedMessages.messages[0].id || null);
                }
              }
              
              // Final scroll to bottom
              scrollToBottom();
              setSending(false);
            },
            onError: (error) => {
              console.error("消息流处理失败:", error);
              message.error(t('paperAnalyzer.sendFailed'));
              setStreamingState({
                isStreaming: false,
                partialMessage: ''
              });
              setSending(false);
            }
          }
        );
      } catch (error) {
        console.error("发送消息失败:", error);
        message.error(t('paperAnalyzer.sendFailed'));
        setStreamingState({
          isStreaming: false,
          partialMessage: ''
        });
        setSending(false);
      }
    } catch (error) {
      console.error("处理问题失败:", error);
      message.error(t('paperAnalyzer.processingFailed'));
      setSending(false);
    }
  };

  // 获取支持的语言列表
  useEffect(() => {
    const fetchLanguages = async () => {
      try {
        const supportedLanguages = await paperAnalyzerApi.getSupportedLanguages();
        setLanguages(supportedLanguages);
      } catch (error) {
        console.error('Failed to fetch languages:', error);
      }
    };
    fetchLanguages();
  }, []);

  // 处理翻译
  const handleTranslate = async () => {
    if (!currentPaperId || !selectedLanguage) {
      message.warning(t('paperAnalyzer.selectTargetLanguage'));
      return;
    }

    try {
      setTranslationLoading(true);
      const translated = await paperAnalyzerApi.translatePaper(currentPaperId, selectedLanguage);
      setTranslatedContent(translated);
      message.success(t('paperAnalyzer.translationComplete'));
    } catch (error: any) {
      console.error('Translation error:', error);
      message.error(error.message || t('paperAnalyzer.translationFailed'));
    } finally {
      setTranslationLoading(false);
    }
  };

  const handleDownload = async (format: string) => {
    if (!translatedContent) {
      message.warning(t('paperAnalyzer.noTranslationContent'));
      return;
    }

    try {
      const response = await paperAnalyzerApi.downloadTranslation(
        currentPaperId,
        selectedLanguage,
        format
      );
      
      // 创建下载链接
      const blob = new Blob([response], { 
        type: format === 'pdf' ? 'application/pdf' : 
              format === 'docx' ? 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' :
              'text/markdown'
      });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `translated_${selectedFile?.name.split('.')[0]}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download error:', error);
      message.error(t('paperAnalyzer.downloadFailed'));
    }
  };

  // 在组件内定义菜单项
  const downloadMenuItems: MenuProps['items'] = [
    {
      key: 'docx',
      label: t('paperAnalyzer.wordDocument'),
      icon: <FileWordOutlined />,
    },
    {
      key: 'pdf',
      label: t('paperAnalyzer.pdfDocument'),
      icon: <FilePdfOutlined />,
    },
    {
      key: 'md',
      label: t('paperAnalyzer.markdown'),
      icon: <FileMarkdownOutlined />,
    },
  ];

  // Update scroll handlers for the new content structure
  const handleOriginalScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    if (!syncScrolling || !translatedContentRef.current || !translatedContent) return;
    
    const originalElement = e.currentTarget;
    const originalScrollable = originalElement.scrollHeight - originalElement.clientHeight;
    if (originalScrollable <= 0) return;
    
    const scrollPercentage = originalElement.scrollTop / originalScrollable;
    
    const translatedElement = translatedContentRef.current.querySelector('.content-wrapper');
    if (!translatedElement) return;
    
    const translatedScrollable = translatedElement.scrollHeight - translatedElement.clientHeight;
    if (translatedScrollable <= 0) return;
    
    translatedElement.scrollTop = scrollPercentage * translatedScrollable;
  }, [syncScrolling, translatedContent]);

  const handleTranslatedScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    if (!syncScrolling || !originalContentRef.current || !documentContent) return;
    
    const translatedElement = e.currentTarget;
    const translatedScrollable = translatedElement.scrollHeight - translatedElement.clientHeight;
    if (translatedScrollable <= 0) return;
    
    const scrollPercentage = translatedElement.scrollTop / translatedScrollable;
    
    const originalElement = originalContentRef.current.querySelector('.content-wrapper');
    if (!originalElement) return;
    
    const originalScrollable = originalElement.scrollHeight - originalElement.clientHeight;
    if (originalScrollable <= 0) return;
    
    originalElement.scrollTop = scrollPercentage * originalScrollable;
  }, [syncScrolling, documentContent]);

  // Fix the resizing handlers with proper TypeScript typing
  const handleResizeStart = (e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsResizing(true);
    startXRef.current = e.clientX;
    startWidthRef.current = siderWidth;
    document.addEventListener('mousemove', handleResizeMove);
    document.addEventListener('mouseup', handleResizeEnd);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  };

  const handleResizeMove = useCallback((e: MouseEvent) => {
    if (!isResizing) return;
    const delta = e.clientX - startXRef.current;
    const newWidth = Math.max(300, Math.min(1000, startWidthRef.current + delta));
    setSiderWidth(newWidth);
  }, [isResizing]);

  const handleResizeEnd = useCallback(() => {
    setIsResizing(false);
    document.removeEventListener('mousemove', handleResizeMove);
    document.removeEventListener('mouseup', handleResizeEnd);
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, [handleResizeMove]);

  // Clean up event listeners with proper dependencies
  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleResizeMove);
      document.removeEventListener('mouseup', handleResizeEnd);
    };
  }, [handleResizeMove, handleResizeEnd]);

  // Add this function to handle session creation
  const handleCreateSession = async (title?: string) => {
    try {
      setCreatingSession(true);
      
      // 确保title是有效的字符串
      const finalTitle = title || `${t('paperAnalyzer.newChat')} ${new Date().toLocaleString()}`;
      
      // 如果当前已有会话ID且没有消息，则不创建新会话
      if (currentSessionId && responses.length === 0) {
        console.log("使用当前空白会话，无需创建新会话");
        return { id: currentSessionId, title: sessionTitle };
      }
      
      let session;
      
      if (currentPaperId) {
        // 创建基于文档的会话
        session = await paperAnalyzerApi.createChatSession({
          title: finalTitle,
          paper_ids: [currentPaperId],
          is_ai_only: false
        });
      } else {
        // 创建纯AI会话
        session = await paperAnalyzerApi.createChatSession({
          title: finalTitle,
          is_ai_only: true,
          paper_ids: []
        });
      }
      
      if (session && session.id) {
        setCurrentSessionId(session.id);
        setSessionTitle(session.title);
        
        try {
          await fetchSessions();
        } catch (err) {
          console.error("获取会话列表失败，但会话创建成功", err);
        }
        
        // 关键修改：不清空现有消息，保留临时状态
        // setResponses([]);  // 删除这一行
        setShowSessions(false);
        
        return session;
      } else {
        throw new Error(t('paperAnalyzer.createSessionFailed2'));
      }
    } catch (error) {
      console.error("创建会话失败:", error);
      message.error(t('paperAnalyzer.createSessionFailed'));
      return null;
    } finally {
      setCreatingSession(false);
    }
  };

  // 修改显示历史会话的按钮处理函数
  const showSessionHistory = async () => {
    try {
      setLoading(true);
      
      // Always use getAllChatSessions instead of filtering by paper_id
      // This ensures we see all user's chat sessions
      const sessionList = await paperAnalyzerApi.getAllChatSessions();
      
      if (Array.isArray(sessionList)) {
        setSessions(sessionList);
      } else {
        console.error('Invalid session list format:', sessionList);
        message.error(t('paperAnalyzer.loadSessionFailed'));
      }
      
      setShowSessions(true);
      console.log("显示会话历史，当前会话数:", sessionList.length);
    } catch (error) {
      console.error("获取会话历史失败:", error);
      message.error(t('paperAnalyzer.loadSessionFailed'));
    } finally {
      setLoading(false);
    }
  };

  // 修改fetchSessions函数来添加更好的错误处理和加载状态
  const fetchSessions = async () => {
    try {
      // Always use getAllChatSessions instead of using paper-specific endpoint
      const sessionList = await paperAnalyzerApi.getAllChatSessions();
      
      // 只有在成功获取会话列表时才更新状态
      if (Array.isArray(sessionList)) {
        setSessions(sessionList);
      } else {
        console.error('Invalid session list format:', sessionList);
        message.error(t('paperAnalyzer.loadSessionFailed'));
      }
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
      message.error(t('paperAnalyzer.loadSessionFailed'));
      throw error; // 重新抛出错误以便调用者处理
    }
  };

  // 优化 fetchSessionDocuments 函数
  const fetchSessionDocuments = async (sessionId: string) => {
    if (!sessionId) return;
    
    try {
      console.log(`正在获取会话(${sessionId})的文档...`);
      const result = await paperAnalyzerApi.getSessionDocuments(sessionId);
      
      if (result && result.documents) {
        console.log(`会话文档获取成功，数量: ${result.documents.length}`);
        setSessionDocuments(result.documents);
        return result.documents; // 返回文档列表以便调用者使用
      } else {
        console.log('会话没有文档或返回格式错误');
        setSessionDocuments([]);
        return [];
      }
    } catch (error) {
      console.error('加载会话文档失败:', error);
      setSessionDocuments([]);
      return [];
    }
  };

  // Modify loadSession function to fetch session documents
  const loadSession = async (sessionId: string) => {
    if (sessionId === currentSessionId) {
      console.log("点击当前会话，无需重新加载");
      setShowSessions(false);
      return;
    }
    
    try {
      setLoading(true);
      
      // 获取聊天历史
      const result = await paperAnalyzerApi.getChatHistory(sessionId, 20);
      
      // 更新当前会话信息
      setCurrentSessionId(sessionId);
      
      // 更新sessionTitle
      const currentSession = sessions.find(s => s.id === sessionId);
      if (currentSession) {
        setSessionTitle(currentSession.title || t('paperAnalyzer.unnamedSession'));
      }
      
      // 设置消息内容 - 使用标准格式
      if (result && result.messages) {
        setResponses(result.messages);
        setHasMoreMessages(result.has_more || false);
        if (result.messages.length > 0) {
          setLastMessageId(result.messages[0].id || null);
        }
      } else {
        setResponses([]);
        setHasMoreMessages(false);
        setLastMessageId(null);
      }
      
      try {
        // 尝试加载会话文档
        await fetchSessionDocuments(sessionId);
      } catch (docError) {
        console.error('Failed to load session documents:', docError);
        setSessionDocuments([]);
      }
      
      // 关闭会话列表窗口
      setShowSessions(false);
      
      // 滚动到底部显示最新消息
      if (result && result.messages && result.messages.length > 0) {
        scrollToBottom(300);
      }
      
    } catch (error) {
      console.error('加载对话历史失败:', error);
      message.error(t('paperAnalyzer.loadSessionFailed'));
    } finally {
      setLoading(false);
    }
  };

  // Add a useEffect to load session documents when currentSessionId changes
  useEffect(() => {
    if (currentSessionId) {
      fetchSessionDocuments(currentSessionId);
    } else {
      setSessionDocuments([]);
    }
  }, [currentSessionId]);

  // 修改自动加载最近会话的逻辑
  useEffect(() => {
    // 页面加载时获取最新的会话
    const loadLatestSession = async () => {
      try {
        // 只在组件首次加载且没有当前会话ID时执行
        if (!currentSessionId) {
          setLoading(true);
          
          try {
            // 获取所有会话
            const allSessions = await paperAnalyzerApi.getAllChatSessions()
              .catch(err => {
                console.error('获取会话列表失败:', err);
                return [];
              });
            
            if (Array.isArray(allSessions) && allSessions.length > 0) {
              // 找到创建时间最新的活跃会话
              const latestSession = allSessions[0]; // API返回已经按时间排序
              
              try {
                // 尝试加载最新会话
                const result = await paperAnalyzerApi.getChatHistory(latestSession.id, 20);
                
                // 使用新返回格式
                setResponses(result.messages);
                setHasMoreMessages(result.has_more);
                setLastMessageId(result.messages.length > 0 ? result.messages[0].id : null);
                
                setCurrentSessionId(latestSession.id);
                setSessionTitle(latestSession.title || t('paperAnalyzer.unnamedSession'));
                console.log("自动加载最新会话:", latestSession.title);
              } catch (sessionError) {
                console.error("加载会话失败:", sessionError);
                // 如果加载失败，创建一个空状态
                setSessionTitle(t('paperAnalyzer.newChat'));
                setResponses([]);
                setCurrentSessionId("");
              }
            } else {
              // 如果没有会话，创建一个新的空会话状态
              setSessionTitle(t('paperAnalyzer.newChat'));
              setResponses([]);
              setCurrentSessionId("");
            }
          } catch (error) {
            console.error("加载会话数据失败:", error);
            // 创建一个空状态
            setSessionTitle(t('paperAnalyzer.newChat'));
            setResponses([]);
            setCurrentSessionId("");
          }
        }
      } catch (error) {
        console.error("自动加载会话失败:", error);
      } finally {
        setLoading(false);
      }
    };

    loadLatestSession();
  }, []); // 空依赖数组确保只在组件挂载时执行一次

  // 添加处理函数：点击查看文档
  const handleViewDocument = async (paperId: string, filename: string) => {
    try {
      setLoading(true);
      
      // 设置为当前查看的文档
      setCurrentPaperId(paperId);
      
      // 加载文档内容
      const contentData = await paperAnalyzerApi.getDocumentContent(paperId);
      setDocumentContent(contentData.content);
      setLineMapping(contentData.line_mapping || {});
      setTotalLines(contentData.total_lines || contentData.content.split('\n').length);
      
      message.success(t('paperAnalyzer.documentLoaded', { filename }));
    } catch (error) {
      console.error('加载文档失败:', error);
      message.error(t('paperAnalyzer.loadDocumentFailed'));
    } finally {
      setLoading(false);
    }
  };

  // Add useEffect to fetch sessions when paper changes
  useEffect(() => {
    if (currentPaperId) {
      fetchSessions();
    }
  }, [currentPaperId]);

  // 替换handleTitleSave方法，调用后端API保存标题
  const handleTitleSave = async () => {
    if (!currentSessionId || !sessionTitle.trim()) {
      setEditingTitle(false);
      return;
    }
    
    try {
      // 调用API更新标题
      const result = await paperAnalyzerApi.updateSessionTitle(currentSessionId, sessionTitle);
      
      if (result.status === 'success') {
        message.success(t('paperAnalyzer.titleUpdated'));
        
        // 如果在会话列表中，也更新该会话的标题
        if (sessions.length > 0) {
          setSessions(sessions.map(session => 
            session.id === currentSessionId 
              ? { ...session, title: sessionTitle } 
              : session
          ));
        }
      }
      
      setEditingTitle(false);
    } catch (error: any) {
      message.error(t('paperAnalyzer.updateTitleFailed', { message: error.message }));
      console.error('保存标题失败:', error);
      setEditingTitle(false);
    }
  };

  // 添加删除会话功能
  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    // 阻止点击事件传播（避免触发行点击事件）
    e.stopPropagation();
    
    // 弹窗确认
    Modal.confirm({
      title: t('paperAnalyzer.confirmDelete'),
      content: t('paperAnalyzer.deleteConfirmText'),
      okText: t('paperAnalyzer.delete'),
      okType: 'danger',
      cancelText: t('paperAnalyzer.cancel'),
      onOk: async () => {
        try {
          await paperAnalyzerApi.deleteSession(sessionId);
          
          // 删除成功后，更新会话列表
          setSessions(sessions.filter(session => session.id !== sessionId));
          
          // 如果删除的是当前会话，则清空当前会话
          if (sessionId === currentSessionId) {
            setCurrentSessionId("");
            setResponses([]);
            setSessionTitle(t('paperAnalyzer.newChat'));
            setShowSessions(false);
          }
          
          message.success(t('paperAnalyzer.sessionDeleted'));
        } catch (error: any) {
          message.error(t('paperAnalyzer.deleteSessionFailed', { message: error.message }));
          console.error('删除对话失败:', error);
        }
      }
    });
  };

  // Modify the "New Chat" button handler
  const handleNewChat = async () => {
    // 如果当前没有响应内容，且没有会话ID，则不需要创建新会话
    if (responses.length === 0 && !currentSessionId) {
      console.log("当前已经是空白新会话，无需创建");
      return;
    }
    
    // 清除当前会话和消息
    setCurrentSessionId("");
    setResponses([]);
    setSessionTitle(t('paperAnalyzer.newChat'));
    setShowSessions(false);
    setSelectedFile(null);
    setCurrentPaperId("");
    setDocumentContent("");
  };

  // Modify the toggleDocumentsList function
  const toggleDocumentsList = async () => {
    // Refresh the documents list when opening
    if (!showDocumentsList && currentSessionId) {
      await fetchSessionDocuments(currentSessionId);
    }
    setShowDocumentsList(!showDocumentsList);
  };

  // 保留 useEffect 监听 responses 变化，但移除 CSS 布局相关的修改
  useEffect(() => {
    if (responses.length > 0 && !showSessions) {
      scrollToBottom();
    }
  }, [responses, showSessions]);

  // Function to load older messages
  const loadMoreMessages = async () => {
    if (!currentSessionId || isLoadingMore || !hasMoreMessages || !lastMessageId) return;
    
    try {
      setIsLoadingMore(true);
      
      const result = await paperAnalyzerApi.getChatHistory(
        currentSessionId, 
        20,
        lastMessageId
      );
      
      if (result && result.messages && result.messages.length > 0) {
        // 将旧消息添加到消息数组开头
        setResponses(prev => [...result.messages, ...prev]);
        
        // 更新分页信息
        if (result.messages.length > 0) {
          setLastMessageId(result.messages[0].id || null);
        }
        setHasMoreMessages(result.has_more || false);
      } else {
        setHasMoreMessages(false);
      }
    } catch (error) {
      console.error('加载更多消息失败:', error);
      message.error(t('paperAnalyzer.loadingMoreMessages'));
    } finally {
      setIsLoadingMore(false);
    }
  };

  // Handle chat scroll to load more messages
  const handleChatScroll = useCallback(() => {
    if (!chatContainerRef.current) return;
    
    const { scrollTop } = chatContainerRef.current;
    // If scrolled to top (or near top) and has more messages, load more
    if (scrollTop < 50 && hasMoreMessages && !isLoadingMore) {
      loadMoreMessages();
    }
  }, [hasMoreMessages, isLoadingMore, currentSessionId, lastMessageId]);

  // Add scroll event listener to chat container
  useEffect(() => {
    const container = chatContainerRef.current;
    if (container) {
      container.addEventListener('scroll', handleChatScroll);
      return () => {
        container.removeEventListener('scroll', handleChatScroll);
      };
    }
  }, [handleChatScroll]);

  // 3. 添加专门监听sending和analyzing状态变化的useEffect
  useEffect(() => {
    if (sending || analyzing) {
      // 当进入思考或分析状态时滚动到底部
      scrollToBottom(10);
      
      // 设置一个额外的定时器以确保在DOM更新后滚动
      const timer = setTimeout(() => scrollToBottom(100), 100);
      return () => clearTimeout(timer);
    }
  }, [sending, analyzing]);

  // 4. 在streamingState变化时也触发滚动
  useEffect(() => {
    if (streamingState.isStreaming) {
      scrollToBottom(10);
    }
  }, [streamingState]);

  return (
    <Layout className="paper-analyzer-layout">
      <Sider 
        width={siderWidth}
        collapsible={false}
        collapsed={collapsed}
        className="paper-analyzer-sider"
        style={{ transition: isResizing ? 'none' : '' }}
      >
        <div className="chat-container" ref={chatContainerRef}>
          <div className="chat-header">
            {showSessions ? (
              <div className="session-header">
                <Button 
                  className="return-button"
                  onClick={() => setShowSessions(false)}
                  icon={<LeftOutlined />}
                >
                  {t('paperAnalyzer.returnToChat')}
                </Button>
                {responses.length > 0 && (
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => {
                  handleNewChat();
                  setShowSessions(false);  // 立即关闭会话列表
                }}
                loading={creatingSession}
              >
                {t('paperAnalyzer.newChat')}
              </Button>
                )}
            </div>
            ) : (
              <>
                <div className="session-info">
                  {editingTitle ? (
                    <div className="edit-title">
                      <AntInput
                        value={sessionTitle}
                        onChange={(e) => setSessionTitle(e.target.value)}
                        onPressEnter={handleTitleSave}
                        size="small"
                        autoFocus
                      />
                      <Button
                        size="small" 
                        type="primary"
                        onClick={handleTitleSave}
                      >
                        {t('paperAnalyzer.saveTitle')}
                      </Button>
                    </div>
                  ) : (
                    <>
                      <h3 className="session-title" onClick={() => setEditingTitle(true)}>
                        {sessionTitle || t('paperAnalyzer.newChat')}
                        {currentSessionId && <EditOutlined className="edit-icon" />}
                      </h3>
                      <div className="session-actions">
                        {responses.length > 0 && (
                          <Tooltip title={t('paperAnalyzer.createNewSession')}>
                            <Button
                              type="text"
                              icon={<PlusOutlined />}
                              onClick={() => {
                                handleNewChat();
                                setShowSessions(false);  // 立即关闭会话列表
                              }}
                              loading={creatingSession}
                              disabled={showSessions}
                            />
                          </Tooltip>
                        )}
                        <Button 
                          type="text" 
                          icon={<HistoryOutlined />} 
                          onClick={showSessionHistory}
                          title={t('paperAnalyzer.chatHistory')}
                        />
                        <Button
                          type="text"
                          icon={collapsed ? <RightOutlined /> : <LeftOutlined />}
                          onClick={() => setCollapsed(!collapsed)}
                          className="collapse-button"
                          title={collapsed ? t('paperAnalyzer.expand') : t('paperAnalyzer.collapse')}
                        />
                      </div>
                    </>
                  )}
                </div>
              </>
            )}
          </div>
          {showSessions ? (
            <div className="sessions-list">
              {loading && (
                <div className="sessions-list-loading">
                  <Spin tip={t('paperAnalyzer.loadingChatHistory')} />
                </div>
              )}
              {sessions.length > 0 ? (
                <List
                  dataSource={sessions}
                  renderItem={(session) => (
                    <List.Item 
                      className={`session-item ${session.id === currentSessionId ? 'active' : ''}`}
                      onClick={() => loadSession(session.id)}
                    >
                      <div className="session-item-content">
                        <div className="session-title">
                          {session.is_ai_only ? <MessageOutlined /> : <FileTextOutlined />}
                          <span>{session.title}</span>
                        </div>
                        <div className="session-meta">
                          <span className="session-time">
                            {new Date(session.updated_at).toLocaleString('zh-CN', {
                              month: 'numeric',
                              day: 'numeric',
                              hour: '2-digit',
                              minute: '2-digit'
                            })}
                          </span>
                          <span className="session-count">{t('paperAnalyzer.messagesCount', { count: session.message_count })}</span>
                          
                          {/* 添加删除按钮 */}
                          <Button
                            type="text"
                            danger
                            icon={<DeleteOutlined />}
                            size="small"
                            title={t('paperAnalyzer.deleteSession')}
                            onClick={(e) => handleDeleteSession(session.id, e)}
                            className="delete-session-btn"
                          />
                        </div>
                        <div className="session-preview">{session.last_message}</div>
                        {session.documents && session.documents.length > 0 && (
                          <div className="session-documents">
                            {session.documents.map(doc => (
                              <Tag 
                                key={doc.id} 
                                color="blue"
                                className="clickable-document-tag"
                                onClick={(e) => {
                                  // 阻止事件冒泡，避免触发会话点击事件
                                  e.stopPropagation();
                                  // 先加载会话
                                  loadSession(session.id);
                                  // 然后显示特定文档 - 这个在loadSession中已经自动关闭会话列表了
                                  handleViewDocument(doc.paper_id, doc.filename);
                                }}
                              >
                                <FileTextOutlined /> {doc.filename}
                              </Tag>
                            ))}
                          </div>
                        )}
                      </div>
                    </List.Item>
                  )}
                />
              ) : (
                !loading && <Empty description={t('paperAnalyzer.noSessionHistory')} />
              )}
            </div>
          ) : (
            <div className="chat-messages" ref={chatMessagesRef}>
              {isLoadingMore && (
                <div className="loading-more-messages">
                  <Spin size="small" /> {t('paperAnalyzer.loadingMoreMessages')}
                </div>
              )}
              <List
                className="response-list"
                itemLayout="vertical"
                dataSource={responses}
                renderItem={(item, index) => (
                  <ChatMessage 
                    message={item} 
                    key={index}
                  />
                )}
              />
              {streamingState.isStreaming && (
                <ChatMessage 
                  message={{
                    id: `streaming-${Date.now()}`,
                    role: 'assistant',
                    content: streamingState.partialMessage || t('paperAnalyzer.thinking'),
                    created_at: new Date().toISOString()
                  }}
                />
              )}
              {/* 移除重复的thinking状态，streamingState.isStreaming已经处理了思考状态 */}
            </div>
          )}
          
          <div className="chat-input-container">
            <div 
              className={`input-area ${analyzing ? 'analyzing' : ''} ${showSessions ? 'disabled' : ''}`}
              onDrop={!showSessions ? handleDrop : undefined}
              onDragOver={!showSessions ? handleDragOver : undefined}
            >
              <TextArea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder={showSessions ? t('paperAnalyzer.inputPlaceholderSelectSession') : analyzing ? t('paperAnalyzer.inputPlaceholderAnalyzing') : t('paperAnalyzer.inputPlaceholder')}
                rows={3}
                disabled={analyzing || showSessions}
                onCompositionStart={() => setIsComposing(true)}
                onCompositionEnd={() => setIsComposing(false)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !isComposing && !analyzing && !showSessions) {
                    e.preventDefault();
                    handleAsk();
                  }
                }}
              />
              <div className="input-actions">
                <div className="left-actions">
                  {!showSessions ? (
                    <>
                      <Upload
                        accept=".pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xls,.txt,.md"
                        maxCount={1}
                        showUploadList={false}
                        beforeUpload={(file) => {
                          handleFileUpload(file);
                          return false;
                        }}
                        disabled={analyzing || showSessions}
                      >
                        <Tooltip title={analyzing ? t('paperAnalyzer.analyzing') : t('paperAnalyzer.uploadFile')}>
                          <Button 
                            icon={<PaperClipOutlined />} 
                            disabled={analyzing || showSessions}
                          />
                        </Tooltip>
                      </Upload>
                      {sessionDocuments.length > 0 && (
                        <Tooltip title={t('paperAnalyzer.viewSessionDocuments')}>
                          <Button
                            icon={<FileTextOutlined />}
                            onClick={toggleDocumentsList}
                            data-has-documents={sessionDocuments.length > 0}
                          />
                        </Tooltip>
                      )}
                    </>
                  ) : (
                    <></>
                  )}
                </div>
                {!showSessions && (
              <Button
                type="primary"
                icon={<SendOutlined />}
                onClick={handleAsk}
                    loading={sending || analyzing}
                    disabled={analyzing || showSessions}
              >
                    {analyzing ? t('paperAnalyzer.analyzing') : t('paperAnalyzer.send')}
              </Button>
                  )}
              </div>
            </div>
          </div>
        </div>

      {/* Add resize handle */}
      <div 
        className="resize-handle"
        onMouseDown={handleResizeStart}
      >
        <div className="handle-bar"></div>
      </div>
    </Sider>

    <Content className="paper-analyzer-content">
      <div className="document-viewer">
        <div className="document-header">
          <div className="header-left">
            <FileTextOutlined /> {selectedFile?.name}
          </div>
          <div className="header-right">
            <Select
              style={{ width: 200 }}
              placeholder={t('paperAnalyzer.selectTargetLanguage')}
              value={selectedLanguage}
              onChange={setSelectedLanguage}
            >
              {Object.entries(languages).map(([code, name]) => (
                <Option key={code} value={code}>{name}</Option>
              ))}
            </Select>
            <Button
              type="primary"
              icon={<TranslationOutlined />}
              onClick={handleTranslate}
              loading={translationLoading}
              disabled={!selectedLanguage}
            >
              {t('paperAnalyzer.translate')}
            </Button>
          </div>
        </div>
        <div className="document-content-split">
          <div className="original-content" ref={originalContentRef}>
            <h3>{t('paperAnalyzer.originalText')}</h3>
            {documentContent ? (
              <div className="content-wrapper" 
                style={{height: 'calc(100% - 30px)', overflow: 'auto'}}
                onScroll={handleOriginalScroll}
              >
                {documentContent.split('\n').map((line, index) => (
                  <div key={index} className="line-container">
                    <span className="line-number">{index + 1}</span>
                    <span className="line-content">{line}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-document">
                <FileTextOutlined style={{ fontSize: '24px' }} />
                <p>{t('paperAnalyzer.uploadDocumentPrompt')}</p>
              </div>
            )}
          </div>
          <div className="translated-content" ref={translatedContentRef}>
            <div className="content-header">
              <h3>{t('paperAnalyzer.translation')}</h3>
              <div className="header-actions">
                {translatedContent && (
                  <>
                    <Switch
                      size="small"
                      checked={syncScrolling}
                      onChange={(checked) => setSyncScrolling(checked)}
                      checkedChildren={t('paperAnalyzer.syncScrolling')}
                      unCheckedChildren={t('paperAnalyzer.independentScrolling')}
                    />
                    <Dropdown
                      menu={{
                        items: downloadMenuItems,
                        onClick: ({ key }) => handleDownload(key as string)
                      }}
                      trigger={['click']}
                    >
                      <Button 
                        type="text" 
                        icon={<DownloadOutlined />} 
                        className="download-button"
                        title={t('paperAnalyzer.downloadTranslation')}
                      />
                    </Dropdown>
                  </>
                )}
              </div>
            </div>
            {translatedContent ? (
              <div className="content-wrapper" 
                style={{height: 'calc(100% - 30px)', overflow: 'auto'}}
                onScroll={handleTranslatedScroll}
              >
                {translatedContent.split('\n').map((line, index) => (
                  <div key={index} className="line-container">
                    <span className="line-number">{index + 1}</span>
                    <span className="line-content">{line}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-translation">
                <TranslationOutlined style={{ fontSize: '24px' }} />
                <p>{t('paperAnalyzer.selectLanguagePrompt')}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </Content>

    {/* Add documents list drawer right before the final Layout closing tag */}
    <Drawer
      title={t('paperAnalyzer.sessionDocuments')}
      placement="right"
      open={showDocumentsList}
      onClose={toggleDocumentsList}
      width={320}
    >
      {sessionDocuments.length > 0 ? (
        <List
          dataSource={sessionDocuments}
          renderItem={(doc) => (
            <List.Item 
              key={doc.id} 
              className="session-document-item"
              onClick={() => {
                handleViewDocument(doc.paper_id, doc.filename);
                setShowDocumentsList(false);
              }}
            >
              <div className="document-item-content">
                <div className="document-title">
                  <FileTextOutlined />
                  <span>{doc.filename}</span>
                </div>
                <div className="document-info">
                  {doc.upload_time && (
                    <span className="document-time">
                      {new Date(doc.upload_time).toLocaleString('zh-CN', {
                        month: 'numeric',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </span>
                  )}
                </div>
              </div>
            </List.Item>
          )}
        />
      ) : (
        <Empty description={t('paperAnalyzer.noDocuments')} />
      )}
    </Drawer>
  </Layout>
  );
};

export default PaperAnalyzer;