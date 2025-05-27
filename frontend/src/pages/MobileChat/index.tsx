import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Input, Button, Spin, Modal, Upload, List, message, Tooltip } from 'antd';
import { 
  SendOutlined,
  MenuOutlined, 
  FileTextOutlined,
  UserOutlined,
  RobotOutlined,
  LeftOutlined,
  PlusOutlined,
  HistoryOutlined,
  DeleteOutlined,
  EditOutlined,
  CloseOutlined,
  LoadingOutlined,
  PaperClipOutlined,
  FileOutlined,
  CopyOutlined,
  CheckOutlined
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { aiChatService } from '../../services/aiChatService';
import { ApiChatMessage, ChatSession, StreamingState } from '../../types/chat';
import { MessagesResponse } from '../../services/aiChatService';
import ReactMarkdown from 'react-markdown';
import './mobileChat.css';
import { authApi } from '../../services/auth';

const { TextArea } = Input;

const MobileChat: React.FC = () => {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const [messageText, setMessageText] = useState<string>('');
  const [messages, setMessages] = useState<ApiChatMessage[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [sending, setSending] = useState<boolean>(false);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [showSessions, setShowSessions] = useState<boolean>(false);
  const [showMenu, setShowMenu] = useState<boolean>(false);
  const [sessionTitle, setSessionTitle] = useState<string>(t('mobileChat.newChat'));
  const [editingTitle, setEditingTitle] = useState<boolean>(false);
  const [hasMoreMessages, setHasMoreMessages] = useState<boolean>(false);
  const [lastMessageId, setLastMessageId] = useState<string | null>(null);
  const [isLoadingMore, setIsLoadingMore] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const [streamingState, setStreamingState] = useState<StreamingState>({
    isStreaming: false,
    partialMessage: ''
  });
  const [isComposing, setIsComposing] = useState(false);
  const [keyboardOpen, setKeyboardOpen] = useState(false);
  const [inputFocused, setInputFocused] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [copyingMessageId, setCopyingMessageId] = useState<string | null>(null);

  // 页面加载时获取最新的会话
  useEffect(() => {
    const loadLatestSession = async () => {
      try {
        setLoading(true);
        const sessionList = await aiChatService.getAllSessions();
        
        if (Array.isArray(sessionList) && sessionList.length > 0) {
          const latestSession = sessionList[0];
          setSessions(sessionList);
          
          try {
            const result = await aiChatService.getMessages(latestSession.id, 20) as MessagesResponse;
            const chatMessages = (result?.messages || []) as unknown as ApiChatMessage[];
            setMessages(chatMessages);
            setHasMoreMessages(Boolean(result?.has_more));
            setLastMessageId((chatMessages[0] as unknown as ApiChatMessage)?.id ?? null);
            
            setCurrentSessionId(latestSession.id);
            setSessionTitle(latestSession.title || t('mobileChat.unnamedSession'));
            
            // 自动滚动到最新消息
            setTimeout(() => scrollToBottom(0), 300);
          } catch (sessionError) {
            console.error(t('mobileChat.loadSessionFailed'), sessionError);
            message.error(t('mobileChat.loadLatestSessionFailed'));
            setSessionTitle(t('mobileChat.newChat'));
            setMessages([]);
            setCurrentSessionId("");
          }
        } else {
          setSessionTitle(t('mobileChat.newChat'));
          setMessages([]);
          setCurrentSessionId("");
        }
      } catch (error) {
        console.error(t('mobileChat.loadSessionListFailed'), error);
        message.error(t('mobileChat.loadSessionListFailed'));
      } finally {
        setLoading(false);
      }
    };

    loadLatestSession();
  }, []);

  // 滚动到底部
  const scrollToBottom = (delay = 100) => {
    setTimeout(() => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
      }
    }, delay);
  };

  // 智能滚动 - 专门处理键盘弹出时的滚动
  const smartScroll = () => {
    if (!chatContainerRef.current || !messagesEndRef.current) return;
    
    const container = chatContainerRef.current;
    
    // 简化滚动逻辑，直接滚动到底部
    setTimeout(() => {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
      });
    }, 100);
  };

  // 处理文件上传
  const handleFileUpload = async (file: File) => {
    try {
      setUploading(true);
      console.log(`[MOBILE_CHAT] Starting file upload: ${file.name}, size: ${file.size} bytes`);
      
      // 确保有会话ID
      let sessionId = currentSessionId;
      if (!sessionId) {
        try {
          const sessionTitle = `${t('mobileChat.fileUploaded').replace('{{filename}}', file.name)}`;
          // 创建纯AI会话，稍后添加文档
          const newSession = await aiChatService.createAiOnlySession(sessionTitle);
          
          if (newSession && newSession.id) {
            sessionId = newSession.id;
            setCurrentSessionId(sessionId);
            setSessionTitle(sessionTitle);
          } else {
            throw new Error(t('mobileChat.createSessionFailed'));
          }
        } catch (error) {
          console.error(t('mobileChat.createSessionFailed'), error);
          message.error(t('mobileChat.createSessionFailed'));
          setUploading(false);
          return;
        }
      }

      // 上传并分析文档
      const result = await aiChatService.uploadDocument(file, sessionId);
      
      if (result.status === 'success' && result.paper_id) {
        console.log(`[MOBILE_CHAT] Document uploaded successfully, paper_id: ${result.paper_id}`);
        
        // 检查文档是否已自动添加到会话
        const wasAddedToSession = result && 'added_to_session' in result 
          ? Boolean(result.added_to_session) 
          : false;
        
        if (!wasAddedToSession) {
          // 只有在文档没有自动添加到会话时才手动添加
          try {
            await aiChatService.addDocumentToSession(sessionId, result.paper_id);
            console.log(`[MOBILE_CHAT] Document manually added to session: ${sessionId}`);
          } catch (error: any) {
            console.error(t('mobileChat.uploadFailed'), error);
            if (error.message && error.message.includes("一个会话最多支持10个文档")) {
              message.error(t('mobileChat.sessionLimitReached'));
              return;
            } else if (error.message && error.message.includes("该文档已添加到会话中")) {
              console.log(t('mobileChat.documentAlreadyInSession'));
            } else {
              // 其他错误仍然抛出
              throw error;
            }
          }
        } else {
          console.log(`[MOBILE_CHAT] Document was automatically added to session: ${sessionId}`);
        }

        // 发送文档分析完成的消息
        const analysisMessage = t('mobileChat.fileAnalysisComplete', { filename: file.name });
        
        // 添加用户消息到UI
        const tempUserMessage: ApiChatMessage = {
          id: `temp-upload-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
          role: 'user',
          content: t('mobileChat.fileUploaded', { filename: file.name }),
          created_at: new Date().toISOString(),
          sources: [],
          confidence: 0,
          reply: [{
            type: 'markdown',
            content: `📎 ${t('mobileChat.fileUploaded', { filename: file.name })}`
          }]
        };
        
        setMessages(prev => [...prev, tempUserMessage]);
        
        // 开始流式状态
        setStreamingState({
          isStreaming: true,
          partialMessage: ''
        });
        
        // 使用流式API发送分析完成消息
        await aiChatService.streamMessage(
          sessionId,
          analysisMessage,
          {
            onChunk: (chunk) => {
              if (chunk && chunk.delta) {
                setStreamingState(prev => ({
                  isStreaming: true,
                  partialMessage: prev.partialMessage + chunk.delta,
                  messageId: chunk.message_id || prev.messageId
                }));
                scrollToBottom();
              }
            },
            onComplete: async () => {
              // 重置流式状态
              setStreamingState({
                isStreaming: false,
                partialMessage: ''
              });
              
              // 获取完整的消息列表
              try {
                const updatedMessages = await aiChatService.getMessages(sessionId, 20);
                if (updatedMessages && updatedMessages.messages) {
                  setMessages((updatedMessages.messages || []) as unknown as ApiChatMessage[]);
                  setHasMoreMessages(Boolean(updatedMessages.has_more));
                  if (updatedMessages.messages.length > 0) {
                    setLastMessageId((updatedMessages.messages[0] as unknown as ApiChatMessage)?.id ?? null);
                  }
                }
              } catch (err) {
                console.error(t('mobileChat.loadSessionFailed'), err);
              }
              
              scrollToBottom();
              setUploading(false);
            },
            onError: (error) => {
              const errorMessage = typeof error === 'string' ? error : 
                                  error?.message || t('mobileChat.unknownError');
              console.error(t('mobileChat.documentAnalysisFailed'), error);
              message.error(t('mobileChat.documentAnalysisFailed', { message: errorMessage }));
              setStreamingState({
                isStreaming: false,
                partialMessage: ''
              });
              setUploading(false);
            }
          }
        );
        
        message.success(t('mobileChat.documentUploadSuccess'));
        setSelectedFile(null);
      } else {
        message.error(t('mobileChat.documentAnalysisFailed2'));
        setUploading(false);
      }
    } catch (error: any) {
      console.error('[MOBILE_CHAT] File upload failed:', error);
      
      if (error.message && error.message.includes("一个会话最多支持10个文档")) {
        message.error(t('mobileChat.sessionLimitReached'));
      } else {
        message.error(t('mobileChat.uploadFailed'));
      }
      setUploading(false);
    }
  };

  // 处理拖拽上传
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      
      // 检查文件大小 (限制为50MB)
      const maxSize = 50 * 1024 * 1024; // 50MB
      if (file.size > maxSize) {
        message.error(t('mobileChat.fileSizeLimit'));
        return;
      }
      
      // 检查文件类型
      const allowedTypes = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.txt', '.md'];
      const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
      
      if (allowedTypes.includes(fileExtension)) {
        setSelectedFile(file);
        message.success(t('mobileChat.fileSelected', { filename: file.name }));
      } else {
        message.error(t('mobileChat.unsupportedFileType'));
      }
    }
  };

  // 复制消息内容
  const handleCopyMessage = async (messageId: string, content: string) => {
    try {
      setCopyingMessageId(messageId);
      
      // 清理markdown格式，提取纯文本
      const cleanContent = content
        .replace(/```[\s\S]*?```/g, (match) => match.replace(/```\w*\n?/g, '').replace(/```/g, ''))
        .replace(/`([^`]+)`/g, '$1')
        .replace(/\*\*([^*]+)\*\*/g, '$1')
        .replace(/\*([^*]+)\*/g, '$1')
        .replace(/#{1,6}\s+/g, '')
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        .trim();
      
      await navigator.clipboard.writeText(cleanContent);
      message.success(t('mobileChat.copied'));
      
      // 2秒后重置复制状态
      setTimeout(() => setCopyingMessageId(null), 2000);
    } catch (error) {
      console.error(t('mobileChat.copyFailed'), error);
      message.error(t('mobileChat.copyFailed'));
      setCopyingMessageId(null);
    }
  };

  // 发送消息
  const handleSendMessage = async () => {
    if (!messageText.trim() && !selectedFile) {
      return;
    }

    // 如果有选中的文件，先上传文件
    if (selectedFile) {
      await handleFileUpload(selectedFile);
      // 上传完成后，如果没有消息文本，就结束
      if (!messageText.trim()) {
        return;
      }
    }

    const userMessage = messageText;
    setMessageText('');
    setSending(true);
    
    // 立即滚动到底部显示思考状态
    scrollToBottom(10);

    try {
      // 如果没有会话ID，先创建会话
      let sessionId = currentSessionId;
      if (!sessionId) {
        try {
          const newSession = await aiChatService.createAiOnlySession();
          
          if (newSession && newSession.id) {
            sessionId = newSession.id;
            setCurrentSessionId(sessionId);
            setSessionTitle(newSession.title || t('mobileChat.newChat'));
          } else {
            throw new Error(t('mobileChat.createSessionFailed'));
          }
        } catch (error) {
          console.error(t('mobileChat.createSessionFailed'), error);
          message.error(t('mobileChat.createSessionFailed'));
          setSending(false);
          return;
        }
      }

      // 添加用户消息到UI - 使用临时ID以便于识别
      const tempUserMessage: ApiChatMessage = {
        id: `temp-user-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
        role: 'user',
        content: userMessage,
        created_at: new Date().toISOString(),
        sources: [],
        confidence: 0,
        reply: [{
          type: 'markdown',
          content: userMessage
        }]
      };
      
      setMessages(prev => [...prev, tempUserMessage]);
      
      // 开始流式状态
      setStreamingState({
        isStreaming: true,
        partialMessage: ''
      });
      
      // 使用流式API
      await aiChatService.streamMessage(
        sessionId,
        userMessage,
        {
          onChunk: (chunk) => {
            if (chunk && chunk.delta) {
              setStreamingState(prev => ({
                isStreaming: true,
                partialMessage: prev.partialMessage + chunk.delta,
                messageId: chunk.message_id || prev.messageId
              }));
              scrollToBottom();
            }
          },
          onComplete: async () => {
            // 重置流式状态
            setStreamingState({
              isStreaming: false,
              partialMessage: ''
            });
            
            // 获取完整的消息列表
            try {
              const updatedMessages = await aiChatService.getMessages(sessionId, 20) as MessagesResponse;
              if (updatedMessages && updatedMessages.messages) {
                setMessages((updatedMessages.messages || []) as unknown as ApiChatMessage[]);
                setHasMoreMessages(Boolean(updatedMessages.has_more));
                if (updatedMessages.messages.length > 0) {
                  setLastMessageId((updatedMessages.messages[0] as unknown as ApiChatMessage)?.id ?? null);
                }
              }
            } catch (err) {
              console.error(t('mobileChat.loadSessionFailed'), err);
            }
            
            scrollToBottom();
            setSending(false);
          },
          onError: (error) => {
            const errorMessage = typeof error === 'string' ? error : 
                                error?.message || t('mobileChat.unknownError');
            console.error(t('mobileChat.sendFailed'), error);
            message.error(t('mobileChat.sendFailed'));
            setStreamingState({
              isStreaming: false,
              partialMessage: ''
            });
            setSending(false);
          }
        }
      );
    } catch (error) {
      console.error(t('mobileChat.sendFailed'), error);
      message.error(t('mobileChat.sendFailed'));
      setSending(false);
    }
  };

  // 加载更多消息
  const loadMoreMessages = async () => {
    if (!currentSessionId || isLoadingMore || !hasMoreMessages || !lastMessageId) return;
    
    try {
      setIsLoadingMore(true);
      
      const result = await aiChatService.getMessages(
        currentSessionId, 
        20,
        lastMessageId
      ) as MessagesResponse;
      
      if (result && result.messages) {
        const newMessages = Array.isArray(result.messages) ? result.messages : [];
        const hasMore = Boolean(result.has_more);
        
        // 将旧消息添加到现有消息前面
        const typedNewMessages = newMessages as unknown as ApiChatMessage[];
        setMessages(prev => [...typedNewMessages, ...prev]);
        setHasMoreMessages(hasMore);
        if (newMessages.length > 0) {
          setLastMessageId((newMessages[0] as unknown as ApiChatMessage)?.id ?? null);
        }
      } else {
        setHasMoreMessages(false);
      }
    } catch (error) {
      console.error(t('mobileChat.loadingMore'), error);
      message.error(t('mobileChat.loadingMore'));
    } finally {
      setIsLoadingMore(false);
    }
  };

  // 处理滚动加载更多消息
  const handleChatScroll = () => {
    if (!chatContainerRef.current) return;
    
    const { scrollTop } = chatContainerRef.current;
    // 滚动到顶部附近时加载更多消息
    if (scrollTop < 50 && hasMoreMessages && !isLoadingMore) {
      loadMoreMessages();
    }
  };

  // 获取所有会话
  const fetchSessions = async () => {
    try {
      const sessionList = await aiChatService.getAllSessions();
      if (Array.isArray(sessionList)) {
        setSessions(sessionList);
      } else {
        console.error('Invalid session list format:', sessionList);
        message.error(t('mobileChat.loadSessionFailed'));
      }
    } catch (error) {
      console.error(t('mobileChat.loadSessionFailed'), error);
      message.error(t('mobileChat.loadSessionFailed'));
    }
  };

  // 显示会话列表
  const showSessionHistory = async () => {
    try {
      setLoading(true);
      await fetchSessions();
      setShowSessions(true);
      setShowMenu(false);
    } catch (error) {
      console.error(t('mobileChat.loadSessionFailed'), error);
      message.error(t('mobileChat.loadSessionFailed'));
    } finally {
      setLoading(false);
    }
  };

  // 加载会话
  const loadSession = async (sessionId: string) => {
    if (sessionId === currentSessionId) {
      setShowSessions(false);
      return;
    }
    
    try {
      setLoading(true);
      
      // 获取聊天历史
      const result = await aiChatService.getMessages(sessionId, 20) as MessagesResponse;
      
      // 更新当前会话信息
      setCurrentSessionId(sessionId);
      
      // 更新sessionTitle
      const currentSession = sessions.find(s => s.id === sessionId);
      if (currentSession) {
        setSessionTitle(currentSession.title || t('mobileChat.unnamedSession'));
      }
      
      // 设置消息内容
      if (result && result.messages) {
        setMessages((result.messages || []) as unknown as ApiChatMessage[]);
        setHasMoreMessages(Boolean(result.has_more));
        if (result.messages.length > 0) {
          setLastMessageId((result.messages[0] as unknown as ApiChatMessage)?.id ?? null);
        }
      } else {
        setMessages([]);
        setHasMoreMessages(false);
        setLastMessageId(null);
      }
      
      // 关闭会话列表窗口
      setShowSessions(false);
      
      // 滚动到底部
      scrollToBottom(300);
    } catch (error) {
      console.error(t('mobileChat.loadSessionFailed'), error);
      message.error(t('mobileChat.loadSessionFailed'));
    } finally {
      setLoading(false);
    }
  };

  // 创建新会话
  const handleNewChat = () => {
    // 如果当前没有消息且没有会话ID，则不需要创建新会话
    if (messages.length === 0 && !currentSessionId) {
      console.log(t('mobileChat.alreadyNewChat'));
      setShowSessions(false);
      setShowMenu(false);
      return;
    }
    
    // 清除当前会话和消息
    setCurrentSessionId("");
    setMessages([]);
    setSessionTitle(t('mobileChat.newChat'));
    setShowSessions(false);
    setShowMenu(false);
  };

  // 删除会话
  const handleDeleteSession = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSessionToDelete(sessionId);
    setDeleteModalVisible(true);
  };

  // 取消删除
  const cancelDeleteSession = () => {
    setDeleteModalVisible(false);
    setSessionToDelete(null);
  };

  // 确认删除
  const confirmDeleteSession = async () => {
    if (!sessionToDelete) return;
    
    try {
      setLoading(true);
      const result = await aiChatService.deleteSession(sessionToDelete);
      
      // 删除成功后，更新会话列表
      setSessions((prev) => prev.filter(session => session.id !== sessionToDelete));
      
      // 如果删除的是当前会话，则清空当前会话
      if (sessionToDelete === currentSessionId) {
        setCurrentSessionId("");
        setMessages([]);
        setSessionTitle(t('mobileChat.newChat'));
        setShowSessions(false);
      }
      
      message.success(t('mobileChat.sessionDeleted'));
    } catch (error: any) {
      // 安全地提取错误消息
      const errorMessage = error?.message || t('mobileChat.unknownError');
      message.error(t('mobileChat.deleteSessionFailed', { message: errorMessage }));
      console.error(t('mobileChat.deleteSessionFailed'), error);
    } finally {
      setDeleteModalVisible(false);
      setSessionToDelete(null);
      setLoading(false);
    }
  };

  // 更新会话标题
  const handleTitleSave = async () => {
    if (!currentSessionId || !sessionTitle.trim()) {
      setEditingTitle(false);
      return;
    }
    
    try {
      // 调用API更新标题
      const result = await aiChatService.updateSessionTitle(currentSessionId, sessionTitle);
      
      // 检查结果格式并显示消息
      if (result && result.status === 'success') {
        message.success(t('mobileChat.titleUpdated'));
        
        // 更新会话列表中的标题
        setSessions(sessions.map(session => 
          session.id === currentSessionId 
            ? { ...session, title: sessionTitle } 
            : session
        ));
      } else {
        message.success(t('mobileChat.titleUpdated'));
      }
      
      setEditingTitle(false);
    } catch (error: any) {
      // 使用安全的错误消息提取方式
      const errorMessage = error?.message || t('mobileChat.unknownError');
      message.error(t('mobileChat.updateTitleFailed', { message: errorMessage }));
      console.error(t('mobileChat.updateTitleFailed'), error);
      setEditingTitle(false);
    }
  };

  useEffect(() => {
    const checkDeviceAndRedirect = () => {
      const isDesktop = window.innerWidth > 768;
      if (isDesktop) {
        navigate('/paper-analyzer'); // 或主页面路径
      }
    };
    
    checkDeviceAndRedirect();
    window.addEventListener('resize', checkDeviceAndRedirect);
    
    return () => window.removeEventListener('resize', checkDeviceAndRedirect);
  }, [navigate]);

  useEffect(() => {
    const checkAuthStatus = () => {
      const accessToken = localStorage.getItem('access_token');
      const expiresAt = localStorage.getItem('expires_at');
      
      if (!accessToken || !expiresAt) {
        handleGracefulLogout(t('mobileChat.loginExpired'));
        return;
      }
      
      const now = Date.now() / 1000;
      const expiresTime = Number(expiresAt);
      
      // 提前5分钟检测到过期，给用户友好提示
      if (now >= expiresTime - 300) { // 5分钟预警
        handleGracefulLogout(t('mobileChat.loginExpiring'));
      }
    };
    
    // 初始检查
    checkAuthStatus();
    
    // 设置定期检查 (每分钟检查一次)
    const interval = setInterval(checkAuthStatus, 60000);
    
    return () => clearInterval(interval);
  }, [navigate]);

  // 添加平滑退出函数
  const handleGracefulLogout = (msg: string) => {
    // 保存当前输入内容到localStorage
    if (messageText) {
      localStorage.setItem('draft_message', messageText);
    }
    
    // 使用modal而不是message，确保用户看到
    Modal.warning({
      title: t('mobileChat.loginStatusWarning'),
      content: msg,
      okText: t('mobileChat.confirm'),
      onOk: () => {
        authApi.logout();
      }
    });
    
    // 5秒后自动登出
    setTimeout(() => {
      authApi.logout();
    }, 5000);
  };

  // 监听发送状态变化，确保滚动到底部
  useEffect(() => {
    if (sending) {
      scrollToBottom(10);
      
      // 设置额外定时器确保DOM更新后滚动
      const timer = setTimeout(() => scrollToBottom(100), 100);
      return () => clearTimeout(timer);
    }
  }, [sending]);

  // 监听流式状态变化，确保滚动到底部
  useEffect(() => {
    if (streamingState.isStreaming) {
      scrollToBottom(10);
    }
  }, [streamingState]);

  // 键盘弹出检测和处理
  useEffect(() => {
    let initialViewportHeight = window.visualViewport?.height || window.innerHeight;
    let keyboardHeight = 0;
    
    const handleViewportChange = () => {
      if (window.visualViewport) {
        const currentHeight = window.visualViewport.height;
        const heightDifference = initialViewportHeight - currentHeight;
        
        // 如果高度差超过150px，认为键盘弹出了
        const isKeyboardOpen = heightDifference > 150;
        keyboardHeight = isKeyboardOpen ? heightDifference : 0;
        
        setKeyboardOpen(isKeyboardOpen);
        
        // 移除手动高度计算，让CSS flexbox处理布局
        
                 // 键盘弹出时智能滚动
        if (isKeyboardOpen && inputFocused) {
          // 延迟滚动，确保布局调整完成
          setTimeout(() => {
            smartScroll();
          }, 200);
        }
      }
    };

    const handleResize = () => {
      // 更新初始视口高度（仅在键盘收起时）
      if (!keyboardOpen) {
        initialViewportHeight = window.visualViewport?.height || window.innerHeight;
      }
      handleViewportChange();
    };

    // 监听视口变化
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', handleViewportChange);
    }
    
    // 监听窗口大小变化（兼容性处理）
    window.addEventListener('resize', handleResize);

    return () => {
      if (window.visualViewport) {
        window.visualViewport.removeEventListener('resize', handleViewportChange);
      }
      window.removeEventListener('resize', handleResize);
    };
  }, [keyboardOpen, inputFocused]);

  // 清理函数
  useEffect(() => {
    return () => {
      // 组件卸载时清理选中的文件
      setSelectedFile(null);
    };
  }, []);

  return (
    <div 
      className={`x-chat-container ${keyboardOpen ? 'keyboard-open' : ''} ${dragOver ? 'drag-over' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* 头部导航区 */}
      <div className="x-chat-header">
        {showSessions ? (
          <div className="x-session-header">
            <Button 
              icon={<LeftOutlined />}
              type="text"
              onClick={() => setShowSessions(false)}
              className="x-back-button"
            />
            <div className="x-header-title">{t('mobileChat.sessionHistory')}</div>
          </div>
        ) : (
          <>
            <div className="x-header-content">
              <div 
                className="x-chat-title" 
                onClick={() => setEditingTitle(true)}
              >
                {editingTitle ? (
                  <div className="x-edit-title">
                    <Input
                      value={sessionTitle}
                      onChange={(e) => setSessionTitle(e.target.value)}
                      onBlur={handleTitleSave}
                      onKeyDown={(e) => e.key === 'Enter' && handleTitleSave()}
                      autoFocus
                      className="x-title-input"
                      maxLength={30}
                    />
                  </div>
                ) : (
                  <div className="x-title-display">
                    <span>{sessionTitle || t('mobileChat.newChat')}</span>
                    {currentSessionId && <EditOutlined className="x-edit-icon" />}
                  </div>
                )}
              </div>
              <Button 
                icon={<MenuOutlined />} 
                type="text" 
                onClick={() => setShowMenu(!showMenu)}
                className="x-menu-button"
                size="middle"
              />
            </div>
            
            {/* 菜单弹出层 */}
            {showMenu && (
              <div className="x-menu-dropdown">
                <div className="x-menu-items">
                  <div className="x-menu-item" onClick={handleNewChat}>
                    <PlusOutlined /> {t('mobileChat.newConversation')}
                  </div>
                  <div className="x-menu-item" onClick={showSessionHistory}>
                    <HistoryOutlined /> {t('mobileChat.sessionHistory')}
                  </div>
                </div>
                <div className="x-menu-overlay" onClick={() => setShowMenu(false)} />
              </div>
            )}
          </>
        )}
      </div>

      {/* 主内容区域 */}
      <div className="x-chat-body">
        {showSessions ? (
          /* 会话列表 */
          <div className="x-sessions-list" ref={chatContainerRef}>
            {loading && <div className="x-loading"><Spin indicator={<LoadingOutlined style={{ fontSize: 24 }} spin />} /></div>}
            
            {sessions.length > 0 ? (
              <div className="x-sessions-items">
                {sessions.map(session => (
                  <div 
                    key={session.id}
                    className={`x-session-item ${session.id === currentSessionId ? 'active' : ''}`}
                    onClick={() => loadSession(session.id)}
                  >
                    <div className="x-session-content">
                      <div className="x-session-info">
                        <div className="x-session-name">
                          {session.is_ai_only ? <RobotOutlined className="x-session-icon" /> : <FileTextOutlined className="x-session-icon" />}
                          <span>{session.title || t('mobileChat.unnamedSession')}</span>
                        </div>
                        <div className="x-session-meta">
                          <span className="x-session-time">
                            {new Date(session.updated_at).toLocaleString('zh-CN', {
                              month: 'numeric',
                              day: 'numeric',
                              hour: '2-digit',
                              minute: '2-digit'
                            })}
                          </span>
                        </div>
                      </div>
                      <div className="x-session-preview">{session.last_message || t('mobileChat.emptySession')}</div>
                      <Button
                        type="text"
                        danger
                        icon={<DeleteOutlined />}
                        size="small"
                        onClick={(e) => handleDeleteSession(session.id, e)}
                        className="x-delete-button"
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              !loading && (
                <div className="x-empty-sessions">
                  <div className="x-empty-icon"><RobotOutlined /></div>
                  <div className="x-empty-text">{t('mobileChat.noSessions')}</div>
                  <Button 
                    type="primary" 
                    onClick={handleNewChat}
                    className="x-create-button"
                  >
                    {t('mobileChat.createNewSession')}
                  </Button>
                </div>
              )
            )}
          </div>
        ) : (
          /* 聊天消息区 */
          <div 
            className="x-messages-container" 
            ref={chatContainerRef}
            onScroll={handleChatScroll}
          >
            {isLoadingMore && (
              <div className="x-loading-more">
                <Spin size="small" /> {t('mobileChat.loadingMore')}
              </div>
            )}
            
            {messages.length === 0 && !loading && (
              <div className="x-welcome">
                <div className="x-welcome-icon"><RobotOutlined /></div>
                <h3>{t('mobileChat.title')}</h3>
                <p>{t('mobileChat.askAnything')}</p>
              </div>
            )}
            
            <div className="x-messages-list">
              {messages.map((msg) => (
                <div 
                  key={`msg-${msg.id || Date.now()}`} 
                  className={`x-message ${msg.role === 'user' ? 'x-user' : 'x-assistant'}`}
                >
                  <div className="x-message-avatar">
                    {msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
                  </div>
                  <div className="x-message-content">
                    <div className="x-message-bubble">
                      <div className="x-message-text">
                        <ReactMarkdown>
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    </div>
                    {/* 复制按钮 - 只在AI回复时显示，放在消息框外部底部 */}
                    {msg.role === 'assistant' && (
                      <div className="x-copy-button-container">
                        <Tooltip title={copyingMessageId === msg.id ? t('mobileChat.copied') : t('mobileChat.copyMessage')}>
                          <button 
                            className="x-copy-button" 
                            onClick={() => handleCopyMessage(msg.id || '', msg.content)}
                            disabled={copyingMessageId === msg.id}
                          >
                            {copyingMessageId === msg.id ? <CheckOutlined /> : <CopyOutlined />}
                          </button>
                        </Tooltip>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {/* 流式响应消息 */}
              {streamingState.isStreaming && (
                <div className="x-message x-assistant">
                  <div className="x-message-avatar">
                    <RobotOutlined />
                  </div>
                  <div className="x-message-bubble x-streaming">
                    <div className="x-message-text">
                      <ReactMarkdown>
                        {streamingState.partialMessage || '...'}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              )}
              
              {uploading && (
                <div className="x-thinking">
                  <Spin size="small" /> <span>{t('mobileChat.uploading')}</span>
                </div>
              )}
              
              {sending && !streamingState.isStreaming && !uploading && (
                <div className="x-thinking">
                  <Spin size="small" /> <span>{t('mobileChat.thinking')}</span>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          </div>
        )}
      </div>

      {/* 底部输入区域 - 只在聊天视图时显示 */}
      {!showSessions && (
        <div className={`x-chat-footer ${inputFocused ? 'x-input-focused' : ''}`}>
          {/* 文件选择显示区域 */}
          {selectedFile && (
            <div className="x-selected-file">
              <div className="x-file-info">
                <FileOutlined />
                <span className="x-file-name">{selectedFile.name}</span>
                <span className="x-file-size">
                  ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                </span>
              </div>
              <Button
                type="text"
                icon={<CloseOutlined />}
                size="small"
                onClick={() => setSelectedFile(null)}
                className="x-remove-file"
              />
            </div>
          )}
          
          <div className="x-input-container">
            <div className="x-input-wrapper">
              <TextArea
                value={messageText}
                onChange={(e) => {
                  setMessageText(e.target.value);
                  // 输入时保持滚动位置
                  if (keyboardOpen && inputFocused) {
                    setTimeout(() => smartScroll(), 50);
                  }
                }}
                placeholder={selectedFile ? t('mobileChat.inputPlaceholderWithFile') : t('mobileChat.inputPlaceholder')}
                autoSize={{ minRows: 1, maxRows: 4 }}
                onCompositionStart={() => setIsComposing(true)}
                onCompositionEnd={() => setIsComposing(false)}
                onFocus={() => {
                  setInputFocused(true);
                  // 延迟滚动，等待键盘完全弹出和布局调整完成
                  setTimeout(() => {
                    smartScroll();
                  }, 400);
                }}
                onBlur={() => {
                  setInputFocused(false);
                  // 延迟重置，避免快速切换时的闪烁
                  setTimeout(() => {
                    if (!document.activeElement?.classList.contains('x-input')) {
                      setKeyboardOpen(false);
                    }
                  }, 100);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !isComposing && !sending && !uploading) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                disabled={sending || uploading}
                className="x-input"
                style={{
                  fontSize: '16px', // 确保字体大小足够大
                  WebkitAppearance: 'none', // 移除iOS默认样式
                }}
              />
              
              {/* 文件上传按钮 */}
              <Upload
                accept=".pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xls,.txt,.md"
                maxCount={1}
                showUploadList={false}
                beforeUpload={(file) => {
                  // 检查文件大小 (限制为50MB)
                  const maxSize = 50 * 1024 * 1024; // 50MB
                  if (file.size > maxSize) {
                    message.error(t('mobileChat.fileSizeLimit'));
                    return false;
                  }
                  
                  // 检查文件类型
                  const allowedTypes = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.txt', '.md'];
                  const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
                  
                  if (!allowedTypes.includes(fileExtension)) {
                    message.error(t('mobileChat.unsupportedFileType'));
                    return false;
                  }
                  
                  setSelectedFile(file);
                  message.success(t('mobileChat.fileSelected', { filename: file.name }));
                  return false; // 阻止自动上传
                }}
                disabled={sending || uploading}
              >
                <Tooltip title={uploading ? t('mobileChat.uploading') : t('mobileChat.uploadFile')}>
                  <Button
                    type="text"
                    icon={<PaperClipOutlined />}
                    disabled={sending || uploading}
                    className="x-upload-button"
                  />
                </Tooltip>
              </Upload>
            </div>
            
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={handleSendMessage}
              loading={sending || uploading}
              disabled={!messageText.trim() && !selectedFile}
              className="x-send-button"
            />
          </div>
        </div>
      )}

      {/* 删除确认弹窗 */}
      <Modal
        title={t('mobileChat.confirmDelete')}
        open={deleteModalVisible}
        onOk={confirmDeleteSession}
        onCancel={cancelDeleteSession}
        okText={t('mobileChat.delete')}
        cancelText={t('mobileChat.cancel')}
        okButtonProps={{ danger: true }}
        closeIcon={<CloseOutlined />}
        className="x-delete-modal"
      >
        <p>{t('mobileChat.deleteConfirmText')}</p>
      </Modal>

      {/* 全局加载状态 */}
      {loading && !showSessions && (
        <div className="x-global-loading">
          <Spin size="large" />
        </div>
      )}
    </div>
  );
};

export default MobileChat;