import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Input, Button, Spin, Modal, Upload, List, message } from 'antd';
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
  LoadingOutlined
} from '@ant-design/icons';
import { aiChatService } from '../../services/aiChatService';
import { ApiChatMessage, ChatSession, StreamingState } from '../../types/chat';
import { MessagesResponse } from '../../services/aiChatService';
import ReactMarkdown from 'react-markdown';
import './mobileChat.css';

const { TextArea } = Input;

const MobileChat: React.FC = () => {
  const navigate = useNavigate();
  const [messageText, setMessageText] = useState<string>('');
  const [messages, setMessages] = useState<ApiChatMessage[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [sending, setSending] = useState<boolean>(false);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [showSessions, setShowSessions] = useState<boolean>(false);
  const [showMenu, setShowMenu] = useState<boolean>(false);
  const [sessionTitle, setSessionTitle] = useState<string>('新对话');
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
            setSessionTitle(latestSession.title || "未命名会话");
            
            // 自动滚动到最新消息
            setTimeout(() => scrollToBottom(0), 300);
          } catch (sessionError) {
            console.error("加载会话失败:", sessionError);
            message.error("加载最新会话失败，已创建新对话");
            setSessionTitle("新对话");
            setMessages([]);
            setCurrentSessionId("");
          }
        } else {
          setSessionTitle("新对话");
          setMessages([]);
          setCurrentSessionId("");
        }
      } catch (error) {
        console.error("自动加载会话失败:", error);
        message.error("加载会话列表失败");
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

  // 发送消息
  const handleSendMessage = async () => {
    if (!messageText.trim()) {
      return;
    }

    const userMessage = messageText;
    setMessageText('');
    setSending(true);

    try {
      // 如果没有会话ID，先创建会话
      let sessionId = currentSessionId;
      if (!sessionId) {
        try {
          const newSession = await aiChatService.createAiOnlySession();
          
          if (newSession && newSession.id) {
            sessionId = newSession.id;
            setCurrentSessionId(sessionId);
            setSessionTitle(newSession.title || '新对话');
          } else {
            throw new Error('创建会话失败');
          }
        } catch (error) {
          console.error("创建会话失败:", error);
          message.error('创建会话失败');
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
              console.error("获取消息失败:", err);
            }
            
            scrollToBottom();
            setSending(false);
          },
          onError: (error) => {
            const errorMessage = typeof error === 'string' ? error : 
                                error?.message || '未知错误';
            console.error("消息流处理失败:", error);
            message.error(`发送失败: ${errorMessage}`);
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
      message.error('发送失败，请重试');
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
      console.error('加载更多消息失败:', error);
      message.error('加载更多消息失败');
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
        message.error('获取对话历史格式错误');
      }
    } catch (error) {
      console.error('获取会话失败:', error);
      message.error('获取会话失败');
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
      console.error("获取会话历史失败:", error);
      message.error('获取会话历史失败');
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
        setSessionTitle(currentSession.title || '无标题会话');
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
      console.error('加载对话历史失败:', error);
      message.error('加载对话历史失败');
    } finally {
      setLoading(false);
    }
  };

  // 创建新会话
  const handleNewChat = () => {
    // 如果当前没有消息且没有会话ID，则不需要创建新会话
    if (messages.length === 0 && !currentSessionId) {
      console.log("当前已经是空白新会话，无需创建");
      setShowSessions(false);
      setShowMenu(false);
      return;
    }
    
    // 清除当前会话和消息
    setCurrentSessionId("");
    setMessages([]);
    setSessionTitle("新对话");
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
        setSessionTitle("新对话");
        setShowSessions(false);
      }
      
      message.success('会话已删除');
    } catch (error: any) {
      // 安全地提取错误消息
      const errorMessage = error?.message || '未知错误';
      message.error(`删除失败: ${errorMessage}`);
      console.error('删除会话失败:', error);
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
        message.success('会话标题已更新');
        
        // 更新会话列表中的标题
        setSessions(sessions.map(session => 
          session.id === currentSessionId 
            ? { ...session, title: sessionTitle } 
            : session
        ));
      } else {
        message.success('标题已更新');
      }
      
      setEditingTitle(false);
    } catch (error: any) {
      // 使用安全的错误消息提取方式
      const errorMessage = error?.message || '未知错误';
      message.error(`保存标题失败: ${errorMessage}`);
      console.error('保存标题失败:', error);
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

  return (
    <div className="x-chat-container">
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
            <div className="x-header-title">会话历史</div>
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
                    <span>{sessionTitle || "新对话"}</span>
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
                    <PlusOutlined /> 新建对话
                  </div>
                  <div className="x-menu-item" onClick={showSessionHistory}>
                    <HistoryOutlined /> 历史会话
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
                          <span>{session.title || "未命名会话"}</span>
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
                      <div className="x-session-preview">{session.last_message || "空会话"}</div>
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
                  <div className="x-empty-text">暂无会话历史</div>
                  <Button 
                    type="primary" 
                    onClick={handleNewChat}
                    className="x-create-button"
                  >
                    创建新会话
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
                <Spin size="small" /> 加载更多消息...
              </div>
            )}
            
            {messages.length === 0 && !loading && (
              <div className="x-welcome">
                <div className="x-welcome-icon"><RobotOutlined /></div>
                <h3>Cargo AI助手</h3>
                <p>有任何问题都可以向我提问</p>
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
                  <div className="x-message-bubble">
                    <div className="x-message-text">
                      <ReactMarkdown>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
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
              
              {sending && !streamingState.isStreaming && (
                <div className="x-thinking">
                  <Spin size="small" /> <span>AI正在思考...</span>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          </div>
        )}
      </div>

      {/* 底部输入区域 - 只在聊天视图时显示 */}
      {!showSessions && (
        <div className="x-chat-footer">
          <div className="x-input-container">
            <TextArea
              value={messageText}
              onChange={(e) => setMessageText(e.target.value)}
              placeholder="输入消息..."
              autoSize={{ minRows: 1, maxRows: 4 }}
              onPressEnter={(e) => {
                if (!e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              disabled={sending}
              className="x-input"
            />
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={handleSendMessage}
              loading={sending}
              disabled={!messageText.trim()}
              className="x-send-button"
            />
          </div>
        </div>
      )}

      {/* 删除确认弹窗 */}
      <Modal
        title="确认删除"
        open={deleteModalVisible}
        onOk={confirmDeleteSession}
        onCancel={cancelDeleteSession}
        okText="删除"
        cancelText="取消"
        okButtonProps={{ danger: true }}
        closeIcon={<CloseOutlined />}
        className="x-delete-modal"
      >
        <p>确定要删除这个会话吗？此操作不可恢复。</p>
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