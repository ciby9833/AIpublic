import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Input, Button, Spin, Avatar, Modal, Upload, List, Dropdown, message } from 'antd';
import { 
  SendOutlined, 
  PlusOutlined, 
  MenuOutlined, 
  FileTextOutlined,
  UserOutlined,
  RobotOutlined,
  LeftOutlined,
  PaperClipOutlined,
  HistoryOutlined,
  DeleteOutlined,
  EditOutlined
} from '@ant-design/icons';
import { aiChatService } from '../../services/aiChatService';
import { ChatMessage, ChatSession } from '../../types/chat';
import ChatMessageComponent from '../PaperAnalyzer/components';
import './mobileChat.css';
import ReactMarkdown from 'react-markdown';

const { TextArea } = Input;

const MobileChat: React.FC = () => {
  const navigate = useNavigate();
  const [message, setMessage] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [sending, setSending] = useState<boolean>(false);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [showSessions, setShowSessions] = useState<boolean>(false);
  const [showMenu, setShowMenu] = useState<boolean>(false);
  const [sessionTitle, setSessionTitle] = useState<string>('新对话');
  const [editingTitle, setEditingTitle] = useState<boolean>(false);
  const [uploadVisible, setUploadVisible] = useState<boolean>(false);
  const [hasMoreMessages, setHasMoreMessages] = useState<boolean>(false);
  const [lastMessageId, setLastMessageId] = useState<string | null>(null);
  const [isLoadingMore, setIsLoadingMore] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const [streamingState, setStreamingState] = useState<{
    isStreaming: boolean;
    partialMessage: string;
    messageId?: string;
  }>({
    isStreaming: false,
    partialMessage: ''
  });

  // 检查是否是移动设备
  useEffect(() => {
    const isMobile = window.innerWidth <= 768;
    if (!isMobile) {
      // 如果不是移动设备，重定向到桌面版
      navigate('/');
    }
  }, [navigate]);

  // 页面加载时获取最新的会话
  useEffect(() => {
    const loadLatestSession = async () => {
      try {
        setLoading(true);
        const allSessions = await aiChatService.getAllSessions();
        
        if (allSessions.length > 0) {
          // 找到创建时间最新的活跃会话
          const latestSession = allSessions[0];
          setSessions(allSessions);
          
          try {
            // 尝试加载最新会话
            const result = await aiChatService.getMessages(latestSession.id, 20);
            setMessages(result.messages || []);
            setHasMoreMessages(result.has_more || false);
            setLastMessageId(result.messages.length > 0 ? result.messages[0].id : null);
            
            setCurrentSessionId(latestSession.id);
            setSessionTitle(latestSession.title || "未命名会话");
            console.log("自动加载最新会话:", latestSession.title);
          } catch (sessionError) {
            console.error("加载会话失败:", sessionError);
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
    if (!message.trim()) {
      return;
    }

    const messageText = message;
    setMessage('');
    setSending(true);

    try {
      // 如果没有会话ID，先创建会话
      let sessionId = currentSessionId;
      if (!sessionId) {
        try {
          const newSession = await aiChatService.createAiOnlySession();
          if (!newSession || !newSession.id) {
            throw new Error('创建会话失败');
          }
          sessionId = newSession.id;
          setCurrentSessionId(sessionId);
          setSessionTitle(newSession.title || '新对话');
        } catch (error) {
          console.error("创建会话失败:", error);
          message.error('创建会话失败');
          setSending(false);
          return;
        }
      }

      // 修改临时用户消息格式，统一为API的返回格式
      const tempUserMessage = {
        id: 'temp-' + Date.now(),
        role: 'user',
        content: messageText,
        created_at: new Date().toISOString(),
        sources: [],
        confidence: 0,
        reply: [{
          type: 'markdown',
          content: messageText
        }]
      };
      
      // 添加用户消息到UI
      setMessages(prev => [...prev, tempUserMessage]);
      
      // 开始流式状态
      setStreamingState({
        isStreaming: true,
        partialMessage: ''
      });
      
      // 使用流式API
      await aiChatService.streamMessage(
        sessionId,
        messageText,
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
            // 重置流式状态
            setStreamingState({
              isStreaming: false,
              partialMessage: ''
            });
            
            // 获取完整的消息列表
            const updatedMessages = await aiChatService.getMessages(sessionId, 20);
            if (updatedMessages && updatedMessages.messages) {
              setMessages(updatedMessages.messages);
              setHasMoreMessages(updatedMessages.has_more || false);
              if (updatedMessages.messages.length > 0) {
                setLastMessageId(updatedMessages.messages[0].id || null);
              }
            }
            
            scrollToBottom();
            setSending(false);
          },
          onError: (error) => {
            console.error("消息流处理失败:", error);
            message.error('发送失败，请重试');
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
      );
      
      if (result && result.messages && result.messages.length > 0) {
        // 将旧消息添加到消息数组开头
        setMessages(prev => [...result.messages, ...prev]);
        
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
      setSessions(sessionList);
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
      const result = await aiChatService.getMessages(sessionId, 20);
      
      // 更新当前会话信息
      setCurrentSessionId(sessionId);
      
      // 更新sessionTitle
      const currentSession = sessions.find(s => s.id === sessionId);
      if (currentSession) {
        setSessionTitle(currentSession.title || '无标题会话');
      }
      
      // 设置消息内容
      if (result && result.messages) {
        setMessages(result.messages);
        setHasMoreMessages(result.has_more || false);
        if (result.messages.length > 0) {
          setLastMessageId(result.messages[0].id || null);
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
  const handleNewChat = async () => {
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

  // 添加执行删除的函数
  const confirmDeleteSession = async () => {
    if (!sessionToDelete) return;
    
    try {
      setLoading(true);
      await aiChatService.deleteSession(sessionToDelete);
      
      // 删除成功后，更新会话列表
      setSessions(sessions.filter(session => session.id !== sessionToDelete));
      
      // 如果删除的是当前会话，则清空当前会话
      if (sessionToDelete === currentSessionId) {
        setCurrentSessionId("");
        setMessages([]);
        setSessionTitle("新对话");
        setShowSessions(false);
      }
      
      // 如果删除后没有会话了，也关闭会话列表
      if (sessions.length <= 1) {
        setShowSessions(false);
      }
      
      message.success('会话已删除');
    } catch (error: any) {
      message.error(`删除失败: ${error.message}`);
      console.error('删除会话失败:', error);
    } finally {
      setDeleteModalVisible(false);
      setSessionToDelete(null);
      setLoading(false);
    }
  };

  // 添加取消删除的函数
  const cancelDeleteSession = () => {
    setDeleteModalVisible(false);
    setSessionToDelete(null);
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
      
      // 更新会话列表中的标题
      if (sessions.length > 0) {
        setSessions(sessions.map(session => 
          session.id === currentSessionId 
            ? { ...session, title: sessionTitle } 
            : session
        ));
      }
      
      setEditingTitle(false);
      message.success('标题已更新');
    } catch (error: any) {
      message.error(`保存标题失败: ${error.message}`);
      console.error('保存标题失败:', error);
      setEditingTitle(false);
    }
  };

  // 处理文件上传
  const handleFileUpload = async (file: File) => {
    try {
      setUploadVisible(false);
      setLoading(true);

      // 1. 上传文档
      const result = await aiChatService.uploadDocument(file, currentSessionId);
      
      if (result.status === 'success' && result.paper_id) {
        message.success('文档上传成功');
        
        // 如果没有当前会话，创建一个新会话
        if (!currentSessionId) {
          const newSession = await aiChatService.createDocumentSession(
            [result.paper_id],
            `关于 ${file.name} 的对话`
          );
          
          setCurrentSessionId(newSession.id);
          setSessionTitle(newSession.title || `关于 ${file.name} 的对话`);
          
          // 获取会话消息
          const messagesResult = await aiChatService.getMessages(newSession.id, 20);
          setMessages(messagesResult.messages || []);
          setHasMoreMessages(messagesResult.has_more || false);
        } else {
          // 将文档添加到当前会话
          await aiChatService.addDocumentToSession(currentSessionId, result.paper_id);
          message.success('文档已添加到当前会话');
          
          // 更新消息列表
          const messagesResult = await aiChatService.getMessages(currentSessionId, 20);
          setMessages(messagesResult.messages || []);
          setHasMoreMessages(messagesResult.has_more || false);
        }
      } else {
        message.error(result.message || '文档处理失败');
      }
    } catch (error: any) {
      console.error('文档上传失败:', error);
      message.error(`上传失败: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 主菜单项
  const menuItems = [
    {
      key: 'new',
      label: '新建对话',
      icon: <PlusOutlined />,
      onClick: handleNewChat
    },
    {
      key: 'history',
      label: '查看历史会话',
      icon: <HistoryOutlined />,
      onClick: showSessionHistory
    },
    {
      key: 'upload',
      label: '上传文档',
      icon: <FileTextOutlined />,
      onClick: () => setUploadVisible(true)
    }
  ];

  return (
    <div className="mobile-chat-container">
      {/* 头部导航 */}
      <div className="mobile-chat-header">
        {showSessions ? (
          <div className="session-header">
            <Button 
              icon={<LeftOutlined />}
              type="link"
              onClick={() => setShowSessions(false)}
              className="back-button"
            />
            <div className="header-title">会话历史</div>
          </div>
        ) : (
          <>
            <div className="header-title" onClick={() => setEditingTitle(true)}>
              {editingTitle ? (
                <div className="edit-title">
                  <Input
                    value={sessionTitle}
                    onChange={(e) => setSessionTitle(e.target.value)}
                    onPressEnter={handleTitleSave}
                    onBlur={handleTitleSave}
                    size="small"
                    autoFocus
                  />
                </div>
              ) : (
                <span>{sessionTitle || "新对话"}</span>
              )}
            </div>
            <Button 
              icon={<MenuOutlined />} 
              type="link" 
              onClick={() => setShowMenu(!showMenu)}
              className="menu-button"
            />
          </>
        )}
      </div>

      {/* 下拉菜单 */}
      {showMenu && (
        <div className="mobile-dropdown-menu">
          <List
            dataSource={menuItems}
            renderItem={(item) => (
              <List.Item 
                onClick={item.onClick}
                className="menu-item"
              >
                {item.icon} {item.label}
              </List.Item>
            )}
          />
        </div>
      )}

      {/* 会话列表 */}
      {showSessions ? (
        <div className="mobile-sessions-list" ref={chatContainerRef}>
          {loading && (
            <div className="loading-container">
              <Spin size="large" />
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
                      <Button
                        type="text"
                        danger
                        icon={<DeleteOutlined />}
                        size="small"
                        onClick={(e) => handleDeleteSession(session.id, e)}
                        className="delete-session-btn"
                      />
                    </div>
                    <div className="session-preview">{session.last_message}</div>
                  </div>
                </List.Item>
              )}
            />
          ) : (
            !loading && (
              <div className="empty-state">
                <div>暂无对话历史</div>
                <Button 
                  type="primary" 
                  onClick={handleNewChat}
                >
                  创建新对话
                </Button>
              </div>
            )
          )}
        </div>
      ) : (
        // 聊天消息区域
        <div 
          className="mobile-chat-messages" 
          ref={chatContainerRef}
          onScroll={handleChatScroll}
        >
          {isLoadingMore && (
            <div className="loading-more">
              <Spin size="small" /> 加载更多...
            </div>
          )}
          
          {messages.length === 0 && !loading && (
            <div className="welcome-message">
              <div className="welcome-icon">
                <RobotOutlined />
              </div>
              <h3>AI助手</h3>
              <p>有任何问题都可以向我提问</p>
            </div>
          )}
          
          {messages.map((msg, index) => (
            <ChatMessageComponent 
              key={msg.id || index} 
              message={{
                ...msg,
                question: msg.content,
                answer: msg.role === 'assistant' ? msg.content : ''
              }} 
            />
          ))}
          
          {/* 流式响应消息 */}
          {streamingState.isStreaming && (
            <div className="chat-message assistant">
              <div className="message-avatar">
                <RobotOutlined />
              </div>
              <div className="message-content">
                <ReactMarkdown>
                  {streamingState.partialMessage || '...'}
                </ReactMarkdown>
              </div>
            </div>
          )}
          
          {sending && (
            <div className="thinking-indicator-container">
              <Spin size="small" tip="AI正在思考..." />
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      )}

      {/* 消息输入框 */}
      {!showSessions && (
        <div className="mobile-chat-input">
          <div className="input-wrapper">
            <Button
              type="text"
              icon={<PaperClipOutlined />}
              onClick={() => setUploadVisible(true)}
              className="upload-button"
            />
            <TextArea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onPressEnter={(e) => {
                if (!e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              placeholder="输入消息..."
              autoSize={{ minRows: 1, maxRows: 4 }}
              disabled={sending}
            />
            <Button
              type="primary"
              shape="circle"
              icon={<SendOutlined />}
              onClick={handleSendMessage}
              loading={sending}
              disabled={!message.trim()}
              className="send-button"
            />
          </div>
        </div>
      )}

      {/* 文件上传弹窗 */}
      <Modal
        title="上传文档"
        open={uploadVisible}
        onCancel={() => setUploadVisible(false)}
        footer={null}
        centered
      >
        <Upload.Dragger
          accept=".pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xls,.txt,.md"
          beforeUpload={(file) => {
            handleFileUpload(file);
            return false;
          }}
          showUploadList={false}
        >
          <p className="ant-upload-drag-icon">
            <FileTextOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
          <p className="ant-upload-hint">
            支持 PDF、Word、Excel、PowerPoint、TXT 文件
          </p>
        </Upload.Dragger>
      </Modal>

      {/* 全局加载状态 */}
      {loading && (
        <div className="global-loading">
          <Spin size="large" />
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
        centered
      >
        <p>确定要删除这个对话吗？此操作不可恢复。</p>
      </Modal>
    </div>
  );
};

export default MobileChat;
