import React, { useState } from 'react';
import { List, message as antMessage } from 'antd';
import { ChatMessage as ChatMessageType, ChatResponse } from '../../../types/chat';
import { DownOutlined, UpOutlined, CopyOutlined, CheckOutlined } from '@ant-design/icons';
import './styles.css';

interface ChatMessageProps {
  message: ChatMessageType;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isStructuredAnswer = typeof message.answer === 'object';
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  // 修改复制功能，使用 antMessage
  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      antMessage.success('复制成功');
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {
      antMessage.error('复制失败');
    });
  };

  // 修复 getCopyText 函数，确保返回字符串
  const getCopyText = (): string => {
    if (isStructuredAnswer) {
      const chatResponse = message.answer as ChatResponse;
      return chatResponse.answer || '';
    }
    return typeof message.answer === 'string' ? message.answer : '';
  };

  // 添加新的格式化函数
  const formatLogContent = (content: string) => {
    // 检查是否包含 | 分隔符
    if (content.includes('|')) {
      const items = content.split('|').map(item => item.trim()).filter(Boolean);
      return (
        <div className="log-content">
          {items.map((item, index) => (
            <div key={index} className="log-item">
              {item}
            </div>
          ))}
        </div>
      );
    }
    return formatText(content);
  };

  // 修改原有的 formatText 函数
  const formatText = (text: string) => {
    return text.split('\n').map((line, index) => (
      <div key={index} className="answer-line">
        {line}
      </div>
    ));
  };

  return (
    <List.Item className="chat-message">
      <div className="message-bubble question-bubble">
        <div className="message-header">问</div>
        <div className="message-content">{message.question}</div>
      </div>
      <div className="message-bubble answer-bubble">
        <div className="message-header">
          <span>答</span>
          <button 
            className="copy-button"
            onClick={() => handleCopy(getCopyText())}
            title="复制回答"
          >
            {copied ? <CheckOutlined /> : <CopyOutlined />}
          </button>
        </div>
        <div className="message-content">
          {isStructuredAnswer ? (
            <>
              <div className="answer-text">
                {formatLogContent((message.answer as ChatResponse).answer)}
              </div>
              {(message.answer as ChatResponse).sources && (
                <div className="answer-sources">
                  <div 
                    className="sources-header"
                    onClick={() => setSourcesExpanded(!sourcesExpanded)}
                  >
                    <div className="sources-title">
                      <span>参考来源</span>
                      {(message.answer as ChatResponse).confidence && (
                        <span className="confidence">
                          可信度: {((message.answer as ChatResponse).confidence * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    <div className="expand-icon">
                      {sourcesExpanded ? <UpOutlined /> : <DownOutlined />}
                    </div>
                  </div>
                  <div className={`sources-content ${sourcesExpanded ? 'expanded' : ''}`}>
                    {(message.answer as ChatResponse).sources.map((source, index) => (
                      <div key={index} className="source-item">
                        <div className="source-header">
                          <span className="page-number">第{source.page}页</span>
                          <span className="line-number">第{source.line_number}行</span>
                          <span className="similarity">
                            相关度: {(source.similarity * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="source-content">
                          {formatLogContent(source.content)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="answer-text">
              {formatLogContent(message.answer as string)}
            </div>
          )}
        </div>
      </div>
    </List.Item>
  );
};

export default ChatMessage;
