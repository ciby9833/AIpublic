import React, { useState, useEffect, useRef } from 'react';
import { List, message as antMessage, Tooltip } from 'antd';
import { ChatMessage as ChatMessageType, RichContentItem } from '../../../types/chat';
import { DownOutlined, UpOutlined, CopyOutlined, CheckOutlined, LeftOutlined, RightOutlined, UserOutlined, RobotOutlined } from '@ant-design/icons';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ReactMarkdown from 'react-markdown';
import mermaid from 'mermaid';
import { useTranslation } from 'react-i18next';
import './styles.css';

// Fix: Import katex for math formula rendering
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface ChatMessageProps {
  message: ChatMessageType;
}

// Helper function to detect code blocks when markdown is not used
const detectCodeBlock = (text: string): { isCode: boolean, language: string, content: string } => {
  const codeBlockRegex = /^```(\w+)?\s*\n([\s\S]*?)```$/;
  const match = text.match(codeBlockRegex);
  
  if (match) {
    return {
      isCode: true,
      language: match[1] || 'text',
      content: match[2]
    };
  }
  
  return { isCode: false, language: '', content: text };
};

// Helper for detecting and rendering tables from markdown-like syntax
const detectTable = (text: string): { isTable: boolean, rows: string[][] } => {
  // Simple table detection - rows separated by newlines, cells by |
  const rows = text.trim().split('\n');
  if (rows.length >= 2 && rows[0].includes('|') && rows[1].includes('|')) {
    // Check for separator row (like |---|---|)
    const separatorRegex = /^\s*\|[-:\s|]*\|\s*$/;
    if (rows.length >= 3 && separatorRegex.test(rows[1])) {
      // Parse rows into cells
      const tableRows = rows
        .filter((row, idx) => idx !== 1) // Skip separator row
        .map(row => 
          row.split('|')
             .filter(cell => cell.trim() !== '')
             .map(cell => cell.trim())
        );
      return { isTable: true, rows: tableRows };
    }
  }
  return { isTable: false, rows: [] };
};

// Improved: Better formula detection in preprocessMermaidContent
const preprocessMermaidContent = (content: string, t?: any): string => {
  if (!content) return '';
  
  // Remove any existing error messages that might be in the content
  const errorMessage = t ? t('chatMessage.chartRenderFailed') : '图表渲染失败';
  const cleanContent = content.replace(new RegExp(errorMessage + '\\s*', 'g'), '');
  
  // Check for diagram type
  if (cleanContent.includes('mindmap')) {
    try {
      // Split by lines and clean up
      const lines = cleanContent.split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);
      
      // Extract only the actual diagram content
      let diagramStarted = false;
      const diagramLines = [];
      
      for (const line of lines) {
        // Skip empty lines and comments
        if (!line || line.startsWith('%%')) continue;
        
        // Mark the start of the diagram definition
        if (line.startsWith('mindmap')) {
          diagramStarted = true;
          diagramLines.push(line);
          continue;
        }
        
        if (diagramStarted) {
          diagramLines.push(line);
        }
      }
      
      // If we found diagram content, process it
      if (diagramLines.length > 0) {
        // Extract just the mindmap definition line
        const mindmapLine = diagramLines.filter(line => line.startsWith('mindmap'))[0] || 'mindmap';
        
        // 改进1: 使用更精确的正则表达式检测根节点
        // 匹配类似 "  root((主题))" 或 "  root[[主题]]" 的格式
        const rootNodeDefined = diagramLines.some(line =>
          /^ {2,}[\w\s\u4e00-\u9fa5]+(\(\(.+\)\)|\[\[.+\]\])$/.test(line)
        );
        
        // If no root node is defined or has multiple roots, fix it
        if (!rootNodeDefined) {
          // Create a new array with proper structure
          const fixedLines = [mindmapLine, '  root((主题))'];
          
          // 改进2: 基于层级进行缩进调整
          diagramLines.forEach(line => {
            if (!line.startsWith('mindmap')) {
              // Check if this line already has indentation
              if (line.startsWith(' ') || line.startsWith('\t')) {
                // Already indented, ensure it's properly indented under root
                if (line.startsWith('  ') || line.startsWith('\t\t')) {
                  // This is likely a second level, keep as is
                  fixedLines.push(line);
                } else {
                  // First level, add more indentation to make it a child of root
                  fixedLines.push('    ' + line.trim());
                }
              } else {
                // No indentation, make it a child of root
                fixedLines.push('    ' + line);
              }
            }
          });
          
          return fixedLines.join('\n');
        }
        
        // Process math formulas and fix other minor syntax issues
        const processedLines = diagramLines.map(line => {
          // Handle the mindmap declaration line
          if (line.startsWith('mindmap')) {
            return line;
          }
          
          // 改进3: 处理非法特殊字符
          let processedLine = line.replace(/&/g, '和')
                                  .replace(/\([^)]*\)/g, (match) => `（${match}）`);
          
          // Process math formulas
          const mathRegex = /([∑∫∏∀∃Σ][^\s:]+|[a-zA-Z][0-9_]+=[^\s:]+|𝛾𝑡|𝑒𝑖|𝑟𝑁𝑡)/g;
          processedLine = processedLine.replace(mathRegex, (match) => {
            return `<span class="math">${match}</span>`;
          });
          
          // Fix quoted text issues
          if (processedLine.includes('"') && !processedLine.startsWith('"')) {
            const quotedContentRegex = /"([^"]+)"/g;
            const matches = Array.from(processedLine.matchAll(quotedContentRegex));
            
            if (matches.length > 0) {
              let fixedLine = processedLine;
              let prevEnd = 0;
              
              for (let i = 0; i < matches.length; i++) {
                const match = matches[i];
                const start = match.index;
                
                if (start && prevEnd > 0 && start === prevEnd) {
                  fixedLine = fixedLine.substring(0, start) + ' ' + fixedLine.substring(start);
                  prevEnd = start + match[0].length + 1;
                } else {
                  prevEnd = start + match[0].length;
                }
              }
              
              return fixedLine;
            }
          }
          
          return processedLine;
        });
        
        // 改进4: 返回完整的mermaid格式
        return processedLines.join('\n');
      }
    } catch (error) {
      console.error('Error preprocessing mermaid content:', error);
      return cleanContent; // Return cleaned content if preprocessing fails
    }
  } else if (cleanContent.includes('graph ') || cleanContent.includes('flowchart ')) {
    // Handle flowcharts - with special syntax fixing for semicolons
    try {
      const lines = cleanContent.split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);
      
      let diagramStarted = false;
      const diagramLines = [];
      
      for (const line of lines) {
        if (!line || line.startsWith('%%')) continue;
        
        if (line.startsWith('graph ') || line.startsWith('flowchart ')) {
          diagramStarted = true;
          diagramLines.push(line);
          continue;
        }
        
        if (diagramStarted) {
          // 改进3: 处理非法特殊字符 (flowchart版本)
          let processedLine = line.replace(/&/g, '和');
          
          // Fix specific flowchart syntax issues
          // 1. Fix lines with semicolons in the middle of connections
          if (processedLine.includes(';') && !processedLine.endsWith(';')) {
            // Split by semicolon and add each part as a separate line
            const parts = processedLine.split(';');
            for (const part of parts) {
              if (part.trim()) {
                diagramLines.push(part.trim());
              }
            }
          } else {
            // Add the line as is
            diagramLines.push(processedLine);
          }
        }
      }
      
      // Fix any lines ending with incomplete square brackets
      for (let i = 0; i < diagramLines.length; i++) {
        const line = diagramLines[i] as string;
        
        // Check for unbalanced brackets
        const openBrackets = (line.match(/\[/g) || []).length;
        const closeBrackets = (line.match(/\]/g) || []).length;
        
        if (openBrackets > closeBrackets && i < diagramLines.length - 1) {
          // If we have more open brackets than close brackets and there's a next line,
          // try to combine with the next line to complete the node
          diagramLines[i] = line + ' ' + diagramLines[i + 1];
          diagramLines.splice(i + 1, 1); // Remove the next line as we merged it
          i--; // Reprocess this line in case there are still unbalanced brackets
        }
      }
      
      return diagramLines.join('\n');
    } catch (error) {
      console.error('Error preprocessing flowchart content:', error);
      return cleanContent;
    }
  }
  
  return cleanContent;
};

const ChatMessage: React.FC<{ message: any }> = ({ message }) => {
  const { t } = useTranslation();
  const { role, content, reply, sources } = message;
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  
  const toggleSources = () => {
    setSourcesExpanded(!sourcesExpanded);
  };

  // 简单的复制功能
  const handleCopy = async () => {
    try {
      // 获取要复制的文本
      let textToCopy = '';
      if (Array.isArray(reply) && reply.length > 0) {
        textToCopy = reply.map(item => item.content || '').join('\n\n');
      } else {
        textToCopy = content || '';
      }
      
      if (textToCopy.trim()) {
        await navigator.clipboard.writeText(textToCopy);
        setCopySuccess(true);
        antMessage.success(t('chatMessage.copySuccess'));
        setTimeout(() => setCopySuccess(false), 2000);
      }
    } catch (error) {
      console.error(t('chatMessage.copyFailed'), error);
      antMessage.error(t('chatMessage.copyFailed'));
    }
  };
  
  return (
    <div className={`chat-message ${role === 'user' ? 'user' : ''}`}>
      <div className="message-avatar">
        {role === 'user' ? <UserOutlined /> : <RobotOutlined />}
      </div>
      <div className="message-content-wrapper">
        <div className="message-content">
          {/* Prioritize using the reply array content */}
          {Array.isArray(reply) && reply.length > 0 ? (
            reply.map((item, index) => {
              if (item.type === 'markdown') {
                return <ReactMarkdown key={index}>{item.content}</ReactMarkdown>;
              } else if (item.type === 'code') {
                return (
                  <pre key={index} className={`language-${item.language || 'text'}`}>
                    <code>{item.content}</code>
                  </pre>
                );
              } else if (item.type === 'table') {
                return <ReactMarkdown key={index}>{item.content}</ReactMarkdown>;
              } else {
                // Default text rendering
                return <div key={index}>{item.content}</div>;
              }
            })
          ) : (
            // Fallback to content field if reply array is empty
            <div>{content}</div>
          )}
          
          {/* Display sources if available - with expand/collapse functionality */}
          {Array.isArray(sources) && sources.length > 0 && (
            <div className="message-sources">
              <div className="sources-header" onClick={toggleSources}>
                <span className="sources-title">
                  {t('chatMessage.referenceSources')}: {t('chatMessage.sourcesCount', { count: sources.length })}
                </span>
                <span className="toggle-icon">
                  {sourcesExpanded ? <UpOutlined /> : <DownOutlined />}
                </span>
              </div>
              <div className={`sources-content ${sourcesExpanded ? 'expanded' : ''}`}>
                <ul className="sources-list">
                  {sources.map((source, idx) => (
                    <li key={idx} className="source-item">
                      {source.content ? (
                        <>
                          {source.document_name && <span className="source-document">{source.document_name}</span>}
                          <span className="source-content">
                            {source.content}
                          </span>
                        </>
                      ) : (
                        <span>{t('chatMessage.unknownSource')}</span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
        
        {/* 简单的复制按钮 */}
        <div className="copy-button-wrapper">
          <button 
            className="copy-btn" 
            onClick={handleCopy}
            title={copySuccess ? t('chatMessage.copied') : t('chatMessage.copyMessage')}
          >
            {copySuccess ? <CheckOutlined /> : <CopyOutlined />}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
