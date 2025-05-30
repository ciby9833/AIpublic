import React, { useState, useEffect, useRef } from 'react';
import { List, message as antMessage, Tooltip } from 'antd';
import { ApiChatMessage as ChatMessageType, RichContentItem } from '../../../types/chat';
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
  isStreaming?: boolean; // 新增：是否为流式消息
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

// 智能内容解析器，参考ChatGPT实现
const SmartContentParser = ({ content, isStreaming = false }: { content: string; isStreaming?: boolean }) => {
  const [parsedContent, setParsedContent] = useState<Array<{
    type: 'text' | 'code' | 'markdown' | 'table';
    content: string;
    language?: string;
  }>>([]);

  useEffect(() => {
    const parseContent = (text: string) => {
      const blocks: Array<{
        type: 'text' | 'code' | 'markdown' | 'table';
        content: string;
        language?: string;
      }> = [];

      // 分割代码块
      const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
      let lastIndex = 0;
      let match;

      while ((match = codeBlockRegex.exec(text)) !== null) {
        // 添加代码块前的文本
        if (match.index > lastIndex) {
          const beforeText = text.slice(lastIndex, match.index);
          if (beforeText.trim()) {
            blocks.push({
              type: isMarkdownContent(beforeText) ? 'markdown' : 'text',
              content: beforeText
            });
          }
        }

        // 添加代码块
        blocks.push({
          type: 'code',
          content: match[2],
          language: match[1] || 'text'
        });

        lastIndex = match.index + match[0].length;
      }

      // 添加剩余文本
      if (lastIndex < text.length) {
        const remainingText = text.slice(lastIndex);
        if (remainingText.trim()) {
          blocks.push({
            type: isMarkdownContent(remainingText) ? 'markdown' : 'text',
            content: remainingText
          });
        }
      }

      // 如果没有代码块，检查整个内容
      if (blocks.length === 0) {
        blocks.push({
          type: isMarkdownContent(text) ? 'markdown' : 'text',
          content: text
        });
      }

      return blocks;
    };

    const isMarkdownContent = (text: string): boolean => {
      // 检测Markdown特征
      return /^[#*\-+>]|\[.*\]\(.*\)|^\d+\.|^\s*[-*+]\s|\|.*\|/.test(text.trim()) ||
             text.includes('**') || text.includes('*') || text.includes('`');
    };

    setParsedContent(parseContent(content));
  }, [content, isStreaming]);

  return (
    <div className="smart-content">
      {parsedContent.map((block, index) => {
        switch (block.type) {
          case 'code':
            return (
              <SyntaxHighlighter
                key={index}
                language={block.language}
                style={vscDarkPlus}
                customStyle={{
                  margin: '8px 0',
                  borderRadius: '6px',
                  fontSize: '14px'
                }}
              >
                {block.content}
              </SyntaxHighlighter>
            );
          case 'markdown':
            return (
              <ReactMarkdown
                key={index}
                components={{
                  code: ({ className, children, ...props }: any) => {
                    const match = /language-(\w+)/.exec(className || '');
                    const isInline = !match;
                    return !isInline ? (
                      <SyntaxHighlighter
                        style={vscDarkPlus as any}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  }
                }}
              >
                {block.content}
              </ReactMarkdown>
            );
          default:
            return (
              <div key={index} className="text-content">
                {block.content}
              </div>
            );
        }
      })}
      {isStreaming && (
        <span className="streaming-cursor"></span>
      )}
    </div>
  );
};

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isStreaming = false }) => {
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

  // ✅ 新增：专门处理reply字段中的富文本内容
  const renderReplyContent = (replyItems: RichContentItem[]) => {
    return replyItems.map((item, index) => {
      const { type, content: itemContent, language, columns, rows } = item;
      
      switch (type) {
        case 'code':
          // 直接渲染代码块，不再经过SmartContentParser
          return (
            <SyntaxHighlighter
              key={index}
              language={language || 'text'}
              style={vscDarkPlus}
              customStyle={{
                margin: '8px 0',
                borderRadius: '6px',
                fontSize: '14px'
              }}
            >
              {itemContent || ''}
            </SyntaxHighlighter>
          );

        case 'table':
          // ✅ 新增：表格渲染支持
          if (columns && rows && Array.isArray(columns) && Array.isArray(rows)) {
            // 使用结构化的表格数据
            return (
              <div key={index} className="table-container" style={{ margin: '12px 0', overflowX: 'auto' }}>
                <table className="message-table" style={{ 
                  borderCollapse: 'collapse', 
                  width: '100%',
                  fontSize: '14px',
                  border: '1px solid #ddd'
                }}>
                  <thead>
                    <tr style={{ backgroundColor: '#f5f5f5' }}>
                      {columns.map((col: string, colIndex: number) => (
                        <th key={colIndex} style={{ 
                          border: '1px solid #ddd', 
                          padding: '8px 12px',
                          textAlign: 'left',
                          fontWeight: '600'
                        }}>
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row: string[], rowIndex: number) => (
                      <tr key={rowIndex} style={{ 
                        backgroundColor: rowIndex % 2 === 0 ? '#fff' : '#fafafa' 
                      }}>
                        {row.map((cell: string, cellIndex: number) => (
                          <td key={cellIndex} style={{ 
                            border: '1px solid #ddd', 
                            padding: '8px 12px',
                            textAlign: 'left'
                          }}>
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          } else if (itemContent) {
            // 如果没有结构化数据，尝试从content解析Markdown表格
            return (
              <ReactMarkdown
                key={index}
                components={{
                  table: ({ children, ...props }) => (
                    <div className="table-container" style={{ margin: '12px 0', overflowX: 'auto' }}>
                      <table className="message-table" style={{ 
                        borderCollapse: 'collapse', 
                        width: '100%',
                        fontSize: '14px',
                        border: '1px solid #ddd'
                      }} {...props}>
                        {children}
                      </table>
                    </div>
                  ),
                  th: ({ children, ...props }) => (
                    <th style={{ 
                      border: '1px solid #ddd', 
                      padding: '8px 12px',
                      backgroundColor: '#f5f5f5',
                      textAlign: 'left',
                      fontWeight: '600'
                    }} {...props}>
                      {children}
                    </th>
                  ),
                  td: ({ children, ...props }) => (
                    <td style={{ 
                      border: '1px solid #ddd', 
                      padding: '8px 12px',
                      textAlign: 'left'
                    }} {...props}>
                      {children}
                    </td>
                  )
                }}
              >
                {itemContent}
              </ReactMarkdown>
            );
          }
          return null;

        case 'list':
        case 'ordered-list':
        case 'unordered-list':
          // ✅ 新增：列表渲染支持
          return (
            <ReactMarkdown key={index}>
              {itemContent || ''}
            </ReactMarkdown>
          );

        case 'image':
          // ✅ 新增：图片渲染支持
          if (itemContent) {
            try {
              // 尝试解析图片URL或base64数据
              const imageUrl = itemContent.startsWith('data:') ? itemContent : 
                             itemContent.startsWith('http') ? itemContent : 
                             `data:image/png;base64,${itemContent}`;
              
              return (
                <div key={index} style={{ margin: '12px 0', textAlign: 'center' }}>
                  <img 
                    src={imageUrl} 
                    alt="Generated content" 
                    style={{ 
                      maxWidth: '100%', 
                      height: 'auto',
                      borderRadius: '8px',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                    }}
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.style.display = 'none';
                    }}
                  />
                </div>
              );
            } catch (error) {
              return (
                <div key={index} style={{ margin: '8px 0', color: '#999', fontStyle: 'italic' }}>
                  [图片加载失败]
                </div>
              );
            }
          }
          return null;

        case 'mermaid':
        case 'diagram':
          // ✅ 新增：Mermaid图表渲染支持
          return (
            <div key={index} className="mermaid-container" style={{ 
              margin: '16px 0', 
              padding: '12px', 
              backgroundColor: '#f9f9f9', 
              borderRadius: '8px',
              overflow: 'hidden'
            }}>
              <div className="mermaid">
                {itemContent}
              </div>
            </div>
          );

        case 'latex':
        case 'math':
          // ✅ 新增：数学公式渲染支持
          try {
            return (
              <div key={index} style={{ margin: '12px 0', textAlign: 'center' }}>
                <div 
                  dangerouslySetInnerHTML={{ 
                    __html: katex.renderToString(itemContent || '', { 
                      throwOnError: false,
                      displayMode: true 
                    }) 
                  }} 
                />
              </div>
            );
          } catch (error) {
            return (
              <div key={index} style={{ 
                margin: '8px 0', 
                padding: '8px', 
                backgroundColor: '#fff2f0',
                border: '1px solid #ffccc7',
                borderRadius: '4px',
                color: '#f5222d',
                fontSize: '13px'
              }}>
                数学公式渲染失败: {itemContent}
              </div>
            );
          }

        case 'html':
          // ✅ 新增：HTML内容渲染支持（谨慎处理安全性）
          return (
            <div 
              key={index} 
              style={{ margin: '8px 0' }}
              dangerouslySetInnerHTML={{ __html: itemContent || '' }}
            />
          );

        case 'json':
          // ✅ 新增：JSON数据渲染支持
          try {
            const formattedJson = JSON.stringify(JSON.parse(itemContent || '{}'), null, 2);
            return (
              <SyntaxHighlighter
                key={index}
                language="json"
                style={vscDarkPlus}
                customStyle={{
                  margin: '8px 0',
                  borderRadius: '6px',
                  fontSize: '14px'
                }}
              >
                {formattedJson}
              </SyntaxHighlighter>
            );
          } catch (error) {
            return (
              <pre key={index} style={{ 
                margin: '8px 0', 
                padding: '12px',
                backgroundColor: '#f5f5f5',
                borderRadius: '6px',
                overflow: 'auto',
                fontSize: '13px',
                fontFamily: 'monospace'
              }}>
                {itemContent}
              </pre>
            );
          }

        case 'markdown':
          // 对于markdown，使用ReactMarkdown渲染
          return (
            <ReactMarkdown
              key={index}
              components={{
                code: ({ className, children, ...props }: any) => {
                  const match = /language-(\w+)/.exec(className || '');
                  const isInline = !match;
                  return !isInline ? (
                    <SyntaxHighlighter
                      style={vscDarkPlus as any}
                      language={match[1]}
                      PreTag="div"
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  ) : (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                },
                table: ({ children, ...props }) => (
                  <div className="table-container" style={{ margin: '12px 0', overflowX: 'auto' }}>
                    <table className="message-table" style={{ 
                      borderCollapse: 'collapse', 
                      width: '100%',
                      fontSize: '14px',
                      border: '1px solid #ddd'
                    }} {...props}>
                      {children}
                    </table>
                  </div>
                ),
                th: ({ children, ...props }) => (
                  <th style={{ 
                    border: '1px solid #ddd', 
                    padding: '8px 12px',
                    backgroundColor: '#f5f5f5',
                    textAlign: 'left',
                    fontWeight: '600'
                  }} {...props}>
                    {children}
                  </th>
                ),
                td: ({ children, ...props }) => (
                  <td style={{ 
                    border: '1px solid #ddd', 
                    padding: '8px 12px',
                    textAlign: 'left'
                  }} {...props}>
                    {children}
                  </td>
                )
              }}
            >
              {itemContent || ''}
            </ReactMarkdown>
          );

        case 'text':
        default:
          // 纯文本内容，检查是否包含markdown格式
          const isMarkdownContent = /^[#*\-+>]|\[.*\]\(.*\)|^\d+\.|^\s*[-*+]\s|\|.*\|/.test((itemContent || '').trim()) ||
                                   (itemContent || '').includes('**') || 
                                   (itemContent || '').includes('*') || 
                                   (itemContent || '').includes('`');
          
          if (isMarkdownContent) {
            return (
              <ReactMarkdown
                key={index}
                components={{
                  code: ({ className, children, ...props }: any) => {
                    const match = /language-(\w+)/.exec(className || '');
                    const isInline = !match;
                    return !isInline ? (
                      <SyntaxHighlighter
                        style={vscDarkPlus as any}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  }
                }}
              >
                {itemContent || ''}
              </ReactMarkdown>
            );
          } else {
            return (
              <div key={index} className="text-content">
                {itemContent || ''}
              </div>
            );
          }
      }
    });
  };
  
  return (
    <div className={`chat-message ${role === 'user' ? 'user' : ''}`}>
      <div className="message-avatar">
        {role === 'user' ? <UserOutlined /> : <RobotOutlined />}
      </div>
      <div className="message-content-wrapper">
        <div className="message-content">
          {/* ✅ 修复：区分处理reply字段和content字段 */}
          {Array.isArray(reply) && reply.length > 0 ? (
            // 有reply字段时，使用专门的渲染函数处理富文本
            <div className="reply-content">
              {renderReplyContent(reply)}
              {isStreaming && (
                <span className="streaming-cursor"></span>
              )}
            </div>
          ) : (
            // 没有reply字段时，使用SmartContentParser处理content
            <SmartContentParser 
              content={content || ''} 
              isStreaming={isStreaming}
            />
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
