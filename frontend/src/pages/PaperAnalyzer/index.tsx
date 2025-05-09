// frontend/src/pages/PaperAnalyzer/index.tsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, Upload, Button, Input, message, List, Layout, Spin, Select, Tooltip, Dropdown, Menu, Switch } from 'antd';
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
  CheckCircleOutlined
} from '@ant-design/icons';
import { paperAnalyzerApi } from '../../services/paperAnalyzer';
import './styles.css';
import type { MenuProps } from 'antd';
import ChatMessage from './components/index';
import { ChatMessage as ChatMessageType } from '../../types/chat';
import { List as VirtualList, AutoSizer, ListRowProps } from 'react-virtualized';
import { throttle } from 'lodash';

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
      
      setResponses(prev => [...prev, {
        question: `正在分析文件: ${file.name}`,
        answer: '请耐心等待，正在处理中...',
        timestamp: new Date().toISOString()
      }]);

      const result = await paperAnalyzerApi.analyzePaper(file);
      if (result.status === 'success' && result.paper_id) {
        setCurrentPaperId(result.paper_id);
        if (result.content) {
          console.log("Setting document content, length:", result.content.length);
          setDocumentContent(result.content);
          
          // Instead of creating complex mapping, just set an empty object
          // The display will use direct line splitting from documentContent
          console.log("No line mapping provided, using direct content display");
          setLineMapping({});
          
          setTotalLines(result.total_lines || result.content.split('\n').length);
        } else {
          const contentData = await paperAnalyzerApi.getDocumentContent(result.paper_id);
          setDocumentContent(contentData.content);
          setLineMapping(contentData.line_mapping || {});
          setTotalLines(contentData.total_lines || 0);
          console.log("Document content fetched:", contentData.content.substring(0, 100));
          console.log("Line mapping:", Object.keys(contentData.line_mapping || {}).length);
        }
        message.success('文档分析完成');
        
        setResponses(prev => [...prev, {
          question: `文件分析完成: ${file.name}`,
          answer: '现在您可以开始提问了',
          timestamp: new Date().toISOString()
        }]);
      } else {
        message.error(result.message || '文档分析失败');
      }
    } catch (error) {
      console.error('Analysis error:', error);
      message.error('文档分析失败');
    } finally {
      setAnalyzing(false);
      setLoading(false);
    }
  };

  // 处理提问
  const handleAsk = async () => {
    if (!question.trim() && !selectedFile) {
      message.warning('请输入问题或上传文件');
      return;
    }

    try {
      setSending(true);
      const result = await paperAnalyzerApi.askQuestion(question, currentPaperId);
      setResponses(prev => [...prev, { 
        question, 
        answer: result.response,
        timestamp: new Date().toISOString()
      }]);
      setQuestion('');
      setSelectedFile(null);
      
      // 滚动到最新消息
      setTimeout(() => {
        if (chatMessagesRef.current) {
          chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
        }
      }, 100);
    } catch (error) {
      message.error('提问失败');
    } finally {
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
      message.warning('请选择目标语言');
      return;
    }

    try {
      setTranslationLoading(true);
      const translated = await paperAnalyzerApi.translatePaper(currentPaperId, selectedLanguage);
      setTranslatedContent(translated);
      message.success('翻译完成');
    } catch (error: any) {
      console.error('Translation error:', error);
      message.error(error.message || '翻译失败');
    } finally {
      setTranslationLoading(false);
    }
  };

  const handleDownload = async (format: string) => {
    if (!translatedContent) {
      message.warning('没有可下载的翻译内容');
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
      message.error('下载失败');
    }
  };

  // 在组件内定义菜单项
  const downloadMenuItems: MenuProps['items'] = [
    {
      key: 'docx',
      label: 'Word 文档',
      icon: <FileWordOutlined />,
    },
    {
      key: 'pdf',
      label: 'PDF 文档',
      icon: <FilePdfOutlined />,
    },
    {
      key: 'md',
      label: 'Markdown',
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

  return (
    <Layout className="paper-analyzer-layout">
      <Sider 
        width={400}
        collapsible={false}
        collapsed={collapsed}
        className="paper-analyzer-sider"
      >
        <div className="chat-container">
          <div className="chat-header">
            {selectedFile ? (
              <div className="selected-file">
                <FileTextOutlined />
                <span className="file-name">{selectedFile.name}</span>
                <Button 
                  type="text" 
                  icon={<DeleteOutlined />} 
                  onClick={() => setSelectedFile(null)}
                  className="delete-file-btn"
                />
              </div>
            ) : (
              <div className="header-placeholder" />
            )}
            <Button
              type="text"
              icon={collapsed ? <RightOutlined /> : <LeftOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              className="collapse-button"
              title={collapsed ? "展开" : "收起"}
            />
          </div>
          <div className="chat-messages" ref={chatMessagesRef}>
            <List
              className="response-list"
              itemLayout="vertical"
              dataSource={responses}
              renderItem={(item) => (
                <ChatMessage message={item} />
              )}
            />
          </div>
          
          <div className="chat-input-container">
            <div 
              className={`input-area ${analyzing ? 'analyzing' : ''}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <TextArea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder={analyzing ? "正在分析文档，请稍候..." : "请输入您的问题或拖拽文件到此处"}
                rows={3}
                disabled={analyzing}
              />
              <div className="input-actions">
                <div className="left-actions">
                  <Upload
                    accept=".pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xls,.txt,.md"
                    maxCount={1}
                    showUploadList={false}
                    beforeUpload={(file) => {
                      handleFileUpload(file);
                      return false;
                    }}
                    disabled={analyzing}
                  >
                    <Tooltip title={analyzing ? "正在分析中..." : "上传文件"}>
                      <Button 
                        icon={<PaperClipOutlined />} 
                        disabled={analyzing}
                      />
                    </Tooltip>
                  </Upload>
                </div>
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  onClick={handleAsk}
                  loading={sending || analyzing}
                  disabled={!currentPaperId || analyzing}
                >
                  {analyzing ? "分析中..." : "发送"}
                </Button>
              </div>
            </div>
          </div>
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
                placeholder="选择目标语言"
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
                翻译
              </Button>
            </div>
          </div>
          <div className="document-content-split">
            <div className="original-content" ref={originalContentRef}>
              <h3>原文</h3>
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
                  <p>请上传文档或等待分析完成</p>
                </div>
              )}
            </div>
            <div className="translated-content" ref={translatedContentRef}>
              <div className="content-header">
                <h3>翻译</h3>
                <div className="header-actions">
                  {translatedContent && (
                    <>
                      <Switch
                        size="small"
                        checked={syncScrolling}
                        onChange={(checked) => setSyncScrolling(checked)}
                        checkedChildren="同步滚动"
                        unCheckedChildren="独立滚动"
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
                          title="下载翻译"
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
                  <p>请选择语言并点击翻译按钮</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </Content>
    </Layout>
  );
};

export default PaperAnalyzer;