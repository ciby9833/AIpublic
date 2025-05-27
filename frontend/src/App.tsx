// frontend/src/App.tsx
import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import FileUpload from './components/FileUpload'
import LanguageSelect from './components/LanguageSelect'
import TranslationStatus from './components/TranslationStatus'
import TextTranslate from './components/TextTranslate'
import TranslationModeSwitch from './components/TranslationModeSwitch'
import Footer from './components/Footer'
import { Tabs, Dropdown, Layout, Menu } from 'antd'
import { GlossaryList, GlossaryEditor } from './components/GlossaryManager'
import './style.css' 
import { translateDocument, checkTranslationStatus, downloadTranslationResult } from './services/api'
import GlossaryDatabaseSearch from './components/GlossaryManager/GlossaryDatabaseSearch'
import { API_BASE_URL } from './config/env'
import { BrowserRouter, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom'
import { AuthCallback } from './pages/AuthCallback'
import { FeishuLogin } from './components/Auth/FeishuLogin'
import { authApi, tokenManager } from './services/auth'
import type { MenuProps } from 'antd'
import { 
  UserOutlined, 
  MenuUnfoldOutlined, 
  MenuFoldOutlined, 
  TranslationOutlined,
  FilePdfOutlined,
  CompassOutlined,
  BookOutlined,
  TeamOutlined
} from '@ant-design/icons'
import UserManagement from './pages/UserManagement'
import DistanceCalculator from './pages/DistanceCalculator'
import PaperAnalyzer from './pages/PaperAnalyzer'
import MobileChat from './pages/MobileChat'
import ErrorBoundary from './components/ErrorBoundary'

const { Header, Content, Sider } = Layout;

export type TranslationStatus = 
    'idle' | 
    'uploading' | 
    'processing' |      
    'extracting' |      
    'creating_glossary' | 
    'translating' | 
    'downloading' | 
    'completed' | 
    'error';
export type TranslationMode = 'text' | 'document';

// 添加路由保护组件
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      console.log('Auth check:', {
        isLoggedIn: tokenManager.isLoggedIn(),
        isTokenExpired: tokenManager.isTokenExpired(),
        isTokenExpiringSoon: tokenManager.isTokenExpiringSoon(),
        timestamp: new Date().toISOString()
      });

      if (!tokenManager.isLoggedIn()) {
        console.log('Not logged in, redirecting to login');
        navigate('/login', { replace: true });
        return;
      }

      // 如果token即将过期，尝试刷新
      if (tokenManager.isTokenExpiringSoon()) {
        console.log('Token expiring soon, attempting refresh');
        const refreshed = await tokenManager.refreshToken();
        if (!refreshed) {
          console.log('Token refresh failed, redirecting to login');
          navigate('/login', { replace: true });
          return;
        }
      }
      
      setIsChecking(false);
    };

    checkAuth();
  }, [navigate]);

  if (isChecking) {
    return (
      <div className="loading-container">
        <div className="loading">{t('loading')}</div>
      </div>
    );
  }

  return <>{children}</>;
};

// 主应用布局组件
const AppLayout = () => {
  const { t, i18n } = useTranslation();
  const [collapsed, setCollapsed] = useState(false);
  const [userInfo, setUserInfo] = useState<any>(null);
  const location = useLocation();
  const navigate = useNavigate();
  
  useEffect(() => {
    const storedUserInfo = localStorage.getItem('user_info');
    if (storedUserInfo) {
      setUserInfo(JSON.parse(storedUserInfo));
    }
  }, []);

  // 下拉菜单项
  const dropdownItems: MenuProps['items'] = [
    {
      key: 'language',
      label: t('language.switch'),
      children: [
        {
          key: 'zh',
          label: t('language.zh'),
          onClick: () => i18n.changeLanguage('zh')
        },
        {
          key: 'en',
          label: t('language.en'),
          onClick: () => i18n.changeLanguage('en')
        },
        {
          key: 'id',
          label: t('language.id'),
          onClick: () => i18n.changeLanguage('id')
        }
      ]
    },
    {
      type: 'divider'
    },
    {
      key: 'logout',
      label: t('logout'),
      danger: true,
      onClick: () => authApi.logout(),
    }
  ];

  // 侧边栏菜单项
  const menuItems = [
    {
      key: '/paper-analyzer',
      icon: <FilePdfOutlined />,
      label: t('tabs.paperAnalyzer')
    },
    {
      key: '/translation',
      icon: <TranslationOutlined />,
      label: t('tabs.translation')
    },
    {
      key: '/distance',
      icon: <CompassOutlined />,
      label: t('tabs.distanceCalculator')
    },
    {
      key: '/glossary',
      icon: <BookOutlined />,
      label: t('tabs.glossaryManagement')
    },
    {
      key: '/users',
      icon: <TeamOutlined />,
      label: t('tabs.userManagement')
    }
  ];

  const handleMenuClick = (e: { key: string }) => {
    navigate(e.key);
  };

  return (
    <Layout className="app-layout">
      <Sider 
        collapsible 
        collapsed={collapsed} 
        onCollapse={value => setCollapsed(value)}
        width={240}
        className="sidebar"
        trigger={null}
      >
        <div className="logo-container">
          {!collapsed && <h1 className="logo-text">{t('title')}</h1>}
          <div className="logo-trigger">
            {collapsed ? (
              <MenuUnfoldOutlined className="trigger" onClick={() => setCollapsed(false)} />
            ) : (
              <MenuFoldOutlined className="trigger" onClick={() => setCollapsed(true)} />
            )}
          </div>
        </div>
        <Menu
          theme="light"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={handleMenuClick}
        />
        {userInfo && (
          <div className="sidebar-user-info">
            <Dropdown menu={{ items: dropdownItems }} placement="topRight">
              <div className="user-info">
                {userInfo.avatar_url ? (
                  <img 
                    src={userInfo.avatar_url} 
                    alt="avatar" 
                    className="user-avatar" 
                  />
                ) : (
                  <UserOutlined className="user-avatar-icon" />
                )}
                {!collapsed && <span className="user-name">{userInfo.name}</span>}
              </div>
            </Dropdown>
          </div>
        )}
      </Sider>
      <Layout>
        <Content className="app-content">
          <Routes>
            <Route path="/" element={<Navigate to="/paper-analyzer" replace />} />
            <Route path="/translation" element={<TranslationPage />} />
            <Route path="/paper-analyzer" element={<PaperAnalyzerPage />} />
            <Route path="/distance" element={<DistanceCalculatorPage />} />
            <Route path="/glossary" element={<GlossaryPage />} />
            <Route path="/users" element={<UserManagementPage />} />
          </Routes>
          <Footer />
        </Content>
      </Layout>
    </Layout>
  );
};

// 各功能页面组件
const TranslationPage = () => {
  const { t } = useTranslation();
  const [mode, setMode] = useState<TranslationMode>(
    localStorage.getItem('translationMode') as TranslationMode || 'text'
  );
  const [status, setStatus] = useState<TranslationStatus>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [targetLanguage, setTargetLanguage] = useState<string>(
    localStorage.getItem('targetLanguage') || 'ID'
  );
  const [sourceLang, setSourceLang] = useState<string>(
    localStorage.getItem('sourceLang') || 'AUTO'
  );
  const [useGlossary, setUseGlossary] = useState<boolean>(true);

  useEffect(() => {
    localStorage.setItem('translationMode', mode);
    localStorage.setItem('targetLanguage', targetLanguage);
    localStorage.setItem('sourceLang', sourceLang);
  }, [mode, targetLanguage, sourceLang]);

  const handleSourceLangChange = (lang: string) => {
    setSourceLang(lang);
  };

  const handleTargetLangChange = (lang: string) => {
    setTargetLanguage(lang);
  };

  const handleTranslate = async () => {
    if (!selectedFile) return;

    try {
        if (useGlossary && (sourceLang === 'AUTO' || !sourceLang)) {
            setErrorMessage(t('error.sourceLanguageRequired'));
            setStatus('error');
            return;
        }

        setStatus('uploading');
        setErrorMessage('');
        
        const response = await translateDocument(
          selectedFile, 
          sourceLang, 
          targetLanguage, 
          useGlossary
        );

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail?.message || 'Upload failed');
        }

        const { document_id, document_key, has_glossary } = await response.json();
        console.log('Upload successful:', { document_id, document_key, has_glossary });

        setStatus('translating');
        let retryCount = 0;
        const maxRetries = 30;

        const checkStatus = async () => {
            const response = await checkTranslationStatus(document_id, document_key);

            if (!response.ok) {
                throw new Error('Status check failed');
            }

            const statusData = await response.json();
            
            switch(statusData.status) {
                case 'processing_document':
                    setStatus('processing');
                    break;
                case 'extracting_terms':
                    setStatus('extracting');
                    break;
                case 'creating_glossary':
                    setStatus('creating_glossary');
                    break;
                case 'translating':
                    setStatus('translating');
                    break;
                case 'done':
                    return true;
                case 'error':
                    throw new Error(statusData.message || t('error.translationFailed'));
            }
            return false;
        };

        while (retryCount < maxRetries) {
            const isDone = await checkStatus();
            if (isDone) break;
            
            retryCount++;
            await new Promise(resolve => setTimeout(resolve, 2000));
        }

        if (retryCount >= maxRetries) {
            throw new Error(t('error.timeout'));
        }

        setStatus('downloading');
        const downloadResponse = await downloadTranslationResult(document_id, document_key);

        if (!downloadResponse.ok) {
            const errorData = await downloadResponse.json();
            throw new Error(errorData.detail?.message || errorData.detail || 'Download failed');
        }

        const blob = await downloadResponse.blob();
        if (blob.size === 0) {
            throw new Error(t('error.emptyFile'));
        }

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = t('download.filename', { 
          filename: selectedFile.name 
        });
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        setStatus('completed');
        setSelectedFile(null);
        setTimeout(() => {
            setStatus('idle');
        }, 2000);

    } catch (error) {
        console.error('Translation error:', error);
        setStatus('error');
        setErrorMessage(t('error.translationFailed'));
    }
  };

  return (
    <div className="content-card">
      <TranslationModeSwitch mode={mode} onModeChange={setMode} />
      <div className="translation-content">
        {mode === 'text' ? (
          <TextTranslate 
            disabled={status !== 'idle' && status !== 'completed' && status !== 'error'}
          />
        ) : (
          <>
            <FileUpload
              onFileSelect={setSelectedFile}
              selectedFile={selectedFile}
              disabled={status !== 'idle' && status !== 'completed' && status !== 'error'}
            />
            <div className="language-controls">
              <LanguageSelect
                value={sourceLang}
                onChange={handleSourceLangChange}
                disabled={status !== 'idle' && status !== 'completed' && status !== 'error'}
                isSource={true}
                disableAuto={useGlossary}
              />
              <LanguageSelect
                value={targetLanguage}
                onChange={handleTargetLangChange}
                disabled={status !== 'idle' && status !== 'completed' && status !== 'error'}
              />
              <div className="glossary-control">
                <label>
                  <input
                    type="checkbox"
                    checked={useGlossary}
                    onChange={(e) => setUseGlossary(e.target.checked)}
                    disabled={status !== 'idle' && status !== 'completed' && status !== 'error'}
                  />
                  {t('useGlossary')}
                </label>
              </div>
            </div>
            <button
              onClick={handleTranslate}
              disabled={!selectedFile || (status !== 'idle' && status !== 'completed' && status !== 'error')}
              className="translate-button"
            >
              {t('button.translate')}
            </button>
            <TranslationStatus status={status} errorMessage={errorMessage} />
          </>
        )}
      </div>
    </div>
  );
};

const PaperAnalyzerPage = () => (
  <div className="content-card paper-analyzer-card">
    <PaperAnalyzer />
  </div>
);

const DistanceCalculatorPage = () => (
  <div className="content-card">
    <DistanceCalculator />
  </div>
);

const GlossaryPage = () => {
  const { t } = useTranslation();
  
  return (
    <div className="content-card">
      <Tabs
        defaultActiveKey="databaseSearch"
        items={[
          {
            key: 'databaseSearch',
            label: t('glossary.databaseSearch'),
            children: <GlossaryDatabaseSearch />
          },
          {
            key: 'list',
            label: t('glossary.list'),
            children: <GlossaryList />
          }
        ]}
      />
    </div>
  );
};

const UserManagementPage = () => (
  <div className="content-card">
    <UserManagement />
  </div>
);

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<FeishuLogin />} />
          <Route path="/auth/callback" element={<AuthCallback />} />
          <Route path="/mobile-chat" element={<MobileChat />} />
          <Route path="/*" element={
            <ProtectedRoute>
              <AppLayout />
            </ProtectedRoute>
          } />
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App