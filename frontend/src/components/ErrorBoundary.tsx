import React, { Component, ErrorInfo, ReactNode } from 'react';
import { message } from 'antd';
import { authApi } from '../services/auth';
import './ErrorBoundary.css';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    // 检查是否是认证相关错误
    if (this.isAuthError(error)) {
      message.error('登录已过期，请重新登录');
      authApi.logout();
      return;
    }

    // 其他错误的处理
    message.error('应用发生错误，请刷新页面重试');
  }

  private isAuthError(error: Error): boolean {
    const authErrorMessages = [
      'Token expired',
      'Token refresh failed',
      'Authentication failed',
      'Invalid token',
      'No access token provided'
    ];
    
    return authErrorMessages.some(msg => 
      error.message.includes(msg) || error.name.includes(msg)
    );
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <div className="error-content">
            <h2>出现了一些问题</h2>
            <p>应用遇到了意外错误，请刷新页面重试。</p>
            <button 
              onClick={() => window.location.reload()}
              className="retry-button"
            >
              刷新页面
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary; 