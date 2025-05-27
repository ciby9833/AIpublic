import React from 'react';
import { authApi } from '../../services/auth';
import './FeishuLogin.css';

export const FeishuLogin: React.FC = () => {
  const handleLogin = async () => {
    try {
      const { auth_url } = await authApi.getFeishuAuthUrl();
      window.location.href = auth_url;
    } catch (error) {
      console.error('Failed to get auth URL:', error);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header">
          <h1>Cargo Tools</h1>
          <p className="subtitle">工具平台</p>
        </div>
        <div className="login-content">
          <button 
            className="feishu-login-button"
            onClick={handleLogin}
          >
            <img src="/feishu.png" alt="飞书" className="feishu-icon" />
            <span>Lark登录</span>
          </button>
        </div>
      </div>
    </div>
  );
};