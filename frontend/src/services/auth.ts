// frontend/src/services/auth.ts
import { FeishuAuthResponse, FeishuLoginResponse } from '../types/auth';
import { API_BASE_URL } from '../config/env';

export const authApi = {
  // 获取飞书登录链接
  getFeishuAuthUrl: async (): Promise<FeishuAuthResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/auth/feishu/login`);
    if (!response.ok) {
      throw new Error('Failed to get auth URL');
    }
    return response.json();
  },

  // 处理飞书回调
  handleFeishuCallback: async (code: string): Promise<FeishuLoginResponse> => {
    try {
      console.log('发送回调请求，code:', code);
      
      const response = await fetch(
        `${API_BASE_URL}/api/auth/feishu/callback?code=${code}`
      );
      
      // 添加响应状态日志
      console.log('Auth callback response:', {
        status: response.status,
        ok: response.ok,
        statusText: response.statusText
      });
      
      const data = await response.json();
      
      // 添加数据验证日志
      console.log('Auth callback data:', {
        hasStatus: !!data.status,
        hasUserInfo: !!data.user_info,
        hasAccessToken: !!data.access_token,
        hasExpiresAt: !!data.expires_at
      });
      
      if (!response.ok) {
        throw new Error(data.detail || '回调请求失败');
      }
      
      if (!data.status || data.status !== 'success') {
        throw new Error('响应状态无效');
      }
      
      return data;
    } catch (error) {
      // 添加错误上报
      console.error('Auth callback error:', {
        error,
        code,
        timestamp: new Date().toISOString()
      });
      throw error;
    }
  },

  // 添加 logout 方法
  logout: () => {
    localStorage.removeItem('user_info');
    localStorage.removeItem('access_token');
    localStorage.removeItem('expires_at');
    window.location.href = '/login';
  },

  // 添加检查登录状态方法
  checkLoginStatus: (): boolean => {
    const userInfo = localStorage.getItem('user_info');
    const accessToken = localStorage.getItem('access_token');
    return !!(userInfo && accessToken);
  }
};