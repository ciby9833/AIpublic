// frontend/src/services/auth.ts
import { FeishuAuthResponse, FeishuLoginResponse } from '../types/auth';
import { API_BASE_URL } from '../config/env';

// Token管理类
class TokenManager {
  private refreshPromise: Promise<boolean> | null = null;

  // 检查token是否即将过期（1小时内）
  isTokenExpiringSoon(): boolean {
    const expiresAt = localStorage.getItem('expires_at');
    if (!expiresAt) return true;
    
    const now = Date.now() / 1000;
    const expiresTime = Number(expiresAt);
    return (expiresTime - now) < 3600; // 1小时
  }

  // 检查token是否已过期
  isTokenExpired(): boolean {
    const expiresAt = localStorage.getItem('expires_at');
    if (!expiresAt) return true;
    
    const now = Date.now() / 1000;
    const expiresTime = Number(expiresAt);
    return now >= expiresTime;
  }

  // 刷新token
  async refreshToken(): Promise<boolean> {
    // 防止并发刷新
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = this._doRefreshToken();
    const result = await this.refreshPromise;
    this.refreshPromise = null;
    return result;
  }

  private async _doRefreshToken(): Promise<boolean> {
    try {
      const accessToken = localStorage.getItem('access_token');
      if (!accessToken) {
        return false;
      }

      const response = await fetch(`${API_BASE_URL}/api/auth/refresh?current_token=${accessToken}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('expires_at', data.expires_at.toString());
        console.log('Token refreshed successfully');
        return true;
      } else {
        console.error('Token refresh failed:', response.status);
        return false;
      }
    } catch (error) {
      console.error('Token refresh error:', error);
      return false;
    }
  }

  // 清除所有认证数据
  clearAuth(): void {
    localStorage.removeItem('user_info');
    localStorage.removeItem('access_token');
    localStorage.removeItem('expires_at');
  }

  // 获取当前token
  getToken(): string | null {
    return localStorage.getItem('access_token');
  }

  // 检查是否已登录
  isLoggedIn(): boolean {
    const userInfo = localStorage.getItem('user_info');
    const accessToken = localStorage.getItem('access_token');
    const expiresAt = localStorage.getItem('expires_at');
    return !!(userInfo && accessToken && expiresAt && !this.isTokenExpired());
  }
}

// 创建token管理器实例
const tokenManager = new TokenManager();

// API请求拦截器
export const apiRequest = async (url: string, options: RequestInit = {}): Promise<Response> => {
  // 检查token状态
  if (tokenManager.isTokenExpired()) {
    console.log('Token expired, redirecting to login');
    authApi.logout();
    throw new Error('Token expired');
  }

  // 如果token即将过期，尝试刷新
  if (tokenManager.isTokenExpiringSoon()) {
    console.log('Token expiring soon, attempting refresh');
    const refreshed = await tokenManager.refreshToken();
    if (!refreshed) {
      console.log('Token refresh failed, redirecting to login');
      authApi.logout();
      throw new Error('Token refresh failed');
    }
  }

  // 添加认证头
  const token = tokenManager.getToken();
  const headers = {
    ...options.headers,
    ...(token && { 'Authorization': `Bearer ${token}` })
  };

  // 发送请求
  const response = await fetch(url, {
    ...options,
    headers
  });

  // 处理401错误
  if (response.status === 401) {
    console.log('Received 401, attempting token refresh');
    const refreshed = await tokenManager.refreshToken();
    
    if (refreshed) {
      // 重试请求
      const newToken = tokenManager.getToken();
      const retryHeaders = {
        ...options.headers,
        ...(newToken && { 'Authorization': `Bearer ${newToken}` })
      };
      
      return fetch(url, {
        ...options,
        headers: retryHeaders
      });
    } else {
      // 刷新失败，跳转登录
      console.log('Token refresh failed after 401, redirecting to login');
      authApi.logout();
      throw new Error('Authentication failed');
    }
  }

  return response;
};

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

  // 登出
  logout: () => {
    tokenManager.clearAuth();
    window.location.href = '/login';
  },

  // 检查登录状态
  checkLoginStatus: (): boolean => {
    return tokenManager.isLoggedIn();
  },

  // 手动刷新token
  refreshToken: (): Promise<boolean> => {
    return tokenManager.refreshToken();
  },

  // 获取token管理器（用于其他模块）
  getTokenManager: () => tokenManager
};

// 导出token管理器供其他模块使用
export { tokenManager };