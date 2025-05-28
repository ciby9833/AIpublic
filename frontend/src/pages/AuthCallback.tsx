// frontend/src/pages/AuthCallback.tsx
import { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { message } from 'antd';

export const AuthCallback = () => {
  const navigate = useNavigate();
  const hasProcessed = useRef(false);

  useEffect(() => {
    // 防止React.StrictMode导致的双重执行
    if (hasProcessed.current) {
      console.log('AuthCallback already processed, skipping...');
      return;
    }
    const handleAuthCallback = async () => {
      try {
        // 检查URL中是否有错误参数
        const urlParams = new URLSearchParams(window.location.search);
        const error = urlParams.get('error');
        
        if (error) {
          console.error('Auth error:', error);
          message.error(getErrorMessage(error));
          navigate('/login');
          return;
        }

        // 从 URL 参数中读取认证数据
        const authData = urlParams.get('data');
        
        console.log('Auth callback data:', {
          hasAuthData: !!authData,
          authDataLength: authData?.length,
          authDataPreview: authData?.substring(0, 50) + '...',
          fullUrl: window.location.href,
          search: window.location.search
        });

        if (!authData) {
          throw new Error('No auth data received in URL parameter');
        }

        // 多层解码处理
        let decodedAuthData;
        try {
          // 1. URL解码
          let urlDecodedData = decodeURIComponent(authData);
          console.log('URL decoded data length:', urlDecodedData.length);
          
          // 2. 清理Base64字符串
          const cleanedAuthData = urlDecodedData.replace(/[^A-Za-z0-9+/=]/g, '');
          console.log('Cleaned auth data length:', cleanedAuthData.length);
          
          // 3. 添加Base64填充（如果需要）
          let paddedData = cleanedAuthData;
          if (cleanedAuthData.length % 4 !== 0) {
            const padding = '='.repeat((4 - (cleanedAuthData.length % 4)) % 4);
            paddedData = cleanedAuthData + padding;
            console.log('Added padding, new length:', paddedData.length);
          }
          
          // 4. Base64解码
          decodedAuthData = atob(paddedData);
          console.log('Base64 decoded successfully, length:', decodedAuthData.length);
          
        } catch (decodeError) {
          console.error('Decode error:', decodeError);
          console.error('Original auth data:', authData);
          
          // 备用解码方案
          try {
            console.log('Trying fallback decode...');
            // 直接尝试Base64解码原始数据
            const directCleaned = authData.replace(/[^A-Za-z0-9+/=]/g, '');
            decodedAuthData = atob(directCleaned);
            console.log('Fallback decode successful');
          } catch (fallbackError) {
            console.error('All decode attempts failed:', fallbackError);
            throw new Error('Failed to decode authentication data');
          }
        }
        
        // 解析JSON数据
        let data;
        try {
          data = JSON.parse(decodedAuthData);
          console.log('JSON parsed successfully');
        } catch (jsonError) {
          console.error('JSON parse error:', jsonError);
          console.error('Decoded data:', decodedAuthData);
          throw new Error('Failed to parse authentication data');
        }
        
        // 验证数据结构
        console.log('Parsed auth data:', {
          hasStatus: !!data.status,
          hasUserInfo: !!data.user_info,
          hasAccessToken: !!data.access_token,
          hasExpiresAt: !!data.expires_at,
          status: data.status,
          userName: data.user_info?.name
        });
        
        if (data.status === 'success' && data.user_info && data.access_token) {
          // 保存认证数据到 localStorage
          localStorage.setItem('user_info', JSON.stringify(data.user_info));
          localStorage.setItem('access_token', data.access_token);
          localStorage.setItem('expires_at', data.expires_at.toString());
          
          console.log('Auth data saved successfully, redirecting to home');
          console.log('Saved user:', data.user_info.name);
          
          hasProcessed.current = true;
          navigate('/', { replace: true });
        } else {
          console.error('Invalid auth data structure:', data);
          throw new Error('Invalid authentication data structure');
        }
        
      } catch (error) {
        console.error('Auth callback error:', error);
        console.error('Error details:', {
          message: error instanceof Error ? error.message : 'Unknown error',
          stack: error instanceof Error ? error.stack : 'No stack trace',
          url: window.location.href,
          userAgent: navigator.userAgent
        });
        
        // 显示用户友好的错误消息
        if (error instanceof Error && error.message.includes('decode')) {
          message.error('登录数据解析失败，请重新登录');
        } else if (error instanceof Error && error.message.includes('parse')) {
          message.error('登录数据格式错误，请重新登录');
        } else {
          message.error('登录处理失败，请重新登录');
        }
        
        hasProcessed.current = true;
        navigate('/login');
      }
    };

    handleAuthCallback();
  }, [navigate]);

  // 错误消息映射
  const getErrorMessage = (error: string) => {
    const errorMessages: Record<string, string> = {
      'token_failed': '获取访问令牌失败',
      'user_info_failed': '获取用户信息失败',
      'no_tenant_key': '未找到组织信息',
      'unauthorized_org': '未授权的组织访问',
      'auth_failed': '认证失败',
      'invalid_state': '无效的状态参数',
      'invalid_user_data': '用户数据无效',
      'missing_user_data': '缺少必要的用户数据',
      'incomplete_data': '认证数据不完整',
      'url_too_long': '认证数据过长，请联系管理员',
      'encoding_error': '数据编码错误，请重新登录'
    };
    return errorMessages[error] || '登录过程中发生错误';
  };

  return (
    <div className="auth-callback-container">
      <div className="loading-message">登录验证中...</div>
    </div>
  );
};