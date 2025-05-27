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
          urlParams: Object.fromEntries(urlParams.entries()),
          fullUrl: window.location.href,
          search: window.location.search,
          hash: window.location.hash,
          pathname: window.location.pathname
        });

        if (!authData) {
          throw new Error('No auth data received in URL parameter');
        }

        // URL解码认证数据（防止URL编码导致的问题）
        let decodedUrlData;
        try {
          decodedUrlData = decodeURIComponent(authData);
          console.log('URL decoded data length:', decodedUrlData.length);
          console.log('URL decoding changed data:', authData !== decodedUrlData);
        } catch (urlDecodeError) {
          console.warn('URL decode failed, using original data:', urlDecodeError);
          decodedUrlData = authData;
        }

        // 解码 base64 编码的认证数据
        let decodedAuthData;
        try {
          // 清理Base64字符串：移除空格和其他非Base64字符
          const cleanedAuthData = decodedUrlData.replace(/[^A-Za-z0-9+/=]/g, '');
          console.log('Original auth data length:', decodedUrlData.length);
          console.log('Cleaned auth data length:', cleanedAuthData.length);
          console.log('Removed characters:', decodedUrlData.length - cleanedAuthData.length);
          
          // 检查Base64字符串的有效性
          if (cleanedAuthData.length % 4 !== 0) {
            console.warn('Base64 string length is not a multiple of 4, padding...');
            // 添加必要的填充
            const padding = '='.repeat((4 - (cleanedAuthData.length % 4)) % 4);
            const paddedData = cleanedAuthData + padding;
            console.log('Padded data length:', paddedData.length);
            decodedAuthData = atob(paddedData);
          } else {
            decodedAuthData = atob(cleanedAuthData);
          }
          
          console.log('Base64 decoded successfully, length:', decodedAuthData.length);
        } catch (decodeError) {
          console.error('Base64 decode error:', decodeError);
          console.error('Original auth data:', authData);
          console.error('URL decoded data:', decodedUrlData);
          console.error('Auth data length:', decodedUrlData.length);
          console.error('Auth data contains invalid chars:', /[^A-Za-z0-9+/=]/.test(decodedUrlData));
          
          // 尝试直接解码原始数据作为备用方案
          try {
            console.log('Attempting to decode original data as fallback...');
            const cleanedOriginal = authData.replace(/[^A-Za-z0-9+/=]/g, '');
            decodedAuthData = atob(cleanedOriginal);
            console.log('Fallback decode successful');
          } catch (fallbackError) {
            console.error('Fallback decode also failed:', fallbackError);
            throw new Error('Failed to decode auth data from URL parameter');
          }
        }
        
        const data = JSON.parse(decodedAuthData);
        
        // 添加数据验证日志
        console.log('Parsed auth data:', {
          hasStatus: !!data.status,
          hasUserInfo: !!data.user_info,
          hasAccessToken: !!data.access_token,
          hasExpiresAt: !!data.expires_at,
          status: data.status
        });
        
        if (data.status === 'success' && data.user_info && data.access_token) {
          // 保存认证数据到 localStorage
          localStorage.setItem('user_info', JSON.stringify(data.user_info));
          localStorage.setItem('access_token', data.access_token);
          localStorage.setItem('expires_at', data.expires_at.toString());
          
          console.log('Auth data saved, redirecting to home');
          hasProcessed.current = true;
          navigate('/', { replace: true });
        } else {
          console.error('Invalid auth data structure:', data);
          throw new Error('Invalid auth data');
        }
      } catch (error) {
        console.error('Auth callback error:', error);
        // 添加更详细的错误信息
        if (error instanceof SyntaxError) {
          console.error('JSON parsing error:', error.message);
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
      'incomplete_data': '认证数据不完整'
    };
    return errorMessages[error] || '登录过程中发生错误';
  };

  return (
    <div className="auth-callback-container">
      <div className="loading-message">登录验证中...</div>
    </div>
  );
};