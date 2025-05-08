// frontend/src/pages/AuthCallback.tsx
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { message } from 'antd';

export const AuthCallback = () => {
  const navigate = useNavigate();

  useEffect(() => {
    try {
      // 检查URL中是否有错误参数
      const urlParams = new URLSearchParams(window.location.search);
      const error = urlParams.get('error');
      
      if (error) {
        console.error('Auth error:', error);
        // 显示适当的错误消息
        // 可以使用antd的message组件
        message.error(getErrorMessage(error));
        navigate('/login');
        return;
      }

      const encodedData = urlParams.get('data');
      
      // 添加日志记录
      console.log('Auth callback data:', {
        hasEncodedData: !!encodedData,
        urlParams: Object.fromEntries(urlParams.entries())
      });

      if (!encodedData) {
        throw new Error('No auth data received');
      }

      // 解码并解析认证数据
      const data = JSON.parse(decodeURIComponent(encodedData));
      
      // 添加数据验证日志
      console.log('Parsed auth data:', {
        hasStatus: !!data.status,
        hasUserInfo: !!data.user_info,
        hasAccessToken: !!data.access_token,
        hasExpiresAt: !!data.expires_at
      });
      
      if (data.status === 'success' && data.user_info && data.access_token) {
        // 保存认证数据
        localStorage.setItem('user_info', JSON.stringify(data.user_info));
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('expires_at', data.expires_at.toString());
        
        console.log('Auth data saved, redirecting to home');
        
        // 使用 navigate 而不是 window.location
        navigate('/', { replace: true });
      } else {
        throw new Error('Invalid auth data');
      }
    } catch (error) {
      console.error('Auth callback error:', error);
      // 添加更详细的错误信息
      if (error instanceof SyntaxError) {
        console.error('JSON parsing error:', error.message);
      }
      navigate('/login');
    }
  }, [navigate]);

  // 错误消息映射
  const getErrorMessage = (error: string) => {
    const errorMessages: Record<string, string> = {
      'token_failed': '获取访问令牌失败',
      'user_info_failed': '获取用户信息失败',
      'no_tenant_key': '未找到组织信息',
      'unauthorized_org': '未授权的组织访问',
      'auth_failed': '认证失败',
      'invalid_state': '无效的状态参数'
    };
    return errorMessages[error] || '登录过程中发生错误';
  };

  return (
    <div className="auth-callback-container">
      <div className="loading-message">登录验证中...</div>
    </div>
  );
};