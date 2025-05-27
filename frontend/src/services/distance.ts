// frontend/src/services/distance.ts
import { message } from 'antd';
import { API_BASE_URL } from '../config/env';
import { apiRequest } from './auth';

// 复用 api.ts 中的 URL 创建方法
const createApiUrl = (path: string) => {
  return import.meta.env.PROD ? path : `${API_BASE_URL}${path}`;
};

// 任务状态类型定义
export interface Task {
  id: string;
  filename: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
  completed_at?: string;
  error?: string;
  result_data?: string;
  result_filename?: string;
}

export const distanceApi = {
  // 计算距离（添加任务到队列）
  calculateDistance: async (file: File): Promise<{ task_id: string }> => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await apiRequest(
        createApiUrl('/api/calculate-distance'),
        {
          method: 'POST',
          body: formData
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to add distance calculation task:', error);
      message.error('添加距离计算任务失败');
      throw error;
    }
  },

  // 获取所有任务
  getAllTasks: async (): Promise<Task[]> => {
    try {
      const response = await apiRequest(
        createApiUrl('/api/tasks'),
        {
          method: 'GET'
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to fetch tasks:', error);
      message.error('获取任务列表失败');
      throw error;
    }
  },

  // 获取单个任务状态
  getTaskStatus: async (taskId: string): Promise<Task> => {
    try {
      const response = await apiRequest(
        createApiUrl(`/api/tasks/${taskId}`),
        {
          method: 'GET'
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to fetch task status:', error);
      message.error('获取任务状态失败');
      throw error;
    }
  },

  // 下载任务结果
  downloadResult: async (taskId: string): Promise<Blob> => {
    try {
      const response = await apiRequest(
        createApiUrl(`/api/tasks/${taskId}/download`),
        {
          method: 'GET',
          headers: {
            'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
          }
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      // 从响应头中获取文件名
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `result_${taskId}.xlsx`;
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename\*=UTF-8''([^;]+)/);
        if (filenameMatch) {
          filename = decodeURIComponent(filenameMatch[1]);
        }
      }

      const blob = await response.blob();
      return new Blob([blob], { 
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' 
      });
    } catch (error) {
      console.error('Failed to download result:', error);
      message.error('下载结果失败');
      throw error;
    }
  },

  // 取消任务
  cancelTask: async (taskId: string): Promise<void> => {
    try {
      const response = await apiRequest(
        createApiUrl(`/api/tasks/${taskId}/cancel`),
        {
          method: 'POST'
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }
    } catch (error) {
      console.error('Failed to cancel task:', error);
      message.error('取消任务失败');
      throw error;
    }
  },

  // 删除任务
  deleteTask: async (taskId: string): Promise<void> => {
    try {
      const response = await apiRequest(
        createApiUrl(`/api/tasks/${taskId}`),
        {
          method: 'DELETE'
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }
    } catch (error) {
      console.error('Failed to delete task:', error);
      message.error('删除任务失败');
      throw error;
    }
  }
};