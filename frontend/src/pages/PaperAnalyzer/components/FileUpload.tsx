import React from 'react';
import { Upload, message } from 'antd';
import { InboxOutlined } from '@ant-design/icons';

const { Dragger } = Upload;

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, disabled }) => {
  return (
    <Dragger
      accept=".pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xls,.txt,.md"
      maxCount={1}
      disabled={disabled}
      beforeUpload={(file) => {
        onFileSelect(file);
        return false;
      }}
    >
      <p className="ant-upload-drag-icon">
        <InboxOutlined />
      </p>
      <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
      <p className="ant-upload-hint">
        支持 PDF、Word、PowerPoint、Excel、TXT、MD 格式的文件
      </p>
    </Dragger>
  );
};

export default FileUpload;
