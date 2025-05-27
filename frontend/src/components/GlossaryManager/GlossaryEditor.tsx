// src/components/GlossaryManager/GlossaryEditor.tsx   前端术语表编辑组件
import React, { useState } from 'react';
import { Form, Input, Select, Button, message } from 'antd';
import { useTranslation } from 'react-i18next';
import { glossaryApi } from '../../services/api';

const { Option } = Select;

interface GlossaryEditorProps {
  onSuccess: () => void;
}

const GlossaryEditor: React.FC<GlossaryEditorProps> = ({ onSuccess }) => {
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);

  const onFinish = async (values: any) => {
    try {
      setLoading(true);
      const response = await glossaryApi.createGlossary({
        name: values.name,
        dictionaries: [{
          source_lang: values.source_lang,
          target_lang: values.target_lang,
          entries: values.entries,
          entries_format: 'tsv'
        }]
      });
      
      if (response.ok) {
        message.success('Glossary created successfully');
        form.resetFields();
        onSuccess();
      } else {
        throw new Error('Failed to create glossary');
      }
    } catch (error) {
      console.error('Create glossary error:', error);
      message.error(t('glossary.createError') || 'Failed to create glossary');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Form form={form} onFinish={onFinish} layout="vertical">
      <Form.Item
        name="name"
        label={t('glossary.name')}
        rules={[{ required: true, message: t('glossary.nameRequired') }]}
      >
        <Input placeholder={t('glossary.namePlaceholder')} />
      </Form.Item>

      <Form.Item
        name="source_lang"
        label={t('glossary.sourceLang')}
        rules={[{ required: true, message: t('glossary.sourceLangRequired') }]}
      >
        <Select>
            <Option value="ZH">{t('language.ZH')}</Option>
            <Option value="EN">{t('language.EN')}</Option>
            <Option value="ID">{t('language.ID')}</Option>
        </Select>
      </Form.Item>

      <Form.Item
        name="target_lang"
        label={t('glossary.targetLang')}
        rules={[{ required: true, message: t('glossary.targetLangRequired') }]}
      >
        <Select>
            <Option value="ZH">{t('language.ZH')}</Option>
            <Option value="EN">{t('language.EN')}</Option>
            <Option value="ID">{t('language.ID')}</Option>
        </Select>
      </Form.Item>

      <Form.Item
        name="entries"
        label={t('glossary.entries')}
        rules={[{ required: true, message: t('glossary.entriesRequired') }]}
      >
        <Input.TextArea 
          placeholder={t('glossary.entriesPlaceholder')}
          rows={10}
        />
      </Form.Item>

      <Form.Item>
        <Button type="primary" htmlType="submit" loading={loading}>
          {t('glossary.create')}
        </Button>
      </Form.Item>
    </Form>
  );
};

export default GlossaryEditor;