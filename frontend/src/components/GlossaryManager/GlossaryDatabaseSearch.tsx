// 搜索术语表和词汇明细本地数据库查询
import React, { useState } from 'react';
import { Form, Input, DatePicker, Select, Button, Table, Card, Tag, Tooltip, message, Popconfirm, Space } from 'antd';
import { useTranslation } from 'react-i18next';
import { glossaryApi } from '../../services/api';

const { RangePicker } = DatePicker;
const { Option } = Select;

interface GlossaryEntry {
  id: number;
  glossary_id: number;
  glossary_name: string;
  source_lang: string;
  target_lang: string;
  source_term: string;
  target_term: string;
  created_at: string;
  glossary_created_at: string;
  glossary_updated_at: string | null;
}

interface Glossary {
  id: number;
  name: string;
  source_lang: string;
  target_lang: string;
  created_at: string;
  updated_at: string | null;
  entries: GlossaryEntry[];
}

const GlossaryDatabaseSearch: React.FC = () => {
  const { t, i18n } = useTranslation();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any[]>([]);
  const [form] = Form.useForm();
  const [pagination, setPagination] = useState({
    current: 1,
    pageSize: 10,
    total: 0,
    showSizeChanger: true,
    showTotal: (total: number) => `共 ${total} 条记录`
  });
  const [editingKey, setEditingKey] = useState<string>('');
  const [editForm] = Form.useForm();

  // 格式化日期
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString(i18n.language, {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // 定义基础列
  const baseColumns = [
    {
      title: t('glossary.name'),
      dataIndex: 'glossary_name',
      key: 'glossary_name',
      width: 200,
      fixed: 'left' as const,
      render: (text: string) => (
        <Tooltip title={text}>
          <div style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {text}
          </div>
        </Tooltip>
      )
    },
    {
      title: t('glossary.sourceLang'),
      dataIndex: 'source_lang',
      key: 'source_lang',
      width: 100,
      render: (text: string) => <Tag color="blue">{text}</Tag>
    },
    {
      title: t('glossary.targetLang'),
      dataIndex: 'target_lang',
      key: 'target_lang',
      width: 100,
      render: (text: string) => <Tag color="green">{text}</Tag>
    },
    {
      title: t('glossary.createdAt'),
      dataIndex: 'glossary_created_at',
      key: 'glossary_created_at',
      width: 180,
      render: (text: string) => (
        <Tooltip title={formatDate(text)}>
          {formatDate(text)}
        </Tooltip>
      )
    },
    {
      title: t('glossary.updatedAt'),
      dataIndex: 'glossary_updated_at',
      key: 'glossary_updated_at',
      width: 180,
      render: (text: string | null) => text && (
        <Tooltip title={formatDate(text)}>
          {formatDate(text)}
        </Tooltip>
      )
    },
    {
      title: t('glossary.sourceTerm'),
      dataIndex: 'source_term',
      key: 'source_term',
      width: 200,
      render: (text: string) => (
        <Tooltip title={text}>
          <div style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {text}
          </div>
        </Tooltip>
      )
    },
    {
      title: t('glossary.termCreatedAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (text: string) => (
        <Tooltip title={formatDate(text)}>
          {formatDate(text)}
        </Tooltip>
      )
    }
  ];

  // 定义目标术语列
  const targetTermColumn = {
    title: t('glossary.targetTerm'),
    dataIndex: 'target_term',
    key: 'target_term',
    width: 200,
    render: (text: string, record: any) => {
      const isEditing = record.id.toString() === editingKey;
      return isEditing ? (
        <Form form={editForm}>
          <Form.Item
            name="target_term"
            style={{ margin: 0 }}
            rules={[{ required: true, message: '请输入目标术语' }]}
          >
            <Input />
          </Form.Item>
        </Form>
      ) : (
        <Tooltip title={text}>
          <div style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {text}
          </div>
        </Tooltip>
      );
    }
  };

  // 定义操作列
  const actionColumn = {
    title: t('glossary.actions'),
    key: 'actions',
    fixed: 'right' as const,
    width: 150,
    render: (_: any, record: any) => {
      const isEditing = record.id.toString() === editingKey;
      return isEditing ? (
        <span>
          <Button
            type="link"
            onClick={() => saveEdit(record)}
            style={{ marginRight: 8 }}
          >
            保存
          </Button>
          <Button type="link" onClick={cancelEdit}>
            取消
          </Button>
        </span>
      ) : (
        <span>
          <Button
            type="link"
            onClick={() => startEdit(record)}
            style={{ marginRight: 8 }}
          >
            编辑
          </Button>
          <Popconfirm
            title={t('glossary.delete.confirm')}
            onConfirm={() => handleDelete(record)}
          >
            <Button type="link" danger>
              删除
            </Button>
          </Popconfirm>
        </span>
      );
    }
  };

  // 组合最终的列配置
  const columns = [...baseColumns, targetTermColumn, actionColumn];

  // 添加语言选项配置
  const languageOptions = [
    { value: 'ZH', label: '中文' },
    { value: 'EN', label: 'English' },
    { value: 'ID', label: 'Bahasa Indonesia' }
  ];

  const handleSearch = async (values: any) => {
    try {
      setLoading(true);
      const [startDate, endDate] = values.dateRange || [];
      
      // 添加页码验证逻辑
      let currentPage = pagination.current;
      const result = await glossaryApi.searchGlossaries({
        name: values.name,
        startDate: startDate?.format('YYYY-MM-DD'),
        endDate: endDate?.format('YYYY-MM-DD'),
        sourceLang: values.sourceLang,
        targetLang: values.targetLang,
        page: currentPage,
        pageSize: pagination.pageSize
      });

      // 如果返回的数据为空且不是第一页，则自动请求最后一页
      if (result.entries.length === 0 && currentPage > 1) {
        currentPage = result.total_pages || 1;
        const newResult = await glossaryApi.searchGlossaries({
          name: values.name,
          startDate: startDate?.format('YYYY-MM-DD'),
          endDate: endDate?.format('YYYY-MM-DD'),
          sourceLang: values.sourceLang,
          targetLang: values.targetLang,
          page: currentPage,
          pageSize: pagination.pageSize
        });
        setData(newResult.entries);
        setPagination({
          ...pagination,
          current: currentPage,
          total: newResult.total
        });
      } else {
        setData(result.entries);
        setPagination({
          ...pagination,
          current: result.page,
          total: result.total
        });
      }
    } catch (error) {
      console.error('Search error:', error);
      message.error(t('glossary.search.error') || '查询失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  const handleTableChange = (newPagination: any, filters: any, sorter: any) => {
    setPagination({
      ...pagination,
      current: newPagination.current,
      pageSize: newPagination.pageSize
    });
    
    form.submit();
  };

  const startEdit = (record: any) => {
    editForm.setFieldsValue({ target_term: record.target_term });
    setEditingKey(record.id.toString());
  };

  const cancelEdit = () => {
    setEditingKey('');
  };

  const saveEdit = async (record: any) => {
    try {
      const values = await editForm.validateFields();
      await glossaryApi.updateGlossaryEntry(record.id, values.target_term);
      message.success(t('glossary.edit.success'));
      setEditingKey('');
      
      // 刷新数据
      form.submit();
    } catch (error) {
      console.error('Edit error:', error);
      message.error(t('glossary.edit.error'));
    }
  };

  const handleDelete = async (record: any) => {
    try {
      await glossaryApi.deleteGlossaryEntry(record.id);
      message.success(t('glossary.delete.success'));
      
      // 刷新数据
      form.submit();
    } catch (error) {
      console.error('Delete error:', error);
      message.error(t('glossary.delete.error'));
    }
  };

  // 添加重置搜索条件函数
  const handleReset = () => {
    form.resetFields();
    // 重置后立即执行一次搜索，获取所有数据
    handleSearch(form.getFieldsValue());
  };

  // 添加语言选择变化处理函数
  const handleSourceLangChange = (value: string | null) => {
    const targetLang = form.getFieldValue('targetLang');
    // 如果源语言和目标语言相同，清空目标语言
    if (value && value === targetLang) {
      form.setFieldValue('targetLang', null);
    }
  };

  const handleTargetLangChange = (value: string | null) => {
    const sourceLang = form.getFieldValue('sourceLang');
    // 如果目标语言和源语言相同，清空源语言
    if (value && value === sourceLang) {
      form.setFieldValue('sourceLang', null);
    }
  };

  return (
    <div className="glossary-database-search">
      <Card className="search-form-card">
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSearch}
        >
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
            <Form.Item name="name" label={t('glossary.search.name')}>
              <Input 
                placeholder={t('glossary.search.namePlaceholder')}
                allowClear // 添加清除功能
              />
            </Form.Item>

            <Form.Item name="dateRange" label={t('glossary.search.dateRange')}>
              <RangePicker 
                style={{ width: '100%' }}
                allowClear // 添加清除功能
              />
            </Form.Item>

            <Form.Item 
              name="sourceLang" 
              label={t('glossary.search.sourceLang')}
              tooltip={t('glossary.search.sourceLangTip')} // 添加提示信息
            >
              <Select
                placeholder={t('glossary.search.selectLanguage')}
                allowClear // 添加清除功能
                onChange={handleSourceLangChange}
                options={languageOptions}
                showSearch // 添加搜索功能
                filterOption={(input, option) =>
                  (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                }
              />
            </Form.Item>

            <Form.Item 
              name="targetLang" 
              label={t('glossary.search.targetLang')}
              tooltip={t('glossary.search.targetLangTip')} // 添加提示信息
            >
              <Select
                placeholder={t('glossary.search.selectLanguage')}
                allowClear // 添加清除功能
                onChange={handleTargetLangChange}
                options={languageOptions}
                showSearch // 添加搜索功能
                filterOption={(input, option) =>
                  (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                }
              />
            </Form.Item>
          </div>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                {t('glossary.search.submit')}
              </Button>
              <Button onClick={handleReset}>
                {t('glossary.search.reset')}
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card className="search-results-card">
        <Table
          columns={columns}
          dataSource={data}
          pagination={{
            ...pagination,
            pageSizeOptions: ['10', '20', '50', '100'],
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 条记录`,
            // 添加页码范围验证
            onChange: (page, pageSize) => {
              const maxPage = Math.ceil(pagination.total / pageSize);
              const validPage = Math.min(page, maxPage);
              handleTableChange({ ...pagination, current: validPage, pageSize }, null, null);
            }
          }}
          onChange={handleTableChange}
          loading={loading}
          scroll={{ x: 1700 }}
          rowKey="id"
          className="glossary-results-table"
          size="middle"
        />
      </Card>
    </div>
  );
};

export default GlossaryDatabaseSearch;