export default {
  translation: {
    title: 'Cargo AI',
    logout: '退出登录',
    loading: '加载中...',
    useGlossary: '使用术语表',
    sourceLanguage: '源语言',  // 'Source Language' / 'Bahasa Sumber'
    targetLanguage: '目标语言', 
    upload: {
      title: '上传文档',
      drag: '拖拽文件到此处，或点击选择文件',
      drop: '将文件放在这里',
      selected: '已选择文件：{{filename}}',
      formats: '支持的格式：PDF、DOCX、PPTX',
      fileTooLarge: '文件太大，最大支持30MB',
      largeFileWarning: '（大文件可能需要较长上传时间）',
      maxSize: '最大文件大小：30MB',
      invalidType: '不支持的文件类型，仅支持PDF、DOCX和PPTX文件',
      warning: '警告',
      processing: '正在处理文件...'
    },
    language: {
      AUTO: '自动检测',
      ZH: '中文',
      EN: '英文',
      ID: '印尼文',
      switch: '切换语言',
      zh: '中文',
      en: '英文',
      id: '印尼文'
    },
    status: {
      uploading: '正在上传文档...',
      translating: '正在翻译...',
      downloading: '准备下载...',
      completed: '翻译完成！',
      autoDownloaded: '文件已自动下载。',
      error: '错误：{{message}}',
      processing: '正在处理文档...',
      extracting: '正在提取术语...',
      creatingGlossary: '正在创建术语表...'
    },
    button: {
      translate: '文档翻译',
      copy: '复制'
    },
    translator: {
        title: '翻译服务',
        select: '选择翻译服务',
        unavailable: '(不可用)',
        deepl: 'DeepL',
        google: '谷歌翻译'
      },
    textTranslate: {
      sourcePlaceholder: "在此输入文本",
      translatedPlaceholder: "翻译结果将显示在这里",
      translate: "翻译",
      translating: "正在翻译...",
      characterLimit: "字符数限制：{count}/3000",
      useAI: "使用AI翻译",
      useDeepL: "使用普通翻译"
    },
    footer: {
        allRightsReserved: '版权所有'
      },
    error: {
        characterLimit: '本月翻译字数已达上限，请联系管理员',
        timeout: '翻译超时',
        emptyFile: '收到的文件为空',
        uploadTimeout: '上传超时，请尝试上传小一点的文件或检查网络连接',
        fileTooLarge: '文件太大，最大支持30MB',
        unexpected: '发生未知错误',
        unsupportedFileType: '不支持的文件类型，仅支持PDF、DOCX和PPTX文件',
        sourceLanguageRequired: '使用术语表时必须指定源语言',
        uploadFailed: '上传失败',
        statusCheckFailed: '状态检查失败',
        downloadFailed: '下载失败',
        translationFailed: '翻译失败'
      },
    or: "或",
    mode: {
      text: "文本翻译",
      document: "文档翻译"
    },
    tabs: {
      translation: "翻译",
      glossaryManagement: "术语表管理",
      userManagement: "用户管理",
      distanceCalculator: "距离计算",
      paperAnalyzer: "AI机器人&翻译"
    },
    glossary: {
      information: "术语表信息",
      sourceLang: "源语言",
      targetLang: "目标语言",
      nameRequired: "请输入术语表名称",
      sourceLangRequired: "请选择源语言",
      targetLangRequired: "请选择目标语言",
      entriesRequired: "请输入术语条目",
      namePlaceholder: "输入术语表名称",
      entriesPlaceholder: "按格式输入术语：源术语[Tab键]目标术语",
      totalEntries: "总条目数",
      updatedAt: "更新时间",
      termCreatedAt: "词汇创建时间",
      languages: "语言",
      actions: "操作",
      entriesNotAvailable: "条目不可用",
      entriesNotAvailableDesc: "无法获取此术语表的条目。术语表基本信息仍然可用。",
      showTotal: '共 {{total}} 条',  // 'Total {{total}} entries' / 'Total {{total}} entri'
      itemsPerPage: '条/页',        // 'items/page' / 'item/halaman'
      jumpTo: '跳转至',            // 'Jump to' / 'Lompat ke'
      jumpToConfirm: '确定',       // 'Confirm' / 'Konfirmasi'
      page: '页',                 // 'page' / 'halaman'
      entriesModalTitle: '术语表条目',  // 'Glossary Entries' / 'Entri Glosarium'
     fetchError: '获取术语表失败',     // 'Failed to fetch glossaries' / 'Gagal mengambil glosarium'
      deleteError: '删除术语表失败',     // 'Failed to delete glossary' / 'Gagal menghapus glosarium'
      search: {
        name: "术语表名称",
        namePlaceholder: "请输入术语表名称",
        dateRange: "创建时间范围",
        sourceLang: "源语言",
        targetLang: "目标语言",
        selectLanguage: "请选择语言",
        sourceLangTip: '选择源语言，可清除重置',
        targetLangTip: '选择目标语言，可清除重置',
        reset: '重置',
        submit: "查询"
      },
      view: "查看",
      delete: "删除",
      deleteSuccess: "删除成功",
      name: "名称",
      createdAt: "创建时间",
      entries: "词条数量",
      databaseSearch: "术语库查询",
      noData: "暂无数据",
      sourceTermLabel: "源词条",
      targetTermLabel: "目标词条"
    },
    download: {
      filename: '已翻译_{{filename}}',
      preparing: '准备下载...',
      completed: '下载完成'
    },
    user: {
      management: "用户管理",
      name: "用户名",
      email: "邮箱",
      lastLogin: "最后登录时间",
      status: "状态", 
      searchPlaceholder: "搜索用户..."
    },
    mobileChat: {
      title: "Cargo AI助手",
      newChat: "新对话",
      newConversation: "新建对话",
      chatHistory: "历史会话",
      sessionHistory: "会话历史",
      noSessions: "暂无会话历史",
      createNewSession: "创建新会话",
      unnamedSession: "未命名会话",
      emptySession: "空会话",
      confirmDelete: "确认删除",
      deleteConfirmText: "确定要删除这个会话吗？此操作不可恢复。",
      delete: "删除",
      cancel: "取消",
      inputPlaceholder: "输入消息...",
      inputPlaceholderWithFile: "添加消息（可选）...",
      thinking: "AI正在思考...",
      uploading: "正在上传文档...",
      loadingMore: "加载更多消息...",
      copied: "已复制",
      copyMessage: "复制消息",
      copyFailed: "复制失败，请手动选择文本复制",
      uploadFile: "上传文件",
      fileSelected: "已选择文件: {{filename}}",
      fileSizeLimit: "文件大小不能超过50MB",
      unsupportedFileType: "不支持的文件类型，请选择PDF、Word、PowerPoint、Excel、文本或Markdown文件",
      sessionLimitReached: "当前会话文档数量已达到上限（10个），请创建新会话",
      createSessionFailed: "创建会话失败",
      uploadFailed: "文档上传失败",
      sendFailed: "发送失败，请重试",
      loadSessionFailed: "加载对话历史失败",
      loadLatestSessionFailed: "加载最新会话失败，已创建新对话",
      loadSessionListFailed: "加载会话列表失败",
      deleteSessionFailed: "删除失败: {{message}}",
      sessionDeleted: "会话已删除",
      titleUpdated: "会话标题已更新",
      updateTitleFailed: "保存标题失败: {{message}}",
      loginExpired: "登录信息已失效，请重新登录",
      loginExpiring: "登录即将过期，请保存重要内容",
      loginStatusWarning: "登录状态提示",
      fileAnalysisComplete: "文件 {{filename}} 已成功分析，可以开始提问了",
      fileUploaded: "上传了文件: {{filename}}",
      documentAnalysisFailed: "文档分析失败: {{message}}",
      documentUploadSuccess: "文档上传成功",
      documentAnalysisFailed2: "文档分析失败",
      dragFileHere: "拖拽文件到此处上传",
      askAnything: "有任何问题都可以向我提问",
      unknownError: "未知错误",
      documentAlreadyInSession: "文档已在会话中，跳过添加步骤",
      alreadyNewChat: "当前已经是空白新会话，无需创建",
      confirm: "确定"
    },
    paperAnalyzer: {
      title: "AI问答&翻译",
      newChat: "新对话",
      chatHistory: "历史对话",
      returnToChat: "返回当前对话",
      newConversation: "新对话",
      sessionTitle: "会话标题",
      unnamedSession: "未命名会话",
      editTitle: "编辑标题",
      saveTitle: "保存",
      deleteSession: "删除对话",
      confirmDelete: "确认删除",
      deleteConfirmText: "确定要删除这个对话吗？此操作不可恢复。",
      delete: "删除",
      cancel: "取消",
      sessionDeleted: "对话已删除",
      createSessionFailed: "创建对话失败",
      loadSessionFailed: "加载对话历史失败",
      deleteSessionFailed: "删除失败: {{message}}",
      titleUpdated: "会话标题已更新",
      updateTitleFailed: "保存标题失败: {{message}}",
      
      // 输入区域
      inputPlaceholder: "请输入您的问题或拖拽文件到此处",
      inputPlaceholderAnalyzing: "正在分析文档，请稍候...",
      inputPlaceholderSelectSession: "选择一个会话开始聊天",
      uploadFile: "上传文件",
      viewSessionDocuments: "查看会话文档",
      send: "发送",
      analyzing: "分析中...",
      thinking: "正在思考...",
      
      // 文档分析
      documentAnalysisComplete: "文档分析完成",
      documentAnalysisFailed: "文档分析失败",
      createSessionFailed2: "创建会话失败",
      addDocumentFailed: "添加文档到会话失败",
      sessionDocumentLimitReached: "当前会话文档数量已达到上限（10个），请创建新会话继续添加文档",
      documentLimitTitle: "文档数量已达到上限",
      documentLimitContent: "当前会话已包含10个文档，是否创建一个新会话？",
      createNewSession: "创建新会话",
      
      // 翻译功能
      selectTargetLanguage: "选择目标语言",
      translate: "翻译",
      translating: "正在翻译...",
      translationComplete: "翻译完成",
      translationFailed: "翻译失败",
      noTranslationContent: "没有可下载的翻译内容",
      downloadFailed: "下载失败",
      
      // 文档查看器
      originalText: "原文",
      translation: "翻译",
      syncScrolling: "同步滚动",
      independentScrolling: "独立滚动",
      downloadTranslation: "下载翻译",
      uploadDocumentPrompt: "请上传文档或等待分析完成",
      selectLanguagePrompt: "请选择语言并点击翻译按钮",
      
      // 下载选项
      wordDocument: "Word 文档",
      pdfDocument: "PDF 文档",
      markdown: "Markdown",
      
      // 会话列表
      noSessionHistory: "暂无对话历史",
      messagesCount: "{{count}}条消息",
      aiOnlySession: "AI对话",
      documentSession: "文档对话",
      
      // 会话文档
      sessionDocuments: "会话文档",
      noDocuments: "当前会话没有文档",
      
      // 加载状态
      loadingMoreMessages: "加载更多消息...",
      
      // 错误消息
      questionRequired: "请输入问题或上传文件",
      sendFailed: "提问失败",
      processingFailed: "处理失败",
      loginExpired: "登录已过期，请重新登录",
      
      // 成功消息
      fileAnalysisComplete: "文件 {{filename}} 已成功分析，可以开始提问了"
    },
    chatMessage: {
      referenceSources: "参考来源",
      sourcesCount: "{{count}}个",
      unknownSource: "未知来源",
      copied: "已复制",
      copyMessage: "复制消息",
      copySuccess: "已复制到剪贴板",
      copyFailed: "复制失败",
      chartRenderFailed: "图表渲染失败"
    }
  }
}
