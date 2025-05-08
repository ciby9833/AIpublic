export default {
  translation: {
    title: 'Cargo翻译神器',
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
      distanceCalculator: "距离计算"
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
    }
  }
}
