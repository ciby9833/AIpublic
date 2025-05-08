export default {
  translation: {
    title: 'Cargo Translation Tool',
    useGlossary: 'Use Glossary',
    sourceLanguage: 'Source Language',
    targetLanguage: 'Target Language',
    autoDetect: 'Auto Detect',
    upload: {
      title: 'Upload Document',
      drag: 'Drag and drop a file here, or click to select',
      drop: 'Drop the file here',
      selected: 'Selected file: {{filename}}',
      formats: 'Supported formats: PDF, DOCX, PPTX',
      largeFileWarning: '（Large files may require longer upload times）',
      fileTooLarge: 'File too large, maximum support 30MB',
      maxSize: 'Maximum file size: 30MB',
      invalidType: 'Invalid file type. Only PDF, DOCX, and PPTX files are supported',
      warning: 'Warning',
      processing: 'Processing file...'
    },
    language: {
      AUTO: 'Auto Detect',
      ZH: 'Chinese',
      EN: 'English',
      ID: 'Indonesian',
      switch: 'Switch Language',
      zh: 'Chinese',
      en: 'English',
      id: 'Bahasa Indonesia'
    },
    status: {
      uploading: 'Uploading document...',
      translating: 'Translating...',
      downloading: 'Preparing download...',
      completed: 'Translation completed!',
      autoDownloaded: 'File has been automatically downloaded.',
      error: 'Error: {{message}}',
      processing: 'Processing document...',
      extracting: 'Extracting terms...',
      creatingGlossary: 'Creating glossary...'
    },
    button: {
      translate: 'Translate Document',
      copy: 'Copy'
    },
    translator: {
      title: 'Translation Service',
      select: 'Select translation service',
      unavailable: '(unavailable)',
      deepl: 'DeepL',
      google: 'Google Translate'
    },
    textTranslate: {
      sourcePlaceholder: "Enter text here",
      translatedPlaceholder: "Translation will appear here",
      translate: "Translate",
      translating: "Translating...",
      characterLimit: "Character limit: {count}/3000",
      useAI: "Use AI Translation",
      useDeepL: "Use Normal Translation"
    },
    footer: {
      allRightsReserved: 'All rights reserved'
    },
    error: {
        characterLimit: 'Monthly translation limit reached. Please contact administrator.',
        timeout: 'Translation timeout',
        emptyFile: 'Received empty file',
        uploadTimeout: 'Upload timeout, please try uploading a smaller file or check your network connection',
        fileTooLarge: 'File is too large. Maximum size is 30MB',
        unexpected: 'An unexpected error occurred',
        unsupportedFileType: 'Unsupported file type. Only PDF, DOCX, and PPTX files are supported',
        sourceLanguageRequired: 'Source language must be specified when using glossaries',
        uploadFailed: 'Upload failed',
        statusCheckFailed: 'Status check failed',
        downloadFailed: 'Download failed',
        translationFailed: 'Translation failed'
      },
    or: "or",
    mode: {
      text: "Text Translation",
      document: "Document Translation"
    },
    tabs: {
      translation: "Translation",
      glossaryManagement: "Glossary Management"
    },
    glossary: {
      list: "Glossary List",
      create: "Create Glossary",
      createSuccess: "Glossary created successfully",
      deleteSuccess: "Glossary deleted successfully",
      viewEntries: "View Entries",
      delete: "Delete",
      name: "Name",
      createdAt: "Created At",
      dictionaries: "Dictionaries",
      sourceTerm: "Source Term",
      targetTerm: "Target Term",
      entries: "Entries",
      information: "Glossary Information",
      sourceLang: "Source Language",
      targetLang: "Target Language",
      nameRequired: "Please enter glossary name",
      sourceLangRequired: "Please select source language",
      targetLangRequired: "Please select target language",
      entriesRequired: "Please enter glossary entries",
      namePlaceholder: "Enter glossary name",
      entriesPlaceholder: "Enter terms in format: source_term[Tab]target_term",
      totalEntries: "Total Entries",
      updatedAt: "Updated At",
      termCreatedAt: "Term Created At",
      languages: "Languages",
      actions: "Actions",
      entriesNotAvailable: "Entries Not Available",
      entriesNotAvailableDesc: "The entries for this glossary could not be retrieved. The glossary information is still available.",
      showTotal: 'Total {{total}} entries',
      itemsPerPage: 'entries/page',
      jumpTo: 'Jump to',
      jumpToConfirm: 'Confirm',
      page: 'page',
      entriesModalTitle: 'Glossary Entries',
      fetchError: 'Failed to fetch glossaries',
      deleteError: 'Failed to delete glossary',
      search: {
        name: "Glossary Name",
        namePlaceholder: "Enter glossary name",
        dateRange: "Date Range",
        sourceLang: "Source Language",
        targetLang: "Target Language",
        selectLanguage: "Select Language",
        sourceLangTip: 'Select source language, can reset',
        targetLangTip: 'Select target language, can reset',
        reset: 'Reset',
        submit: "Search"
      },
    },
    download: {
      filename: 'translated_{{filename}}',
      preparing: 'Preparing download...',
      completed: 'Download completed'
    }
  }
}
