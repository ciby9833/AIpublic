export default {
  translation: {
    title: 'Cargo AI',
    logout: 'Logout',
    loading: 'Loading...',
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
      glossaryManagement: "Glossary Management",
      userManagement: "User Management",
      distanceCalculator: "Distance Calculator",
      paperAnalyzer: "AI Robot & Translation"
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
    },
    mobileChat: {
      title: "Cargo AI Assistant",
      newChat: "New Chat",
      newConversation: "New Conversation",
      chatHistory: "Chat History",
      sessionHistory: "Session History",
      noSessions: "No chat history",
      createNewSession: "Create New Session",
      unnamedSession: "Unnamed Session",
      emptySession: "Empty Session",
      confirmDelete: "Confirm Delete",
      deleteConfirmText: "Are you sure you want to delete this session? This action cannot be undone.",
      delete: "Delete",
      cancel: "Cancel",
      inputPlaceholder: "Type a message...",
      inputPlaceholderWithFile: "Add message (optional)...",
      thinking: "AI is thinking...",
      uploading: "Uploading document...",
      loadingMore: "Loading more messages...",
      copied: "Copied",
      copyMessage: "Copy message",
      copyFailed: "Copy failed, please manually select text to copy",
      uploadFile: "Upload file",
      fileSelected: "File selected: {{filename}}",
      fileSizeLimit: "File size cannot exceed 50MB",
      unsupportedFileType: "Unsupported file type, please select PDF, Word, PowerPoint, Excel, text or Markdown files",
      sessionLimitReached: "Current session has reached the document limit (10), please create a new session",
      createSessionFailed: "Failed to create session",
      uploadFailed: "Document upload failed",
      sendFailed: "Send failed, please try again",
      loadSessionFailed: "Failed to load chat history",
      loadLatestSessionFailed: "Failed to load latest session, created new chat",
      loadSessionListFailed: "Failed to load session list",
      deleteSessionFailed: "Delete failed: {{message}}",
      sessionDeleted: "Session deleted",
      titleUpdated: "Session title updated",
      updateTitleFailed: "Failed to save title: {{message}}",
      loginExpired: "Login information has expired, please log in again",
      loginExpiring: "Login is about to expire, please save important content",
      loginStatusWarning: "Login Status Alert",
      fileAnalysisComplete: "File {{filename}} has been successfully analyzed, you can start asking questions",
      fileUploaded: "Uploaded file: {{filename}}",
      documentAnalysisFailed: "Document analysis failed: {{message}}",
      documentUploadSuccess: "Document uploaded successfully",
      documentAnalysisFailed2: "Document analysis failed",
      dragFileHere: "Drag file here to upload",
      askAnything: "Feel free to ask me anything",
      unknownError: "An unexpected error occurred",
      documentAlreadyInSession: "Document already in session, skip adding step",
      alreadyNewChat: "Current chat is already empty, no need to create a new chat",
      confirm: "Confirm"
    },
    paperAnalyzer: {
      title: "AI Q&A & Translation",
      newChat: "New Chat",
      chatHistory: "Chat History",
      returnToChat: "Return to Current Chat",
      newConversation: "New Conversation",
      sessionTitle: "Session Title",
      unnamedSession: "Unnamed Session",
      editTitle: "Edit Title",
      saveTitle: "Save",
      deleteSession: "Delete Chat",
      confirmDelete: "Confirm Delete",
      deleteConfirmText: "Are you sure you want to delete this chat? This action cannot be undone.",
      delete: "Delete",
      cancel: "Cancel",
      sessionDeleted: "Chat deleted",
      createSessionFailed: "Failed to create chat",
      loadSessionFailed: "Failed to load chat history",
      deleteSessionFailed: "Delete failed: {{message}}",
      titleUpdated: "Session title updated",
      updateTitleFailed: "Failed to save title: {{message}}",
      chatWelcomeTitle: "Welcome to Cargo AI Assistant",
      chatWelcomeDescription: "Please enter your question or share files to chat",  
      uploadDocumentHint: "Please upload a document or wait for analysis to complete",
      // Input area
      inputPlaceholder: "Enter your question or drag files here",
      inputPlaceholderAnalyzing: "Analyzing document, please wait...",
      inputPlaceholderSelectSession: "Select a session to start chatting",
      uploadFile: "Upload file",
      viewSessionDocuments: "View session documents",
      send: "Send",
      analyzing: "Analyzing...",
      thinking: "Thinking...",
      
      // Document analysis
      documentAnalysisComplete: "Document analysis complete",
      documentAnalysisFailed: "Document analysis failed",
      createSessionFailed2: "Failed to create session",
      addDocumentFailed: "Failed to add document to session",
      sessionDocumentLimitReached: "Current session has reached the document limit (10), please create a new session to continue adding documents",
      documentLimitTitle: "Document limit reached",
      documentLimitContent: "Current session already contains 10 documents. Would you like to create a new session?",
      createNewSession: "Create New Session",
      
      // Translation features
      selectTargetLanguage: "Select target language",
      translate: "Translate",
      translating: "Translating...",
      translationComplete: "Translation complete",
      translationFailed: "Translation failed",
      noTranslationContent: "No translation content available for download",
      downloadFailed: "Download failed",
      
      // Document viewer
      originalText: "Original",
      translation: "Translation",
      syncScrolling: "Sync Scrolling",
      independentScrolling: "Independent Scrolling",
      downloadTranslation: "Download Translation",
      uploadDocumentPrompt: "Please upload a document or wait for analysis to complete",
      selectLanguagePrompt: "Please select a language and click translate",
      
      // Download options
      wordDocument: "Word Document",
      pdfDocument: "PDF Document",
      markdown: "Markdown",
      
      // Session list
      noSessionHistory: "No chat history",
      messagesCount: "{{count}} messages",
      aiOnlySession: "AI Chat",
      documentSession: "Document Chat",
      
      // Session documents
      sessionDocuments: "Session Documents",
      noDocuments: "No documents in current session",
      
      // Loading states
      loadingMoreMessages: "Loading more messages...",
      
      // Error messages
      questionRequired: "Please enter a question or upload a file",
      sendFailed: "Failed to ask question",
      processingFailed: "Processing failed",
      loginExpired: "Login has expired, please log in again",
      
      // Success messages
      fileAnalysisComplete: "File {{filename}} has been successfully analyzed, you can start asking questions"
    },
    chatMessage: {
      referenceSources: "Reference Sources",
      sourcesCount: "{{count}} sources",
      unknownSource: "Unknown source",
      copied: "Copied",
      copyMessage: "Copy message",
      copySuccess: "Copied to clipboard",
      copyFailed: "Copy failed",
      chartRenderFailed: "Chart rendering failed"
    }
  }
}
