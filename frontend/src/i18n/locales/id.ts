export default {
  translation: {
    title: 'Cargo Alat Terjemahan',
    useGlossary: 'Gunakan Tabel Istilah',
    sourceLanguage: 'Bahasa Sumber',
    targetLanguage: 'Bahasa Tujuan',
    autoDetect: 'Deteksi Otomatis',
    upload: {
      title: 'Unggah Dokumen',
      drag: 'Seret dan lepas file di sini, atau klik untuk memilih',
      drop: 'Letakkan file di sini',
      selected: 'File yang dipilih: {{filename}}',
      formats: 'Format yang didukung: PDF, DOCX, PPTX',
      largeFileWarning: '（File besar mungkin memerlukan waktu unggah yang lebih lama）',
      fileTooLarge: 'File terlalu besar, maksimal 30MB',
      maxSize: 'Ukuran file maksimum: 30MB',
      invalidType: 'Jenis file tidak valid. Hanya mendukung file PDF, DOCX, dan PPTX',
      warning: 'Peringatan',
      processing: 'Memproses file...'
    },
    language: {
      AUTO: 'Deteksi Otomatis',
      ZH: 'Bahasa Mandarin',
      EN: 'Bahasa Inggris',
      ID: 'Bahasa Indonesia',
      switch: 'Ganti Bahasa',
      zh: 'Bahasa Mandarin',
      en: 'Bahasa Inggris',
      id: 'Bahasa Indonesia'
    },
    status: {
      uploading: 'Mengunggah dokumen...',
      translating: 'Menerjemahkan...',
      downloading: 'Menyiapkan unduhan...',
      completed: 'Terjemahan selesai!',
      autoDownloaded: 'Dokumen telah diunduh secara otomatis.',
      error: 'Kesalahan: {{message}}',
      processing: 'Memproses dokumen...',
      extracting: 'Mengekstrak istilah...',
      creatingGlossary: 'Membuat glosarium...'
    },
    button: {
      translate: 'Terjemahkan Dokumen',
      copy: 'Salin'
    },
    translator: {
      title: 'Layanan Terjemahan',
      select: 'Pilih layanan terjemahan',
      unavailable: '(tidak tersedia)',
      deepl: 'DeepL',
      google: 'Google Translate'
    },
    textTranslate: {
      sourcePlaceholder: "Masukkan teks di sini",
      translatedPlaceholder: "Hasil terjemahan akan muncul di sini",
      translate: "Terjemahkan",
      translating: "Menerjemahkan...",
      characterLimit: "Batas karakter: {count}/3000",
      useAI: "Gunakan AI Terjemahan",
      useDeepL: "Gunakan Terjemahan Biasa"
    },
    footer: {
      allRightsReserved: 'Hak Cipta Dilindungi'
    },
    error: {
        characterLimit: 'Batas karakter terlampaui. Silakan hubungi administrator.',
        timeout: 'Waktu terjemahan habis',
        emptyFile: 'Dokumen yang diterima kosong',
        unexpected: 'Terjadi kesalahan yang tidak diketahui',
        uploadTimeout: 'Upload timeout, please try uploading a smaller file or check your network connection',
        fileTooLarge: 'File terlalu besar. Ukuran maksimum adalah 30MB',
        unsupportedFileType: 'Jenis file tidak didukung. Hanya file PDF, DOCX, dan PPTX yang didukung',
        sourceLanguageRequired: 'Bahasa sumber harus ditentukan saat menggunakan glosarium',
        uploadFailed: 'Gagal mengunggah',
        statusCheckFailed: 'Gagal memeriksa status',
        downloadFailed: 'Gagal mengunduh',
        translationFailed: 'Gagal menerjemahkan'
      },
    or: "atau",
    mode: {
      text: "Terjemahan Teks",
      document: "Terjemahan Dokumen"
    },
    tabs: {
      translation: "Terjemahan",
      glossaryManagement: "Pengelolaan Glosarium"
    },
    glossary: {
      information: "Informasi Glosarium",
      sourceLang: "Bahasa Sumber",
      targetLang: "Bahasa Target",
      nameRequired: "Silakan masukkan nama glosarium",
      sourceLangRequired: "Silakan pilih bahasa sumber",
      targetLangRequired: "Silakan pilih bahasa target",
      entriesRequired: "Silakan masukkan entri glosarium",
      namePlaceholder: "Masukkan nama glosarium",
      entriesPlaceholder: "Masukkan istilah dalam format: istilah_sumber[Tab]istilah_target",
      totalEntries: "Total Entri",
      updatedAt: "Waktu Pembaruan",
      termCreatedAt: "Waktu Penciptaan Istilah",
      languages: "Bahasa",
      actions: "Tindakan",
      entriesNotAvailable: "Entri Tidak Tersedia",
      entriesNotAvailableDesc: "Entri untuk glosarium ini tidak dapat diambil. Informasi glosarium masih tersedia.",
      showTotal: 'Total {{total}} entri',
      itemsPerPage: 'entri/halaman',
      jumpTo: 'Lompat ke',
      jumpToConfirm: 'Konfirmasi',
      page: 'halaman',
      entriesModalTitle: 'Entri Glosarium',
      fetchError: 'Gagal mengambil glosarium',
      deleteError: 'Gagal menghapus glosarium',
      search: {
        name: "Nama Glosarium",
        namePlaceholder: "Masukkan nama glosarium",
        dateRange: "Rentang Tanggal",
        sourceLang: "Bahasa Sumber",
        targetLang: "Bahasa Target",
        selectLanguage: "Pilih Bahasa",
        sourceLangTip: 'Pilih bahasa sumber, dapat membersihkan ulang',
        targetLangTip: 'Pilih bahasa target, dapat membersihkan ulang',
        reset: 'Membersihkan ulang',
        submit: "Cari"
      },
      view: "Lihat",
      delete: "Hapus",
      deleteSuccess: "Berhasil menghapus",
      name: "Nama",
      createdAt: "Tanggal Dibuat",
      entries: "Jumlah Entri"
    },
    download: {
      filename: 'terjemahan_{{filename}}',
      preparing: 'Menyiapkan unduhan...',
      completed: 'Unduhan selesai'
    }
  }
}
