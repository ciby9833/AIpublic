// frontend/src/components/TextTranslate.tsx   文本翻译
import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import './TextTranslate.css';
import { translateText, translateMultilingual } from '../services/api';
import { debounce } from 'lodash';
import { Switch } from 'antd';  // 添加 Switch 组件导入

interface TextTranslateProps {
  disabled?: boolean;
}

interface Language {
  code: string;
  name: string;
  autoDetect?: boolean;
}

interface AITranslationResult {
  status: string;
  translations: {
    detected_language: string;
    english: string;
    chinese: string;
    indonesian: string;
  };
}

const LANGUAGES: Language[] = [
  { code: 'AUTO', name: '自动检测', autoDetect: true },
  { code: 'ID', name: '印尼语' },
  { code: 'ZH', name: '中文' },
  { code: 'EN', name: '英语' },
];

// 修改常量配置
const DEBOUNCE_DELAY = 500; // 修改为500毫秒延迟
const MAX_CHARACTERS = 3000;

// 添加工具函数处理空白字符
const removeExtraWhitespace = (text: string) => {
  return text.trim().replace(/\s+/g, ' ');
};

// 修改工具函数，添加检查最后一个字符是否为空格
const isLastCharSpace = (text: string) => {
  return text.charAt(text.length - 1) === ' ';
};

// 添加输入验证工具函数
const isValidInput = (text: string): boolean => {
  // 如果是空或者只包含空白字符
  if (!text || !text.trim()) {
    return false;
  }

  // 检查是否只包含特殊字符
  const specialCharsOnly = /^[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>/?]*$/;
  if (specialCharsOnly.test(text.trim())) {
    return false;
  }

  // 至少包含一个有效字符（字母、数字、中文等）
  const hasValidChars = /[a-zA-Z0-9\u4e00-\u9fa5]/;
  return hasValidChars.test(text);
};

// 添加缓存接口定义
interface TranslationCache {
  text: string;
  targetLang: string;
  result: string;
}

const TextTranslate = ({ disabled }: TextTranslateProps) => {
  const { t } = useTranslation();
  const [sourceText, setSourceText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [error, setError] = useState('');
  const [sourceLang, setSourceLang] = useState<string>(
    localStorage.getItem('textSourceLang') || 'AUTO'
  );
  const [targetLang, setTargetLang] = useState<string>(
    localStorage.getItem('textTargetLang') || 'ID'
  );
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [copySuccess, setCopySuccess] = useState(false);
  const [isComposing, setIsComposing] = useState(false);
  const [lastTranslatedText, setLastTranslatedText] = useState('');
  const [lastValidText, setLastValidText] = useState('');
  const [pendingText, setPendingText] = useState('');

  // 添加翻译缓存状态
  const [translationCache, setTranslationCache] = useState<TranslationCache[]>([]);
  const MAX_CACHE_SIZE = 50; // 最大缓存数量

  // 添加 AI 翻译开关状态
  const [useAI, setUseAI] = useState<boolean>(true);  // 默认启用 AI

  // 添加缓存查找函数
  const findInCache = (text: string, targetLang: string): string | null => {
    const cached = translationCache.find(
      item => item.text === text && item.targetLang === targetLang
    );
    return cached ? cached.result : null;
  };

  // 添加缓存更新函数
  const updateCache = (text: string, targetLang: string, result: string) => {
    setTranslationCache(prevCache => {
      // 检查是否已存在相同的条目
      const existingIndex = prevCache.findIndex(
        item => item.text === text && item.targetLang === targetLang
      );

      if (existingIndex !== -1) {
        // 如果存在，更新现有条目
        const newCache = [...prevCache];
        newCache[existingIndex] = { text, targetLang, result };
        return newCache;
      }

      // 如果不存在，添加新条目
      const newCache = [{ text, targetLang, result }, ...prevCache];
      
      // 如果超出最大缓存数量，删除最旧的条目
      if (newCache.length > MAX_CACHE_SIZE) {
        newCache.pop();
      }

      return newCache;
    });
  };

  // 修改自动翻译函数
  const debouncedTranslate = useCallback(
    debounce(async (text: string, targetLang: string) => {
      const processedText = removeExtraWhitespace(text);
      if (!isValidInput(processedText) || disabled) {
        setTranslatedText('');
        setLastValidText('');
        return;
      }

      try {
        setIsTranslating(true);
        setError('');

        if (useAI) {
          // AI 翻译模式
          const result = await translateMultilingual(processedText);
          
          // 语言代码映射
          const langMap: Record<string, keyof typeof result.translations> = {
            'en': 'english',
            'zh': 'chinese',
            'id': 'indonesian'
          };
          
          const targetLangKey = langMap[targetLang.toLowerCase()];
          if (!targetLangKey) {
            throw new Error(`Unsupported target language: ${targetLang}`);
          }
          
          const translatedText = result.translations[targetLangKey];
          if (!translatedText) {
            throw new Error(`No translation available for ${targetLang}`);
          }
          
          // 更新缓存和状态
          updateCache(processedText, targetLang, translatedText);
          setTranslatedText(translatedText);
          setLastValidText(processedText);
          
        } else {
          // 原有的 DeepL 翻译逻辑
          const response = await translateText(processedText, targetLang);
          if (!response.ok) {
            throw new Error(await response.text());
          }

          const result = await response.json();
          const translatedText = result.translations[0].text;
          
          updateCache(processedText, targetLang, translatedText);
          setTranslatedText(translatedText);
          setLastValidText(processedText);
        }
      } catch (error) {
        setError(t('error.translationFailed'));
        console.error('Translation error:', error);
      } finally {
        setIsTranslating(false);
      }
    }, DEBOUNCE_DELAY),
    [disabled, useAI, t]
  );

  // 修改立即翻译函数
  const translateImmediate = useCallback(async (text: string, targetLang: string) => {
    if (!text.trim() || disabled) {
      setTranslatedText('');
      return;
    }

    try {
      setIsTranslating(true);
      setError('');

      if (useAI) {
        // 使用 AI 多语言翻译
        const result = await translateMultilingual(text);
        
        // 根据目标语言选择对应的翻译结果
        let translatedText = '';
        switch (targetLang.toLowerCase()) {
          case 'en':
            translatedText = result.translations.english;
            break;
          case 'zh':
            translatedText = result.translations.chinese;
            break;
          case 'id':
            translatedText = result.translations.indonesian;
            break;
          default:
            translatedText = '';
        }
        
        setTranslatedText(translatedText);
      } else {
        // 使用原有的翻译方法
        const response = await translateText(text, targetLang);
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Translation failed');
        }
        const result = await response.json();
        setTranslatedText(result.translations[0].text);
      }
    } catch (error) {
      setError(t('error.translationFailed'));
      console.error('Translation error:', error);
    } finally {
      setIsTranslating(false);
    }
  }, [disabled, useAI, t]);

  // 修改文本变化处理函数
  const handleTextChange = (text: string) => {
    setSourceText(text);
    
    // 更新历史记录
    const newHistory = [...history.slice(0, historyIndex + 1), text];
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
    
    // 添加输入验证
    if (!isValidInput(text)) {
      setTranslatedText('');
      setLastValidText('');
      return;
    }

    // 如果是空格结尾，不触发翻译
    if (isLastCharSpace(text)) return;

    // 调用防抖函数
    debouncedTranslate(text, targetLang);
  };

  // 修改清除函数，添加缓存清理选项
  const handleClear = () => {
    setSourceText('');
    setTranslatedText('');
    setLastTranslatedText('');
    setLastValidText('');
    setPendingText('');
    setHistory([]);
    setHistoryIndex(-1);
    setError('');
    // 可选：清除翻译缓存
    // setTranslationCache([]);
  };

  // 修改撤销/重做函数
  const handleUndo = () => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      const newText = history[newIndex];
      setHistoryIndex(newIndex);
      setSourceText(newText);
      
      if (newText.trim() && !isLastCharSpace(newText)) {
        debouncedTranslate(newText, targetLang);
      }
    }
  };

  const handleRedo = () => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      const newText = history[newIndex];
      setHistoryIndex(newIndex);
      setSourceText(newText);
      
      if (newText.trim() && !isLastCharSpace(newText)) {
        debouncedTranslate(newText, targetLang);
      }
    }
  };

  // 修改语言切换函数
  const handleSwapLanguages = () => {
    if (!translatedText) return;

    // 如果使用 AI 翻译，不允许将源语言设置为 'AUTO'
    const newSourceLang = useAI && targetLang === 'AUTO' ? 'ZH' : targetLang;
    const newTargetLang = sourceLang === 'AUTO' 
      ? LANGUAGES.find(lang => !lang.autoDetect)?.code || 'ZH' 
      : sourceLang;
    
    // 保存当前的翻译结果作为新的源文本
    const newSourceText = translatedText;
    
    // 更新状态
    setSourceLang(newSourceLang);
    setTargetLang(newTargetLang);
    setSourceText(newSourceText);
    
    // 使用统一的翻译逻辑
    debouncedTranslate(newSourceText, newTargetLang);
  };

  // 添加复制函数
  const handleCopy = async () => {
    if (!translatedText) return;
    
    try {
      await navigator.clipboard.writeText(translatedText);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 1000);
    } catch (error) {
      setError(t('error.copyFailed'));
    }
  };

  // 修改键盘快捷键处理
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Ctrl+Enter 触发翻译
    if (e.ctrlKey && e.key === 'Enter') {
      // 添加输入验证
      if (isValidInput(sourceText)) {
        debouncedTranslate(sourceText, targetLang);
      }
    }
    
    // Ctrl+C 复制翻译结果（当焦点在翻译结果框时）
    if (e.ctrlKey && e.key === 'c' && document.activeElement?.classList.contains('translated')) {
      e.preventDefault();
      handleCopy();
    }
  }, [sourceText, debouncedTranslate, handleCopy]);

  // 添加键盘事件监听
  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

  // 修改语言切换时的翻译处理
  useEffect(() => {
    if (sourceText.trim() && !isLastCharSpace(sourceText)) {
      // 语言切换时立即翻译
      translateImmediate(sourceText, targetLang);
    }
  }, [targetLang, translateImmediate]);

  // 修改输入法完成事件处理
  const handleCompositionEnd = (e: React.CompositionEvent<HTMLTextAreaElement>) => {
    setIsComposing(false);
    const text = e.currentTarget.value;
    // 添加输入验证
    if (isValidInput(text) && !isLastCharSpace(text)) {
      debouncedTranslate(text, targetLang);
    }
  };

  // 添加语言选择持久化
  useEffect(() => {
    localStorage.setItem('textSourceLang', sourceLang);
  }, [sourceLang]);

  useEffect(() => {
    localStorage.setItem('textTargetLang', targetLang);
  }, [targetLang]);

  // 修改语言选择处理函数
  const handleSourceLangChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newLang = e.target.value;
    setSourceLang(newLang);
    localStorage.setItem('textSourceLang', newLang);
    
    // 如果有文本，立即触发翻译
    if (sourceText.trim() && !isLastCharSpace(sourceText)) {
      translateImmediate(sourceText, targetLang);
    }
  };

  // 修改目标语言选择处理函数
  const handleTargetLangChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newLang = e.target.value;
    setTargetLang(newLang);
    localStorage.setItem('textTargetLang', newLang);
    
    // 如果有文本，立即触发翻译
    if (sourceText.trim() && !isLastCharSpace(sourceText)) {
      translateImmediate(sourceText, newLang);
    }
  };

  return (
    <div className="text-translate-container">
      <div className="text-area-wrapper">
        <div className="text-area-container">
          {/* 翻译选项和语言选择放在一起 */}
          <div className="translation-header">
            <div className="translation-options">
              <Switch
                checked={useAI}
                onChange={(checked) => setUseAI(checked)}
                disabled={disabled}
                checkedChildren={t('textTranslate.useAI')}
                unCheckedChildren={t('textTranslate.useDeepL')}
              />
            </div>
            <div className="language-select-group">
              <select
                value={sourceLang}
                onChange={handleSourceLangChange}
                disabled={disabled || isTranslating}
              >
                {LANGUAGES.filter(lang => !useAI || !lang.autoDetect).map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.autoDetect ? `${lang.name}` : lang.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
          
          <textarea
            className="text-area source"
            placeholder={t('textTranslate.sourcePlaceholder')}
            value={sourceText}
            onChange={(e) => handleTextChange(e.target.value)}
            onCompositionStart={() => setIsComposing(true)}
            onCompositionEnd={handleCompositionEnd}
            disabled={disabled}
          />
          <div className="text-area-controls">
            <button
              className="icon-button"
              onClick={() => handleClear()}
              disabled={!sourceText}
              title={t('button.clear')}
            >
              <span className="material-icons">close</span>
            </button>
            <button
              className="icon-button"
              onClick={handleUndo}
              disabled={historyIndex <= 0}
              title={t('button.undo')}
            >
              <span className="material-icons">undo</span>
            </button>
            <button
              className="icon-button"
              onClick={handleRedo}
              disabled={historyIndex >= history.length - 1}
              title={t('button.redo')}
            >
              <span className="material-icons">redo</span>
            </button>
          </div>
          <div className="character-count">
            {sourceText.length} / 3000
          </div>
          <div className="shortcut-hint">
            Ctrl + Enter {t('textTranslate.shortcutTranslate')}
          </div>
        </div>

        <div className="text-area-container">
          {/* 目标语言选择器 */}
          <div className="language-select-group target">
            <button
              className="swap-button"
              onClick={handleSwapLanguages}
              disabled={disabled || isTranslating || !translatedText}
              title={t('button.swap')}
            >
              <span className="material-icons">swap_horiz</span>
            </button>
            <select
              value={targetLang}
              onChange={handleTargetLangChange}
              disabled={disabled || isTranslating}
            >
              {LANGUAGES.filter(lang => !lang.autoDetect).map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.name}
                </option>
              ))}
            </select>
          </div>
          <textarea
            className="text-area translated"
            placeholder={t('textTranslate.translatedPlaceholder')}
            value={translatedText}
            readOnly
          />
          <div className="translated-controls">
            <button
              className={`icon-button ${copySuccess ? 'copy-success' : ''}`}
              onClick={handleCopy}
              disabled={!translatedText}
              title={`${t('button.copy')} (Ctrl+C)`}
            >
              <span className="material-icons">
                {copySuccess ? 'check' : 'content_copy'}
              </span>
            </button>
          </div>
          <div className="shortcut-hint">
            Ctrl + C {t('textTranslate.shortcutCopy')}
          </div>
        </div>
      </div>
      
      <div className="controls">
        {error && <div className="error-message">{error}</div>}
        {isTranslating && <div className="translating-message">{t('textTranslate.translating')}</div>}
      </div>
    </div>
  );
};

export default TextTranslate;
