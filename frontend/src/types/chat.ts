export interface SourceInfo {
  line_number: number;
  content: string;
  page: number;
  start_pos: number;
  end_pos: number;
  is_scanned: boolean;
  similarity: number;
}

export interface ChatResponse {
  answer: string;
  sources: SourceInfo[];
  confidence: number;
}

export interface ChatMessage {
  question: string;
  answer: string | ChatResponse;
  timestamp?: string;
}

export interface LineInfo {
  content: string;
  page: number;
  start_pos: number;
  end_pos: number;
  is_scanned?: boolean;
}

export interface LineMapping {
  [key: string]: LineInfo;
}

export interface PaperContent {
  content: string;
  line_mapping: LineMapping;
  total_lines: number;
}

export interface TranslationProgress {
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number;
  content?: string;
  error?: string;
}

export interface PaperAnalysisResponse {
  status: string;
  message: string;
  paper_id?: string;
  content?: string;
  line_mapping?: LineMapping;
  total_lines?: number;
  is_scanned?: boolean;
}

export interface TranslationResponse {
  status: string;
  content?: string;
  message?: string;
  language?: string;
}
