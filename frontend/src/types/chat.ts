export interface SourceInfo {
  line_number: number;
  content: string;
  page: number;
  start_pos: number;
  end_pos: number;
  is_scanned: boolean;
  similarity: number;
  document_id?: string;
  document_name?: string;
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
  sources?: SourceInfo[];
  confidence?: number;
  message_type?: string;
  reply?: RichContentItem[];
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

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  paper_id?: string;
  paper_ids?: string[];
  is_ai_only?: boolean;
  session_type?: string;
  message_count: number;
  last_message: string;
  documents?: SessionDocument[];
}

export interface SessionDocument {
  id: string;
  paper_id: string;
  filename: string;
  order: number;
}

export interface ChatHistoryResponse {
  messages: ChatMessage[];
  session: ChatSession;
}

export interface ServerMessage {
  id: string;
  role: string;
  content: string;
  created_at: string;
  sources?: SourceInfo[];
  confidence?: number;
  message_type?: string;
}

export interface CreateSessionRequest {
  title?: string;
  paper_ids?: string[];
  is_ai_only?: boolean;
}

export interface DocumentRequest {
  paper_id: string;
}

// Rich content types for messages
export type MessageContentType = 'text' | 'markdown' | 'code' | 'table' | 'image';

export interface RichContentItem {
  type: MessageContentType;
  content: string;
  language?: string;  // For code blocks
  alt?: string;       // For images
  url?: string;       // For images
  columns?: string[]; // For tables
  rows?: any[][];     // For tables
  metadata?: Record<string, any>; // Additional metadata
}
