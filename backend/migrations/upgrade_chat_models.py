# 升级数据库模型脚本专用  需要   pip install sqlalchemy支持
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import uuid
import logging

# 配置 logger
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 使用与 database.py 相同的数据库 URL 构建方式
DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

print(f"使用数据库连接: {DATABASE_URL}")

# 创建与 database.py 相同配置的引擎
engine = create_engine(
    DATABASE_URL,
    pool_size=int(os.getenv('POSTGRES_POOL_SIZE', 5)),
    max_overflow=int(os.getenv('POSTGRES_MAX_OVERFLOW', 10)),
    pool_timeout=int(os.getenv('POSTGRES_POOL_TIMEOUT', 30)),
    pool_recycle=int(os.getenv('POSTGRES_POOL_RECYCLE', 1800))
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def upgrade():
    """升级数据库模型"""
    session = SessionLocal()
    try:
        # 1. 添加新表
        session.execute(text("""
        -- 创建会话文档关联表
        CREATE TABLE IF NOT EXISTS session_documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
            paper_id UUID NOT NULL,
            added_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            "order" INTEGER DEFAULT 0,
            filename VARCHAR(255),
            CONSTRAINT fk_session FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        );
        
        -- 创建会话主题表
        CREATE TABLE IF NOT EXISTS chat_topics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        
        -- 创建会话主题关联表
        CREATE TABLE IF NOT EXISTS chat_topic_sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            topic_id UUID NOT NULL REFERENCES chat_topics(id) ON DELETE CASCADE,
            session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
            added_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        
        -- 创建聊天文件表
        CREATE TABLE IF NOT EXISTS chat_files (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id),
            session_id UUID NOT NULL REFERENCES chat_sessions(id),
            message_id UUID REFERENCES chat_messages(id),
            filename VARCHAR(255) NOT NULL,
            file_type VARCHAR(50),
            file_size INTEGER,
            file_path VARCHAR(512),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            description TEXT,
            is_processed BOOLEAN DEFAULT FALSE,
            processing_status VARCHAR(20) DEFAULT 'pending'
        );
        """))
        
        # 2. 修改现有表
        session.execute(text("""
        -- 修改 chat_sessions 表
        ALTER TABLE chat_sessions 
            ALTER COLUMN paper_id DROP NOT NULL,
            ADD COLUMN IF NOT EXISTS session_type VARCHAR(20) DEFAULT 'document',
            ADD COLUMN IF NOT EXISTS is_ai_only BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS paper_ids UUID[] DEFAULT '{}';
            
        -- 修改 chat_messages 表
        ALTER TABLE chat_messages
            ADD COLUMN IF NOT EXISTS document_id UUID,
            ADD COLUMN IF NOT EXISTS message_type VARCHAR(20) DEFAULT 'text',
            ADD COLUMN IF NOT EXISTS rich_content JSONB;
            
        -- 修改 paper_analysis 表
        ALTER TABLE paper_analysis
            ADD COLUMN IF NOT EXISTS summary TEXT,
            ADD COLUMN IF NOT EXISTS tags VARCHAR[] DEFAULT '{}',
            ADD COLUMN IF NOT EXISTS embeddings JSONB,
            ADD COLUMN IF NOT EXISTS documents JSONB,
            ADD COLUMN IF NOT EXISTS index_built BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS is_scanned BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS line_mapping JSONB,
            ADD COLUMN IF NOT EXISTS total_lines INTEGER,
            ADD COLUMN IF NOT EXISTS translation_line_mapping JSONB;
            
        -- 修改 users 表
        ALTER TABLE users
            ADD COLUMN IF NOT EXISTS feishu_user_id VARCHAR(255),
            ADD COLUMN IF NOT EXISTS name VARCHAR(255),
            ADD COLUMN IF NOT EXISTS en_name VARCHAR(255),
            ADD COLUMN IF NOT EXISTS email VARCHAR(255),
            ADD COLUMN IF NOT EXISTS mobile VARCHAR(255),
            ADD COLUMN IF NOT EXISTS avatar_url VARCHAR(1024),
            ADD COLUMN IF NOT EXISTS tenant_key VARCHAR(255),
            ADD COLUMN IF NOT EXISTS access_token VARCHAR(255),
            ADD COLUMN IF NOT EXISTS refresh_token VARCHAR(255),
            ADD COLUMN IF NOT EXISTS token_expires_at TIMESTAMP,
            ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE,
            ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP,
            ADD COLUMN IF NOT EXISTS login_count INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT now(),
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT now();
        """))
        
        # 3. 数据迁移：将现有会话的session_type设置为document
        session.execute(text("""
        UPDATE chat_sessions SET session_type = 'document' WHERE paper_id IS NOT NULL;
        """))
        
        # 4. 数据迁移：创建会话-文档关联记录
        session.execute(text("""
        INSERT INTO session_documents (session_id, paper_id, filename, "order")
        SELECT id, paper_id, 
            (SELECT filename FROM paper_analysis WHERE paper_id = chat_sessions.paper_id LIMIT 1),
            0
        FROM chat_sessions
        WHERE paper_id IS NOT NULL
        ON CONFLICT DO NOTHING;
        """))
        
        # 5. 添加约束条件
        session.execute(text("""
        -- 添加唯一约束（如果尚未存在）
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint 
                WHERE conname = 'users_feishu_user_id_key' 
                AND conrelid = 'users'::regclass
            ) THEN
                ALTER TABLE users ADD CONSTRAINT users_feishu_user_id_key UNIQUE (feishu_user_id);
            END IF;
        EXCEPTION WHEN OTHERS THEN
            -- 如果表不存在或其他错误，不执行操作
            NULL;
        END $$;
        """))
        
        # 6. 更新可能缺少的NOT NULL约束
        session.execute(text("""
        -- 确保必要的NOT NULL约束
        DO $$
        BEGIN
            -- 尝试为users表中必须字段添加NOT NULL约束（如果没有）
            ALTER TABLE users ALTER COLUMN feishu_user_id SET NOT NULL;
            ALTER TABLE users ALTER COLUMN name SET NOT NULL;
            ALTER TABLE users ALTER COLUMN tenant_key SET NOT NULL;
        EXCEPTION WHEN OTHERS THEN
            -- 如果约束已存在或其他错误，继续
            NULL;
        END $$;
        """))
        
        session.commit()
        print("数据库升级成功完成!")
        
    except Exception as e:
        session.rollback()
        print(f"数据库升级错误: {str(e)}")
        raise
    finally:
        session.close()
        
if __name__ == "__main__":
    upgrade()
