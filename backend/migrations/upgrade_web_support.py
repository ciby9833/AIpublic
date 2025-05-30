"""
数据库迁移脚本 - 添加网页支持功能 (生产环境修复版)
执行命令: python migrations/upgrade_web_support_fixed.py
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime

# 获取数据库连接
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:xiaotao4vip@localhost:5432/translation")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def upgrade():
    """升级数据库模型 - 添加网页支持"""
    session = SessionLocal()
    
    try:
        print(f"[{datetime.now()}] 开始升级数据库以支持网页功能...")
        
        # 1. 扩展 paper_analysis 表 - 添加网页支持字段
        print("步骤 1: 扩展 paper_analysis 表...")
        session.execute(text("""
        ALTER TABLE paper_analysis
            ADD COLUMN IF NOT EXISTS source_url VARCHAR(2048),
            ADD COLUMN IF NOT EXISTS web_metadata JSONB,
            ADD COLUMN IF NOT EXISTS web_links JSONB,
            ADD COLUMN IF NOT EXISTS web_images JSONB,
            ADD COLUMN IF NOT EXISTS fetch_time TIMESTAMP WITH TIME ZONE,
            ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);
        
        -- 更新file_type字段长度以支持'web'类型
        ALTER TABLE paper_analysis 
            ALTER COLUMN file_type TYPE VARCHAR(20);
        """))
        
        # 2. 扩展 session_documents 表 - 支持网页类型
        print("步骤 2: 扩展 session_documents 表...")
        session.execute(text("""
        ALTER TABLE session_documents
            ADD COLUMN IF NOT EXISTS content_type VARCHAR(20) DEFAULT 'document',
            ADD COLUMN IF NOT EXISTS source_url VARCHAR(2048);
        """))
        
        # 3. 扩展 chat_sessions 表 - 支持混合会话
        print("步骤 3: 扩展 chat_sessions 表...")
        session.execute(text("""
        ALTER TABLE chat_sessions
            ADD COLUMN IF NOT EXISTS web_ids UUID[],
            ADD COLUMN IF NOT EXISTS mixed_content_count INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS last_content_type VARCHAR(20);
        
        -- 更新session_type允许的值
        UPDATE chat_sessions SET session_type = 'general' WHERE session_type IS NULL;
        """))
        
        # 4. ✅ 修复: 创建 web_indexes 表 - 引用正确的主键字段
        print("步骤 4: 创建 web_indexes 表...")
        session.execute(text("""
        CREATE TABLE IF NOT EXISTS web_indexes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            paper_id UUID NOT NULL REFERENCES paper_analysis(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding JSONB,
            chunk_metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        
        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_web_indexes_paper_id ON web_indexes(paper_id);
        CREATE INDEX IF NOT EXISTS idx_web_indexes_chunk_index ON web_indexes(chunk_index);
        """))
        
        # 5. ✅ 修复: 创建 web_monitors 表
        print("步骤 5: 创建 web_monitors 表...")
        session.execute(text("""
        CREATE TABLE IF NOT EXISTS web_monitors (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            paper_id UUID NOT NULL REFERENCES paper_analysis(id) ON DELETE CASCADE,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            url VARCHAR(2048) NOT NULL,
            last_check TIMESTAMP WITH TIME ZONE DEFAULT now(),
            last_content_hash VARCHAR(64),
            check_interval_hours INTEGER DEFAULT 24,
            is_active BOOLEAN DEFAULT true,
            change_detected BOOLEAN DEFAULT false,
            notification_sent BOOLEAN DEFAULT false
        );
        
        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_web_monitors_user_id ON web_monitors(user_id);
        CREATE INDEX IF NOT EXISTS idx_web_monitors_paper_id ON web_monitors(paper_id);
        CREATE INDEX IF NOT EXISTS idx_web_monitors_url ON web_monitors(url);
        CREATE INDEX IF NOT EXISTS idx_web_monitors_active ON web_monitors(is_active);
        """))
        
        # 6. 创建 web_bookmarks 表
        print("步骤 6: 创建 web_bookmarks 表...")
        session.execute(text("""
        CREATE TABLE IF NOT EXISTS web_bookmarks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            paper_id UUID NOT NULL REFERENCES paper_analysis(id) ON DELETE CASCADE,
            folder_name VARCHAR(100) DEFAULT '默认',
            notes TEXT,
            tags VARCHAR[] DEFAULT '{}',
            is_favorite BOOLEAN DEFAULT false,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        
        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_web_bookmarks_user_id ON web_bookmarks(user_id);
        CREATE INDEX IF NOT EXISTS idx_web_bookmarks_paper_id ON web_bookmarks(paper_id);
        CREATE INDEX IF NOT EXISTS idx_web_bookmarks_folder ON web_bookmarks(folder_name);
        CREATE INDEX IF NOT EXISTS idx_web_bookmarks_favorite ON web_bookmarks(is_favorite);
        """))
        
        # 7. 创建 content_type_configs 表
        print("步骤 7: 创建 content_type_configs 表...")
        session.execute(text("""
        CREATE TABLE IF NOT EXISTS content_type_configs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content_type VARCHAR(20) NOT NULL UNIQUE,
            display_name VARCHAR(50) NOT NULL,
            icon VARCHAR(50),
            color VARCHAR(20),
            description TEXT,
            is_active BOOLEAN DEFAULT true,
            sort_order INTEGER DEFAULT 0
        );
        
        -- 插入默认配置
        INSERT INTO content_type_configs (content_type, display_name, icon, color, description, sort_order)
        VALUES 
            ('document', '文档', 'FileText', '#1890ff', '传统文档文件(PDF, Word, Excel等)', 1),
            ('web', '网页', 'Globe', '#52c41a', '网页内容和在线资源', 2),
            ('mixed', '混合', 'Layers', '#722ed1', '包含文档和网页的混合会话', 3)
        ON CONFLICT (content_type) DO NOTHING;
        """))
        
        # 8. 创建 content_recommendations 表
        print("步骤 8: 创建 content_recommendations 表...")
        session.execute(text("""
        CREATE TABLE IF NOT EXISTS content_recommendations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            content_id UUID NOT NULL,
            content_type VARCHAR(20) NOT NULL,
            recommendation_type VARCHAR(20) NOT NULL,
            score REAL DEFAULT 0.0,
            reason TEXT,
            is_shown BOOLEAN DEFAULT false,
            is_clicked BOOLEAN DEFAULT false,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        
        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_content_recommendations_user_id ON content_recommendations(user_id);
        CREATE INDEX IF NOT EXISTS idx_content_recommendations_content ON content_recommendations(content_id, content_type);
        CREATE INDEX IF NOT EXISTS idx_content_recommendations_type ON content_recommendations(recommendation_type);
        CREATE INDEX IF NOT EXISTS idx_content_recommendations_score ON content_recommendations(score DESC);
        """))
        
        # 9. 创建性能索引
        print("步骤 9: 创建性能优化索引...")
        session.execute(text("""
        -- paper_analysis 表的性能索引
        CREATE INDEX IF NOT EXISTS idx_paper_analysis_file_type ON paper_analysis(file_type);
        CREATE INDEX IF NOT EXISTS idx_paper_analysis_source_url ON paper_analysis(source_url);
        CREATE INDEX IF NOT EXISTS idx_paper_analysis_content_hash ON paper_analysis(content_hash);
        
        -- session_documents 表的性能索引  
        CREATE INDEX IF NOT EXISTS idx_session_documents_content_type ON session_documents(content_type);
        CREATE INDEX IF NOT EXISTS idx_session_documents_session_content ON session_documents(session_id, content_type);
        
        -- chat_sessions 表的性能索引
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_type ON chat_sessions(session_type);
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_type ON chat_sessions(user_id, session_type);
        """))
        
        # 10. 数据迁移 - 更新现有数据
        print("步骤 10: 更新现有数据...")
        session.execute(text("""
        -- 为现有的session_documents设置content_type
        UPDATE session_documents 
        SET content_type = 'document' 
        WHERE content_type IS NULL;
        
        -- 为现有的chat_sessions设置默认值
        UPDATE chat_sessions 
        SET mixed_content_count = 0, session_type = 'document'
        WHERE session_type = 'document' AND mixed_content_count IS NULL;
        """))
        
        # 提交所有更改
        session.commit()
        print(f"[{datetime.now()}] ✅ 数据库升级完成！网页支持功能已激活。")
        
        # 验证升级结果
        print("\n验证升级结果:")
        
        # 检查新表是否创建成功
        tables_to_check = ['web_indexes', 'web_monitors', 'web_bookmarks', 'content_type_configs', 'content_recommendations']
        for table in tables_to_check:
            try:
                result = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                print(f"  ✓ {table} 表: {result} 条记录")
            except Exception as e:
                print(f"  ❌ {table} 表检查失败: {e}")
        
        # 检查新字段是否添加成功
        result = session.execute(text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'paper_analysis' 
        AND column_name IN ('source_url', 'web_metadata', 'content_hash');
        """)).fetchall()
        print(f"  ✓ paper_analysis 新字段: {[r[0] for r in result]}")
        
        result = session.execute(text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'session_documents' 
        AND column_name IN ('content_type', 'source_url');
        """)).fetchall()
        print(f"  ✓ session_documents 新字段: {[r[0] for r in result]}")
        
        result = session.execute(text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'chat_sessions' 
        AND column_name IN ('web_ids', 'mixed_content_count', 'last_content_type');
        """)).fetchall()
        print(f"  ✓ chat_sessions 新字段: {[r[0] for r in result]}")
        
        print(f"\n🎉 升级成功！现在可以使用网页分析功能了。")
        
    except Exception as e:
        session.rollback()
        print(f"❌ 升级失败: {str(e)}")
        raise
    finally:
        session.close()

def downgrade():
    """降级数据库模型 - 移除网页支持"""
    session = SessionLocal()
    
    try:
        print(f"[{datetime.now()}] 开始降级数据库，移除网页功能...")
        
        # 删除新创建的表
        tables_to_drop = ['content_recommendations', 'web_bookmarks', 'content_type_configs', 'web_monitors', 'web_indexes']
        for table in tables_to_drop:
            session.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE;"))
            print(f"  ✓ 删除表: {table}")
        
        # 移除新增的字段
        session.execute(text("""
        ALTER TABLE paper_analysis
            DROP COLUMN IF EXISTS source_url,
            DROP COLUMN IF EXISTS web_metadata,
            DROP COLUMN IF EXISTS web_links,
            DROP COLUMN IF EXISTS web_images,
            DROP COLUMN IF EXISTS fetch_time,
            DROP COLUMN IF EXISTS content_hash;
        """))
        
        session.execute(text("""
        ALTER TABLE session_documents
            DROP COLUMN IF EXISTS content_type,
            DROP COLUMN IF EXISTS source_url;
        """))
        
        session.execute(text("""
        ALTER TABLE chat_sessions
            DROP COLUMN IF EXISTS web_ids,
            DROP COLUMN IF EXISTS mixed_content_count,
            DROP COLUMN IF EXISTS last_content_type;
        """))
        
        session.commit()
        print(f"[{datetime.now()}] ✅ 数据库降级完成！")
        
    except Exception as e:
        session.rollback()
        print(f"❌ 降级失败: {str(e)}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "downgrade":
        downgrade()
    else:
        upgrade() 