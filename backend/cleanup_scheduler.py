 # 文件：backend/cleanup_scheduler.py    实现数据库清理调度器    
"""
数据库清理调度器
定期清理旧的临时会话、孤立数据等
"""

import asyncio
import schedule
import time
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from database import engine
from services.paper_analyzer import PaperAnalyzerService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseCleaner:
    def __init__(self):
        self.SessionLocal = sessionmaker(bind=engine)
        
    async def run_cleanup_job(self, job_type: str = "all"):
        """执行清理任务"""
        try:
            logger.info(f"开始执行清理任务: {job_type}")
            
            # 创建数据库会话
            db = self.SessionLocal()
            try:
                service = PaperAnalyzerService(db)
                
                if job_type in ["all", "old_sessions"]:
                    # 清理旧的临时会话（7天）
                    logger.info("清理旧的临时会话...")
                    old_session_stats = await service.cleanup_old_sessions(
                        days_threshold=7, 
                        dry_run=False
                    )
                    logger.info(f"旧会话清理完成: {old_session_stats}")
                
                if job_type in ["all", "orphaned_data"]:
                    # 清理孤立数据
                    logger.info("清理孤立数据...")
                    orphaned_stats = await service.cleanup_orphaned_data(dry_run=False)
                    logger.info(f"孤立数据清理完成: {orphaned_stats}")
                
                if job_type in ["all", "large_fragments"]:
                    # 清理过期的大型分片（30天）
                    logger.info("清理过期的大型分片...")
                    await self.cleanup_old_fragments(service, days_threshold=30)
                
                logger.info(f"清理任务完成: {job_type}")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"清理任务失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def cleanup_old_fragments(self, service, days_threshold: int = 30):
        """清理过期的消息分片"""
        try:
            from models.chat import MessageFragment, ChatMessage
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            
            # 查找过期的分片（通过关联的消息创建时间）
            old_fragments = service.db.query(MessageFragment).join(
                ChatMessage, MessageFragment.message_id == ChatMessage.id
            ).filter(
                ChatMessage.created_at < cutoff_date
            ).all()
            
            if old_fragments:
                logger.info(f"找到 {len(old_fragments)} 个过期分片")
                
                # 删除过期分片
                fragment_ids = [f.id for f in old_fragments]
                deleted_count = service.db.query(MessageFragment).filter(
                    MessageFragment.id.in_(fragment_ids)
                ).delete(synchronize_session=False)
                
                service.db.commit()
                logger.info(f"删除了 {deleted_count} 个过期分片")
            else:
                logger.info("没有找到过期分片")
                
        except Exception as e:
            service.db.rollback()
            logger.error(f"清理过期分片失败: {str(e)}")
    
    def run_daily_cleanup(self):
        """每日清理任务"""
        logger.info("执行每日清理任务")
        asyncio.run(self.run_cleanup_job("all"))
    
    def run_weekly_deep_cleanup(self):
        """每周深度清理任务"""
        logger.info("执行每周深度清理任务")
        asyncio.run(self.run_cleanup_job("all"))
        # 可以添加更多深度清理逻辑
    
    def run_hourly_orphaned_cleanup(self):
        """每小时清理孤立数据"""
        logger.info("执行每小时孤立数据清理")
        asyncio.run(self.run_cleanup_job("orphaned_data"))

def setup_scheduler():
    """设置清理调度"""
    cleaner = DatabaseCleaner()
    
    # 每天凌晨2点执行日常清理
    schedule.every().day.at("02:00").do(cleaner.run_daily_cleanup)
    
    # 每周日凌晨3点执行深度清理
    schedule.every().sunday.at("03:00").do(cleaner.run_weekly_deep_cleanup)
    
    # 每小时清理孤立数据
    schedule.every().hour.do(cleaner.run_hourly_orphaned_cleanup)
    
    logger.info("清理调度器已设置")
    logger.info("- 每日清理: 每天凌晨2点")
    logger.info("- 深度清理: 每周日凌晨3点")
    logger.info("- 孤立数据清理: 每小时")

def run_scheduler():
    """运行调度器"""
    setup_scheduler()
    
    logger.info("清理调度器启动")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    except KeyboardInterrupt:
        logger.info("清理调度器停止")
    except Exception as e:
        logger.error(f"调度器运行错误: {str(e)}")

def run_manual_cleanup(job_type: str = "all", dry_run: bool = False):
    """手动执行清理任务"""
    cleaner = DatabaseCleaner()
    
    if dry_run:
        logger.info(f"模拟执行清理任务: {job_type}")
        # 这里可以添加模拟清理的逻辑
    else:
        logger.info(f"手动执行清理任务: {job_type}")
        asyncio.run(cleaner.run_cleanup_job(job_type))

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "schedule":
            # 启动调度器
            run_scheduler()
        elif command == "manual":
            # 手动清理
            job_type = sys.argv[2] if len(sys.argv) > 2 else "all"
            dry_run = "--dry-run" in sys.argv
            run_manual_cleanup(job_type, dry_run)
        elif command == "test":
            # 测试清理（模拟运行）
            job_type = sys.argv[2] if len(sys.argv) > 2 else "all"
            run_manual_cleanup(job_type, dry_run=True)
        else:
            print("用法:")
            print("  python cleanup_scheduler.py schedule     # 启动调度器")
            print("  python cleanup_scheduler.py manual [type] # 手动清理")
            print("  python cleanup_scheduler.py test [type]   # 测试清理")
            print("  清理类型: all, old_sessions, orphaned_data, large_fragments")
    else:
        print("用法:")
        print("  python cleanup_scheduler.py schedule     # 启动调度器")
        print("  python cleanup_scheduler.py manual [type] # 手动清理")
        print("  python cleanup_scheduler.py test [type]   # 测试清理")
        print("  清理类型: all, old_sessions, orphaned_data, large_fragments") 