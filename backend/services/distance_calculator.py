# backend/services/distance_calculator.py    经纬度计算器   
import openpyxl
import httpx
import os
from typing import Tuple, Optional
from datetime import datetime
import logging
from dotenv import load_dotenv
from io import BytesIO
import asyncio

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistanceCalculator:
    def __init__(self):
        # 从环境变量获取配置
        load_dotenv()
        self.api_url = 'https://maps.googleapis.com/maps/api/distancematrix/json'
        self.api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        
        if not self.api_key:
            raise ValueError("Google Maps API key not configured")
        
        # 添加 API 调用计数器
        self.api_call_count = 0
        self.api_call_limit = 100  # 每100次调用后休眠

    def _get_cell_value(self, sheet, cell: str) -> Optional[str]:
        """安全地获取单元格值，处理各种异常情况"""
        try:
            value = sheet[cell].value
            if value is None:
                return None
            return str(value).strip()
        except Exception:
            return None

    def _is_valid_coordinates(self, coord_str: str) -> bool:
        """验证坐标格式是否有效"""
        try:
            # 移除所有空白字符
            coord_str = "".join(coord_str.split())
            
            # 检查是否包含逗号
            if ',' not in coord_str:
                return False

            # 分割并转换为浮点数
            lat, lng = map(float, coord_str.split(','))
            
            # 检查范围是否有效
            return -90 <= lat <= 90 and -180 <= lng <= 180
        except Exception:
            return False

    async def process_excel(self, file_content: bytes, filename: str) -> Tuple[bytes, str]:
        """处理Excel文件并返回结果"""
        try:
            # 使用 BytesIO 读取文件内容
            wb = openpyxl.load_workbook(BytesIO(file_content), data_only=True)
            sheet = wb.active

            # 添加表头
            headers = {
                'G1': '距离',
                'H1': '时间',
                'I1': '处理状态'  # 新增状态列
            }
            for cell, value in headers.items():
                if not sheet[cell].value:
                    sheet[cell] = value

            # 重置 API 调用计数器
            self.api_call_count = 0
            processed_rows = 0
            skipped_rows = 0

            # 处理每一行
            for row in range(2, sheet.max_row + 1):
                # 读取起点和终点坐标
                origin = self._get_cell_value(sheet, f"C{row}")
                destination = self._get_cell_value(sheet, f"F{row}")

                # 跳过空行或无效坐标
                if not origin or not destination:
                    sheet[f"I{row}"] = "跳过 - 坐标为空"
                    skipped_rows += 1
                    continue

                if not self._is_valid_coordinates(origin) or not self._is_valid_coordinates(destination):
                    sheet[f"I{row}"] = "跳过 - 坐标格式无效"
                    skipped_rows += 1
                    continue

                try:
                    # API 调用频率限制
                    if self.api_call_count >= self.api_call_limit:
                        logger.info("Reached API call limit, sleeping for 1 second...")
                        await asyncio.sleep(1)
                        self.api_call_count = 0

                    distance, duration = await self.get_distance_and_duration(origin, destination)
                    sheet[f"G{row}"] = distance
                    sheet[f"H{row}"] = duration
                    
                    if distance == "请求失败" or distance == "API错误" or distance == "解析失败":
                        sheet[f"I{row}"] = "处理失败"
                        skipped_rows += 1
                    else:
                        sheet[f"I{row}"] = "处理成功"
                        processed_rows += 1
                    
                    self.api_call_count += 1

                except Exception as e:
                    logger.error(f"Error processing row {row}: {str(e)}")
                    sheet[f"G{row}"] = "处理失败"
                    sheet[f"H{row}"] = "处理失败"
                    sheet[f"I{row}"] = f"处理失败 - {str(e)}"
                    skipped_rows += 1

            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"distance_result_{timestamp}_{filename}"

            # 保存到字节流
            output = BytesIO()
            wb.save(output)
            output.seek(0)
            
            logger.info(f"处理完成: 成功 {processed_rows} 行, 跳过 {skipped_rows} 行")
            return output.getvalue(), output_filename

        except Exception as e:
            logger.error(f"Error processing excel file: {str(e)}")
            raise

    async def get_distance_and_duration(self, origin: str, destination: str) -> Tuple[str, str]:
        """调用Google Maps API获取距离和时间"""
        try:
            params = {
                'origins': origin,
                'destinations': destination,
                'key': self.api_key,
                'mode': 'driving',
                'language': 'zh-CN'  # 使用中文返回结果
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(self.api_url, params=params)
                
                if response.status_code == 429:  # Rate limit exceeded
                    logger.warning("API rate limit exceeded, sleeping for 2 seconds...")
                    await asyncio.sleep(2)
                    return await self.get_distance_and_duration(origin, destination)
                
                if response.status_code != 200:
                    logger.error(f"API request failed: {response.status_code}")
                    return "请求失败", "请求失败"

                data = response.json()
                
                if data.get('status') != 'OK':
                    logger.error(f"API returned error status: {data.get('status')}")
                    return f"API错误: {data.get('status')}", f"API错误: {data.get('status')}"

                try:
                    element = data['rows'][0]['elements'][0]
                    if element['status'] != 'OK':
                        return f"路线错误: {element['status']}", f"路线错误: {element['status']}"
                    
                    distance = element['distance']['text']
                    duration = element['duration']['text']
                    return distance, duration
                except (KeyError, IndexError) as e:
                    logger.error(f"Error parsing API response: {str(e)}")
                    return "解析失败", "解析失败"

        except Exception as e:
            logger.error(f"Error calling Google Maps API: {str(e)}")
            return "API错误", "API错误"