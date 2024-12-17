import json
import asyncio
import psutil
import GPUtil
import time
from channels.generic.websocket import AsyncWebsocketConsumer

class SystemMonitorConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.running = True
        asyncio.create_task(self.send_system_stats())

    async def disconnect(self, close_code):
        self.running = False

    async def send_system_stats(self):
        while self.running:
            try:
                # 获取CPU信息
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_count = psutil.cpu_count()
                cpu_freq = psutil.cpu_freq()
                cpu_stats = {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'freq_current': round(cpu_freq.current, 2),
                    'freq_min': round(cpu_freq.min, 2),
                    'freq_max': round(cpu_freq.max, 2)
                }
                
                # 获取内存使用情况
                memory = psutil.virtual_memory()
                memory_stats = {
                    'percent': memory.percent,
                    'used': round(memory.used / (1024 * 1024 * 1024), 2),  # GB
                    'total': round(memory.total / (1024 * 1024 * 1024), 2),
                    'available': round(memory.available / (1024 * 1024 * 1024), 2),
                    'cached': round(memory.cached / (1024 * 1024 * 1024), 2) if hasattr(memory, 'cached') else 0
                }
                
                # 获取GPU使用情况
                gpu_stats = []
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_stats.append({
                            'id': gpu.id,
                            'name': gpu.name,
                            'load': round(gpu.load * 100, 2),
                            'memory_used': round(gpu.memoryUsed, 2),
                            'memory_total': round(gpu.memoryTotal, 2),
                            'temperature': round(gpu.temperature, 2),
                            'power_draw': round(gpu.powerDraw, 2) if hasattr(gpu, 'powerDraw') else 0
                        })
                except:
                    pass
                
                # 获取磁盘使用情况
                disk = psutil.disk_usage('/')
                disk_io = psutil.disk_io_counters()
                disk_stats = {
                    'percent': disk.percent,
                    'used': round(disk.used / (1024 * 1024 * 1024), 2),
                    'total': round(disk.total / (1024 * 1024 * 1024), 2),
                    'read_bytes': round(disk_io.read_bytes / (1024 * 1024 * 1024), 2),
                    'write_bytes': round(disk_io.write_bytes / (1024 * 1024 * 1024), 2)
                }
                
                # 获取网络使用情况
                net_io = psutil.net_io_counters()
                net_stats = {
                    'bytes_sent': round(net_io.bytes_sent / (1024 * 1024), 2),  # MB
                    'bytes_recv': round(net_io.bytes_recv / (1024 * 1024), 2),
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errin': net_io.errin,
                    'errout': net_io.errout,
                    'dropin': net_io.dropin,
                    'dropout': net_io.dropout
                }
                
                # 发送数据
                await self.send(json.dumps({
                    'type': 'system_stats',
                    'data': {
                        'cpu': cpu_stats,
                        'memory': memory_stats,
                        'gpu': gpu_stats,
                        'disk': disk_stats,
                        'network': net_stats,
                        'timestamp': int(time.time())
                    }
                }))
                
                # 每3秒更新一次
                await asyncio.sleep(3)
                
            except Exception as e:
                print(f"Error in system monitor: {str(e)}")
                await asyncio.sleep(3) 