from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.views import TokenObtainPairView
from django.contrib.auth import get_user_model
from .serializers import UserSerializer
from apps.data.models import Dataset
from apps.model.models import RainClassificationModel, RainRegressionModel, TrainingRecord
from apps.model.views import ModelTrainView, get_model_predictions
from apps.model.serializers import TrainingRecordSerializer
from rest_framework.views import APIView
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta
import psutil
import GPUtil
import time
import os
import pandas as pd
import numpy as np
from django.conf import settings
import tensorflow as tf
import traceback
from rest_framework.test import APIRequestFactory, force_authenticate

User = get_user_model()

class CustomTokenObtainPairView(TokenObtainPairView):
    """自定义令牌获取视图"""
    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        if response.status_code == 200:
            user = User.objects.get(username=request.data['username'])
            response.data['user'] = UserSerializer(user).data
        return response

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    """用户注册视图"""
    try:
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(
                {"message": "注册成功"},
                status=status.HTTP_201_CREATED
            )
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        return Response(
            {"detail": str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_system_stats(request):
    try:
        # 获取当前用户的数据集统计
        total_datasets = Dataset.objects.filter(created_by=request.user).count()
        user_datasets = total_datasets  # 因为已经过滤了当前用户，所以相同
        
        # 获取当前用户的模型统计
        total_models = (
            RainClassificationModel.objects.filter(created_by=request.user).count() +
            RainRegressionModel.objects.filter(created_by=request.user).count()
        )
        user_models = total_models  # 因为已经过滤了当前用户，所以相同
        
        # 获取当前用户的分类模型和回归模型的数量
        classification_models = RainClassificationModel.objects.filter(created_by=request.user).count()
        regression_models = RainRegressionModel.objects.filter(created_by=request.user).count()

        return Response({
            'total_datasets': total_datasets,
            'user_datasets': user_datasets,
            'total_models': total_models,
            'user_models': user_models,
            'classification_models': classification_models,
            'regression_models': regression_models
        })
    except Exception as e:
        return Response({'detail': str(e)}, status=500)

class DashboardStatsView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        try:
            # 获取当前时间和上个月的时间
            now = timezone.now()
            last_month = now - timedelta(days=30)
            
            # 获取当前用户的数据集统计
            total_datasets = Dataset.objects.filter(created_by=request.user).count()
            last_month_datasets = Dataset.objects.filter(
                created_by=request.user,
                created_time__gte=last_month
            ).count()
            dataset_trend = ((total_datasets - last_month_datasets) / last_month_datasets * 100 
                            if last_month_datasets > 0 else 0)

            # 获取当前用户的模型统计
            total_models = (
                RainClassificationModel.objects.filter(created_by=request.user).count() +
                RainRegressionModel.objects.filter(created_by=request.user).count()
            )
            classification_models = RainClassificationModel.objects.filter(created_by=request.user).count()
            regression_models = RainRegressionModel.objects.filter(created_by=request.user).count()
            last_month_models = (
                RainClassificationModel.objects.filter(
                    created_by=request.user,
                    created_time__gte=last_month
                ).count() +
                RainRegressionModel.objects.filter(
                    created_by=request.user,
                    created_time__gte=last_month
                ).count()
            )
            model_trend = ((total_models - last_month_models) / last_month_models * 100 
                          if last_month_models > 0 else 0)

            # 获取当前用户的训练任务统计
            training_success = (
                RainClassificationModel.objects.filter(
                    created_by=request.user,
                    training_status='success'
                ).count() +
                RainRegressionModel.objects.filter(
                    created_by=request.user,
                    training_status='success'
                ).count()
            )
            training_failed = (
                RainClassificationModel.objects.filter(
                    created_by=request.user,
                    training_status='failed'
                ).count() +
                RainRegressionModel.objects.filter(
                    created_by=request.user,
                    training_status='failed'
                ).count()
            )
            training_count = training_success + training_failed

            # 获取月度数据
            monthly_datasets = []
            monthly_models = []
            monthly_trainings = []
            for i in range(6):
                month_start = now - timedelta(days=30*(i+1))
                month_end = now - timedelta(days=30*i)
                
                datasets = Dataset.objects.filter(
                    created_by=request.user,
                    created_time__gte=month_start,
                    created_time__lt=month_end
                ).count()
                monthly_datasets.append(datasets)
                
                models = (
                    RainClassificationModel.objects.filter(
                        created_by=request.user,
                        created_time__gte=month_start,
                        created_time__lt=month_end
                    ).count() +
                    RainRegressionModel.objects.filter(
                        created_by=request.user,
                        created_time__gte=month_start,
                        created_time__lt=month_end
                    ).count()
                )
                monthly_models.append(models)
                
                trainings = (
                    RainClassificationModel.objects.filter(
                        created_by=request.user,
                        created_time__gte=month_start,
                        created_time__lt=month_end,
                        training_status='success'
                    ).count() +
                    RainRegressionModel.objects.filter(
                        created_by=request.user,
                        created_time__gte=month_start,
                        created_time__lt=month_end,
                        training_status='success'
                    ).count()
                )
                monthly_trainings.append(trainings)

            # 获取系统资源使用情况
            try:
                # CPU 率
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
                
                # 内存使用情况
                memory = psutil.virtual_memory()
                memory_stats = {
                    'percent': memory.percent,
                    'used': round(memory.used / (1024 * 1024 * 1024), 2),  # GB
                    'total': round(memory.total / (1024 * 1024 * 1024), 2),
                    'available': round(memory.available / (1024 * 1024 * 1024), 2),
                    'cached': round(memory.cached / (1024 * 1024 * 1024), 2) if hasattr(memory, 'cached') else 0
                }
                
                # GPU 使用情况
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
                
                # 磁盘使用情况
                disk = psutil.disk_usage('/')
                disk_io = psutil.disk_io_counters()
                disk_stats = {
                    'percent': disk.percent,
                    'used': round(disk.used / (1024 * 1024 * 1024), 2),
                    'total': round(disk.total / (1024 * 1024 * 1024), 2),
                    'read_bytes': round(disk_io.read_bytes / (1024 * 1024 * 1024), 2),
                    'write_bytes': round(disk_io.write_bytes / (1024 * 1024 * 1024), 2)
                }

                system_resources = {
                    'cpu': cpu_stats,
                    'memory': memory_stats,
                    'gpu': gpu_stats,
                    'disk': disk_stats,
                    'timestamp': int(time.time())
                }
            except Exception as e:
                print(f"Error getting system resources: {str(e)}")
                system_resources = None

            # 获取最新的晴雨区分和降雨反演模型数据
            latest_detection_model = RainClassificationModel.objects.filter(
                created_by=request.user
            ).order_by('-created_time').first()

            latest_inversion_model = RainRegressionModel.objects.filter(
                created_by=request.user
            ).order_by('-created_time').first()

            # 获取训练记录和预测结果
            detection_records = []
            inversion_records = []
            detection_predictions = []
            inversion_predictions = []

            if latest_detection_model:
                print(f"Found detection model: {latest_detection_model.id}")
                detection_records = TrainingRecord.objects.filter(
                    model_id=latest_detection_model.id
                ).order_by('epoch')
                print(f"Found {len(detection_records)} detection records")
                
                # 获取晴雨区分预测结果
                try:
                    factory = APIRequestFactory()
                    request = factory.get(f'/api/model/models/{latest_detection_model.id}/predictions/')
                    force_authenticate(request, user=self.request.user)
                    response = get_model_predictions(request, latest_detection_model.id)
                    if response.status_code == 200:
                        detection_predictions = response.data.get('predictions', [])
                        print(f"Found {len(detection_predictions)} detection predictions")
                except Exception as e:
                    print(f"Error getting detection predictions: {str(e)}")
                    print(traceback.format_exc())

            if latest_inversion_model:
                print(f"Found inversion model: {latest_inversion_model.id}")
                inversion_records = TrainingRecord.objects.filter(
                    model_id=latest_inversion_model.id
                ).order_by('epoch')
                print(f"Found {len(inversion_records)} inversion records")
                
                # 获取降雨反演预测结果
                try:
                    factory = APIRequestFactory()
                    request = factory.get(f'/api/model/models/{latest_inversion_model.id}/predictions/')
                    force_authenticate(request, user=self.request.user)
                    response = get_model_predictions(request, latest_inversion_model.id)
                    if response.status_code == 200:
                        inversion_predictions = response.data.get('predictions', [])
                        print(f"Found {len(inversion_predictions)} inversion predictions")
                except Exception as e:
                    print(f"Error getting inversion predictions: {str(e)}")
                    print(traceback.format_exc())

            print(f"Detection records: {len(detection_records)}")
            print(f"Inversion records: {len(inversion_records)}")
            print(f"Detection predictions: {len(detection_predictions)}")
            print(f"Inversion predictions: {len(inversion_predictions)}")

            return Response({
                'total_datasets': total_datasets,
                'dataset_trend': dataset_trend,
                'total_models': total_models,
                'classification_models': classification_models,
                'regression_models': regression_models,
                'model_trend': model_trend,
                'training_count': training_count,
                'training_success': training_success,
                'training_failed': training_failed,
                'prediction_count': 0,
                'monthly_datasets': monthly_datasets,
                'monthly_models': monthly_models,
                'monthly_trainings': monthly_trainings,
                'latest_detection': {
                    'records': TrainingRecordSerializer(detection_records, many=True).data,
                    'predictions': detection_predictions
                },
                'latest_inversion': {
                    'records': TrainingRecordSerializer(inversion_records, many=True).data,
                    'predictions': inversion_predictions
                }
            })
            
        except Exception as e:
            return Response({
                'detail': f'获取统计数据失败: {str(e)}'
            }, status=500)

@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def update_profile(request):
    """更新个人信息"""
    try:
        user = request.user
        serializer = UserSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def change_password(request):
    """修改密码"""
    try:
        user = request.user
        old_password = request.data.get('old_password')
        new_password = request.data.get('new_password')

        if not user.check_password(old_password):
            return Response({'detail': '原密码错误'}, status=status.HTTP_400_BAD_REQUEST)

        user.set_password(new_password)
        user.save()
        return Response({'detail': '密码修改成功'})
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)
