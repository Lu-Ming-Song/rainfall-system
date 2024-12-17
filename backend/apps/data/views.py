from django.shortcuts import render
import os
from django.conf import settings
import pandas as pd
import numpy as np
from rest_framework import status, generics
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from django.http import FileResponse
from .models import Dataset
from .serializers import DatasetSerializer
import time
from sklearn.impute import KNNImputer
# from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
# Create your views here.

class DatasetUploadView(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request):
        try:
            file = request.FILES['file']
            name = request.data.get('name', file.name)
            missing_value_strategy = request.data.get('missing_value_strategy', 'knn')
            feature_engineering = request.data.get('feature_engineering', [])

            # 读取数据
            df = pd.read_csv(file)
            
            # 1. 数据验证
            required_columns = ['time']
            if not all(col in df.columns for col in required_columns):
                return Response(
                    {'detail': f'数据集必须包含以下列: {", ".join(required_columns)}'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 验证时间格式
            try:
                df['time'] = pd.to_datetime(df['time'])
            except:
                return Response(
                    {'detail': 'time列格式不正确，应为标准时间格式'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 2. 特征工程
            if 'time_features' in feature_engineering:
                df['hour'] = df['time'].dt.hour
                df['day_of_week'] = df['time'].dt.dayofweek
                df['month'] = df['time'].dt.month
                df['quarter'] = df['time'].dt.quarter
                df['is_weekend'] = df['time'].dt.dayofweek.isin([5, 6]).astype(int)

            if 'statistical_features' in feature_engineering:
                window_sizes = [5, 10, 15]  # 多个窗口大小
                for col in ['attenuation', 'attenuation_max', 'attenuation_min', 'attenuation_mean']:
                    if col in df.columns:
                        for window in window_sizes:
                            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                            df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
                            df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                            df[f'{col}_rolling_kurt_{window}'] = df[col].rolling(window=window).kurt()
                            df[f'{col}_rolling_skew_{window}'] = df[col].rolling(window=window).skew()

            if 'lag_features' in feature_engineering:
                lag_periods = [1, 2, 3, 5, 10]  # 多个滞后周期
                for col in ['attenuation', 'attenuation_max', 'attenuation_min', 'attenuation_mean']:
                    if col in df.columns:
                        for lag in lag_periods:
                            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

            # 3. 缺失值处理
            if missing_value_strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
            elif missing_value_strategy == 'iterative':
                imputer = SimpleImputer(max_iter=10, random_state=0)
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
            elif missing_value_strategy == 'mean':
                df = df.fillna(df.mean())
            elif missing_value_strategy == 'median':
                df = df.fillna(df.median())
            elif missing_value_strategy == 'forward':
                df = df.fillna(method='ffill')
            elif missing_value_strategy == 'backward':
                df = df.fillna(method='bfill')
            elif missing_value_strategy == 'linear':
                df = df.interpolate(method='linear')
            elif missing_value_strategy == 'cubic':
                df = df.interpolate(method='cubic')
            elif missing_value_strategy == 'drop':
                df = df.dropna()

            # 4. 异常值处理
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

            # 5. 数据压缩和优化
            # 优化数据类型
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = df[col].astype('float32')
                elif df[col].dtype == 'int64':
                    df[col] = df[col].astype('int32')

            # 保存文件
            file_name = f'dataset_{int(time.time())}_{file.name}'
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 使用压缩保存
            df.to_parquet(file_path.replace('.csv', '.parquet'), compression='snappy')

            # 创建数据集记录
            dataset = Dataset.objects.create(
                name=name,
                file=file_name.replace('.csv', '.parquet'),
                row_count=len(df),
                column_count=len(df.columns),
                feature_count=len([col for col in df.columns if col not in ['time', 'rainfall_intensity']]),
                missing_value_strategy=missing_value_strategy,
                feature_engineering=feature_engineering,
                created_by=request.user,
                file_size=os.path.getsize(file_path.replace('.csv', '.parquet')),
                data_range={
                    'start': df['time'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': df['time'].max().strftime('%Y-%m-%d %H:%M:%S')
                }
            )

            serializer = DatasetSerializer(dataset)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response(
                {'detail': f'数据集上传失败: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

class DatasetListView(generics.ListAPIView):
    serializer_class = DatasetSerializer
    filterset_fields = ['status']
    search_fields = ['name']
    ordering_fields = ['created_time', 'row_count']

    def get_queryset(self):
        # 只返回当前用户的数据集
        queryset = Dataset.objects.filter(created_by=self.request.user)
        name = self.request.query_params.get('name', None)
        start_time = self.request.query_params.get('start_time', None)
        end_time = self.request.query_params.get('end_time', None)

        if name:
            queryset = queryset.filter(name__icontains=name)
        
        if start_time and end_time:
            queryset = queryset.filter(created_time__range=[start_time, end_time])

        return queryset.order_by('-created_time')

class DatasetDetailView(APIView):
    def get(self, request, pk):
        try:
            dataset = Dataset.objects.get(pk=pk)
            file_path = os.path.join(settings.MEDIA_ROOT, str(dataset.file))
            
            if not os.path.exists(file_path):
                return Response(
                    {'detail': '数据文件不存在'},
                    status=status.HTTP_404_NOT_FOUND
                )

            df = pd.read_csv(file_path)
            # 确保时间列是正确的格式
            df['time'] = pd.to_datetime(df['time'])
            data_range = {
                'start': df['time'].min().strftime('%Y-%m-%d %H:%M:%S') if not df.empty else None,
                'end': df['time'].max().strftime('%Y-%m-%d %H:%M:%S') if not df.empty else None,
            }

            serializer = DatasetSerializer(dataset)
            response_data = serializer.data
            response_data['data_range'] = data_range
            response_data['created_by'] = {
                'username': dataset.created_by.username,
                'id': dataset.created_by.id
            }
            
            return Response(response_data)
        except Dataset.DoesNotExist:
            return Response(
                {'detail': '数据集不存在'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'detail': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

    def patch(self, request, pk):
        try:
            dataset = Dataset.objects.get(pk=pk)
            serializer = DatasetSerializer(dataset, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Dataset.DoesNotExist:
            return Response(
                {'detail': '数据集不存在'},
                status=status.HTTP_404_NOT_FOUND
            )

    def delete(self, request, pk):
        try:
            dataset = Dataset.objects.get(pk=pk)
            # 删除文件
            file_path = os.path.join(settings.MEDIA_ROOT, str(dataset.file))
            if os.path.exists(file_path):
                os.remove(file_path)
            # 删除数据集记录
            dataset.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Dataset.DoesNotExist:
            return Response(
                {'detail': '数据集不存在'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'detail': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

class DatasetDataView(APIView):
    def get(self, request, pk, row_id=None):
        try:
            dataset = Dataset.objects.get(pk=pk)
            file_path = os.path.join(settings.MEDIA_ROOT, str(dataset.file))
            
            if not os.path.exists(file_path):
                return Response(
                    {'detail': '数据文件不存在'},
                    status=status.HTTP_404_NOT_FOUND
                )

            df = pd.read_csv(file_path)
            # 确保时间列是正确的格式
            df['time'] = pd.to_datetime(df['time'])
            
            # 时间范围过滤
            start_time = request.query_params.get('start_time')
            end_time = request.query_params.get('end_time')
            if start_time and end_time:
                start_time = pd.to_datetime(start_time)
                end_time = pd.to_datetime(end_time)
                df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
            
            # 检查是否是可视化请求
            is_visualization = request.query_params.get('visualization') == 'true'
            
            if is_visualization:
                # 对于可视化请求，进行数据采样
                total_points = len(df)
                if total_points > 500:  # 减少采样点数
                    # 计算采样间隔
                    sample_interval = total_points // 500
                    df = df.iloc[::sample_interval].copy()
                
                # 将时间转换回字符串格式
                df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                records = df.to_dict('records')
                for idx, record in enumerate(records):
                    record['id'] = idx
                
                return Response({
                    'count': len(records),
                    'results': records,
                })
            else:
                # 对于表格请求，使用分页
                page = int(request.query_params.get('page', 1))
                page_size = int(request.query_params.get('page_size', 10))
                
                total_count = len(df)
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, total_count)
                
                page_data = df.iloc[start_idx:end_idx]
                
                records = []
                for idx, row in page_data.iterrows():
                    record = row.to_dict()
                    record['id'] = idx
                    records.append(record)
                
                return Response({
                    'count': total_count,
                    'results': records,
                })
                
        except Dataset.DoesNotExist:
            return Response(
                {'detail': '数据集不存在'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'detail': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

    def put(self, request, pk, row_id):
        try:
            dataset = Dataset.objects.get(pk=pk)
            file_path = os.path.join(settings.MEDIA_ROOT, str(dataset.file))
            
            if not os.path.exists(file_path):
                return Response(
                    {'detail': '数据文件不存在'},
                    status=status.HTTP_404_NOT_FOUND
                )

            df = pd.read_csv(file_path)
            if row_id >= len(df):
                return Response(
                    {'detail': '数据行不存在'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # 更新数据
            df.loc[row_id, 'attenuation'] = request.data.get('attenuation')
            df.loc[row_id, 'rainfall_intensity'] = request.data.get('rainfall_intensity')
            df.to_csv(file_path, index=False)
            
            return Response({'detail': '更新成��'})
        except Dataset.DoesNotExist:
            return Response(
                {'detail': '数据集不存在'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'detail': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

    def delete(self, request, pk, row_id):
        try:
            dataset = Dataset.objects.get(pk=pk)
            file_path = os.path.join(settings.MEDIA_ROOT, str(dataset.file))
            
            if not os.path.exists(file_path):
                return Response(
                    {'detail': '数据文件不存在'},
                    status=status.HTTP_404_NOT_FOUND
                )

            df = pd.read_csv(file_path)
            if row_id >= len(df):
                return Response(
                    {'detail': '数据行不存在'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # 删除数据行
            df = df.drop(row_id)
            df.to_csv(file_path, index=False)
            
            # 更新数据集行数
            dataset.row_count = len(df)
            dataset.save()
            
            return Response({'detail': '删除成功'})
        except Dataset.DoesNotExist:
            return Response(
                {'detail': '数据集不存在'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'detail': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

class DatasetExportView(APIView):
    def get(self, request, pk=None):
        try:
            if pk:
                # 导出单个数据集
                dataset = Dataset.objects.get(pk=pk)
                file_path = os.path.join(settings.MEDIA_ROOT, str(dataset.file))
                if os.path.exists(file_path):
                    return FileResponse(
                        open(file_path, 'rb'),
                        as_attachment=True,
                        filename=dataset.name
                    )
                else:
                    return Response(
                        {'detail': '文件不存在'},
                        status=status.HTTP_404_NOT_FOUND
                    )
            else:
                # 导出所有数据集
                datasets = Dataset.objects.all()
                data = []
                for dataset in datasets:
                    file_path = os.path.join(settings.MEDIA_ROOT, str(dataset.file))
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        data.append(df)
                
                if data:
                    combined_df = pd.concat(data, ignore_index=True)
                    export_path = os.path.join(settings.MEDIA_ROOT, 'exports')
                    os.makedirs(export_path, exist_ok=True)
                    export_file = os.path.join(export_path, 'combined_data.csv')
                    combined_df.to_csv(export_file, index=False)
                    return FileResponse(
                        open(export_file, 'rb'),
                        as_attachment=True,
                        filename='combined_data.csv'
                    )
                else:
                    return Response(
                        {'detail': '没有可导出的数据'},
                        status=status.HTTP_404_NOT_FOUND
                    )
                
        except Dataset.DoesNotExist:
            return Response(
                {'detail': '数据集不存在'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'detail': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
