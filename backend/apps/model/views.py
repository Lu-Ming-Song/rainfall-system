from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import RainClassificationModel, RainRegressionModel, TrainingRecord
from .serializers import (
    RainClassificationModelSerializer,
    RainRegressionModelSerializer,
    TrainingRecordSerializer
)
from apps.data.models import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, GRU,
    MultiHeadAttention, LayerNormalization, BatchNormalization
)
import os
from django.conf import settings
import json
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import time
from scipy import signal
import pywt
import traceback
from django.http import HttpResponse
from rest_framework.viewsets import ViewSet
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, r2_score
import random
from sklearn.impute import KNNImputer
from scipy.stats import pearsonr  # 添加导入
from rest_framework.decorators import api_view, permission_classes
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback
from rest_framework.permissions import IsAuthenticated

# 设置全局随机种子
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 设置 TensorFlow 的确定性操作
try:
    # 对于较新版本的 TensorFlow
    tf.config.experimental.enable_op_determinism()
except (AttributeError, ImportError):
    # 对于较老版本的 TensorFlow
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# 定义训练回调类
class TrainingCallback(Callback):
    def __init__(self, training_id, channel_layer):
        super().__init__()
        self.training_id = training_id
        self.channel_layer = channel_layer

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        async_to_sync(self.channel_layer.group_send)(
            f'training_{self.training_id}',
            {
                'type': 'training_progress',
                'data': {
                    'epoch': epoch + 1,
                    'total_epochs': self.params['epochs'],
                    'loss': float(logs.get('loss', 0)),
                    'metrics': {
                        k: float(v) for k, v in logs.items()
                        if k not in ['loss']
                    }
                }
            }
        )

class ModelTrainView(APIView):
    def build_model(self, model_architecture, input_shape, params, train_type='rain_inversion'):
        """构建模型的通用方法"""
        if model_architecture == 'lstm':
            return self.build_lstm_model(input_shape, params, train_type)
        elif model_architecture == 'bilstm':
            return self.build_bilstm_model(input_shape, params, train_type)
        elif model_architecture == 'gru':
            return self.build_gru_model(input_shape, params, train_type)
        elif model_architecture == 'transformer':
            return self.build_transformer_model(input_shape, params, train_type)
        else:
            raise ValueError(f'不支持的模型架构: {model_architecture}')

    def build_lstm_model(self, input_shape, params, train_type):
        # 重新设置随机种子
        tf.random.set_seed(SEED)
        
        # 设置权重初始化器的随机种子
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=SEED)
        recurrent_initializer = tf.keras.initializers.Orthogonal(seed=SEED)
        
        model = Sequential([
            LSTM(params['hidden_size'], 
                 input_shape=input_shape,
                 return_sequences=True,
                 kernel_initializer=kernel_initializer,
                 recurrent_initializer=recurrent_initializer,
                 bias_initializer='zeros'),
            BatchNormalization(),
            Dropout(params['dropout_rate'], seed=SEED),
            
            LSTM(params['hidden_size'] // 2,
                 kernel_initializer=kernel_initializer,
                 recurrent_initializer=recurrent_initializer,
                 bias_initializer='zeros'),
            BatchNormalization(),
            Dropout(params['dropout_rate'], seed=SEED),
            
            Dense(32, activation='relu',
                  kernel_initializer=kernel_initializer,
                  bias_initializer='zeros'),
            BatchNormalization(),
            
            Dense(1, activation='sigmoid' if train_type == 'rain_detection' else None,
                  kernel_initializer=kernel_initializer,
                  bias_initializer='zeros')
        ])
        return model

    def build_bilstm_model(self, input_shape, params, train_type):
        """构建BiLSTM模型"""
        model = Sequential([
            # 第一层BiLSTM
            Bidirectional(
                LSTM(100, return_sequences=True),
                input_shape=input_shape
            ),
            Dropout(0.3),
            
            # 第二层BiLSTM
            Bidirectional(
                LSTM(100, return_sequences=True)
            ),
            Dropout(0.3),
            
            # 第三层BiLSTM
            Bidirectional(
                LSTM(50)
            ),
            Dropout(0.3),
            
            # 输出层
            Dense(1)
        ])
        
        return model

    def build_gru_model(self, input_shape, params, train_type):
        model = Sequential()
        for _ in range(params['lstm_layers']):
            model.add(GRU(
                params['hidden_size'],
                return_sequences=True,
                input_shape=input_shape
            ))
            model.add(Dropout(params['dropout_rate']))
        
        model.add(GRU(params['hidden_size'] // 2))
        model.add(Dropout(params['dropout_rate']))
        
        if train_type == 'rain_detection':
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(1))
        return model

    def positional_encoding(self, inputs, hidden_size):
        """位置编码"""
        def get_angles(pos, i, hidden_size):
            # 使用 tf.cast 确保数值类型正确
            angle_rates = 1 / tf.pow(10000.0, tf.cast(2 * (i//2), tf.float32) / tf.cast(hidden_size, tf.float32))
            return tf.cast(pos, tf.float32) * angle_rates

        def create_positional_encoding(length, hidden_size):
            # 使用 tf.range 替代 np.arange
            pos = tf.expand_dims(tf.range(tf.cast(length, tf.float32), dtype=tf.float32), axis=1)
            i = tf.range(tf.cast(hidden_size, tf.float32), dtype=tf.float32)
            i = tf.expand_dims(i, axis=0)
            
            angle_rads = get_angles(pos, i, hidden_size)
            
            # 使用 tf.sin 和 tf.cos 替代 np.sin 和 np.cos
            sines = tf.sin(angle_rads[:, 0::2])
            cosines = tf.cos(angle_rads[:, 1::2])
            
            # 交错合并 sin 和 cos
            pos_encoding = tf.concat([sines, cosines], axis=-1)
            pos_encoding = tf.expand_dims(pos_encoding, axis=0)
            
            return tf.cast(pos_encoding, dtype=tf.float32)

        # 获取序列长度
        seq_len = tf.shape(inputs)[1]
        
        # 创建位置编码
        pos_encoding = create_positional_encoding(seq_len, hidden_size)
        
        return inputs + pos_encoding

    def transformer_encoder(self, inputs, hidden_size, num_heads, dropout_rate):
        """Transformer编码器层"""
        # 多头自注意力
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads
        )(inputs, inputs)
        attention = tf.keras.layers.Dropout(dropout_rate)(attention)
        
        # 残差连接和层归一化
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
        
        # 前馈神经网络
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size * 4, activation='relu'),
            tf.keras.layers.Dense(hidden_size)
        ])
        
        ffn_output = ffn(attention)
        ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
        
        # 残差连接和层归一化
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + ffn_output)

    def build_transformer_model(self, input_shape, params, train_type):
        """构建Transformer模型"""
        # 设置默认参数
        num_layers = params.get('num_layers', 2)
        hidden_size = params.get('hidden_size', 64)
        num_heads = params.get('num_heads', 4)
        dropout_rate = params.get('dropout_rate', 0.1)
        
        inputs = tf.keras.Input(shape=input_shape)
        
        # 线性投影层
        x = tf.keras.layers.Dense(hidden_size)(inputs)
        
        # 添加位置编码
        x = self.positional_encoding(x, hidden_size)
        
        # Dropout
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        # Transformer编码器层
        for _ in range(num_layers):
            x = self.transformer_encoder(x, hidden_size, num_heads, dropout_rate)
        
        # 全局平均池化
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # 输出层
        if train_type == 'rain_detection':
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def preprocess_data(self, df, params):
        """数据预处理函数"""
        # 1. 转换时间列为datetime型
        df['time'] = pd.to_datetime(df['time'])
        
        # 2. 检查数据集包含的列并进行相应处理
        if 'attenuation' in df.columns:
            # 如果只有 attenuation 列，计算统计特征
            df['attenuation_max'] = df['attenuation'].rolling(window=5).max()
            df['attenuation_min'] = df['attenuation'].rolling(window=5).min()
            df['attenuation_mean'] = df['attenuation'].rolling(window=5).mean()
            base_feature = 'attenuation'
        else:
            # 如果已经有 attenuation_max, attenuation_min, attenuation_mean 列
            base_feature = 'attenuation_mean'
        
        # 3. 提取时间特征
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        df['quarter'] = df['time'].dt.quarter
        
        # 4. 创建滞后特征
        for lag in range(1, 6):
            df[f'attenuation_lag_{lag}'] = df[base_feature].shift(lag)
        
        # 5. 计算高阶统计特征
        df['attenuation_kurt'] = df[base_feature].rolling(window=5).apply(pd.Series.kurt, raw=False)
        df['attenuation_skew'] = df[base_feature].rolling(window=5).apply(pd.Series.skew, raw=False)
        df['attenuation_std'] = df[base_feature].rolling(window=5).std()
        
        
        # 7. 选择特征
        features = [
            'attenuation_max', 'attenuation_min', 'attenuation_mean',
            'hour', 'day_of_week', 'month', 'quarter',
            'attenuation_skew', 'attenuation_std'
        ] + [f'attenuation_lag_{lag}' for lag in range(1, 6)]
        
        # 8. 数据标准化（只标准化特征，不标准化目标变量）
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[features] = scaler.fit_transform(df[features])

        return df, features

    def post(self, request):
        try:
            train_type = request.data.get('train_type')
            if train_type == 'rain_detection':
                return self.train_rain_detection(request)
            elif train_type == 'rain_inversion':
                return self.train_rain_inversion(request)
            else:
                return Response(
                    {'detail': '不支持的训练类型'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        except Exception as e:
            return Response(
                {'detail': f'训练失败: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

    def train_rain_detection(self, request):
        """晴雨区分训练函数"""
        try:
            # 获取基本参数
            model_architecture = request.data.get('model_architecture')
            dataset_id = request.data.get('dataset_id')
            split_method = request.data.get('split_method', 'random')

            # 验证必要参数
            if not all([model_architecture, dataset_id]):
                return Response(
                    {'detail': '缺少必要参数'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 获取数据集
            try:
                dataset = Dataset.objects.get(id=dataset_id)
            except Dataset.DoesNotExist:
                return Response(
                    {'detail': '数据集不存在'},
                    status=status.HTTP_404_NOT_FOUND
                )

            # 读取数据集
            df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, str(dataset.file)))

            # 生成训练ID
            training_id = f"detection_{model_architecture}_{int(time.time())}"
            channel_layer = get_channel_layer()

            # 数据预处理
            try:
                df, features = self.preprocess_detection_data(df, request.data)
            except Exception as e:
                return Response(
                    {'detail': f'数据预处理失败: {str(e)}'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 准备数据
            X = df[features].values
            y = (df['rainfall_intensity'].values > 0.1).astype(np.float32)  # 二值化
            print(f"Positive samples ratio: {np.mean(y):.4f}")

            # 数据划分
            split_ratio = float(request.data.get('split_ratio', 80)) / 100
            if split_method == 'random':
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=1-split_ratio, random_state=SEED, stratify=y
                )
            else:
                split_idx = int(len(X) * split_ratio)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

            # 根据模型类型进行不同处理
            if model_architecture == 'xgboost':
                return self.train_xgboost_detection(
                    X_train, X_test, y_train, y_test,
                    request.data, training_id, dataset
                )
            else:
                return self.train_dl_detection(
                    X_train, X_test, y_train, y_test,
                    model_architecture, request.data, training_id, dataset
                )

        except Exception as e:
            print(f"Training error: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return Response(
                {'detail': f'训练失败: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

    def train_rain_inversion(self, request):
        """降雨反演训练函数"""
        try:
            # 获取基参数
            model_architecture = request.data.get('model_architecture')
            dataset_id = request.data.get('dataset_id')

            # 获取数据集
            try:
                dataset = Dataset.objects.get(id=dataset_id)
            except Dataset.DoesNotExist:
                return Response(
                    {'detail': '数据集不存在'},
                    status=status.HTTP_404_NOT_FOUND
                )

            # 读取数据集
            df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, str(dataset.file)))

            # 生成训练ID
            training_id = f"inversion_{model_architecture}_{int(time.time())}"

            # 数据预处理
            try:
                df, features = self.preprocess_inversion_data(df, request.data)
            except Exception as e:
                return Response(
                    {'detail': f'数据预处理失败: {str(e)}'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 准备数据
            X = df[features].values
            y = df['rainfall_intensity'].values

            # 对目标变量进行对数变换
            min_nonzero = np.min(y[y > 0])
            epsilon = min_nonzero / 10
            y = np.log1p(y + epsilon)

            # 数据划分
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # 重塑数据
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

            return self.train_dl_inversion(
                X_train, X_test, y_train, y_test,
                model_architecture, request.data, training_id, dataset,
                epsilon
            )

        except Exception as e:
            print(f"Training error: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return Response(
                {'detail': f'训练失败: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

    def preprocess_detection_data(self, df, params):
        """晴雨分的数据预处理函数"""
        try:
            features = []
            
            # 确定可用的衰减特征
            available_features = df.columns.tolist()
            attenuation_features = [col for col in ['attenuation', 'attenuation_max', 'attenuation_min', 'attenuation_mean'] 
                                  if col in available_features]
            
            # 如果只有单一衰减值使用它作为主要特征
            if 'attenuation' in attenuation_features:
                main_attenuation = 'attenuation'
            # 如果有多��衰减特征，优先使��mean，其次使用max和min的平均值
            elif 'attenuation_mean' in attenuation_features:
                main_attenuation = 'attenuation_mean'
            elif all(f in attenuation_features for f in ['attenuation_max', 'attenuation_min']):
                df['attenuation'] = (df['attenuation_max'] + df['attenuation_min']) / 2
                main_attenuation = 'attenuation'
            else:
                raise ValueError("数据集缺少必要衰减特征")

            # 时间特征
            if 'time_features' in params.get('feature_engineering', []):
                df['hour'] = pd.to_datetime(df['time']).dt.hour
                df['day'] = pd.to_datetime(df['time']).dt.day
                df['month'] = pd.to_datetime(df['time']).dt.month
                features.extend(['hour', 'day', 'month'])

            # 统计特征
            if 'statistical_features' in params.get('feature_engineering', []):
                window = int(params.get('rolling_window', 5))
                # 使用主要衰减特征计算统计量
                df['rolling_mean'] = df[main_attenuation].rolling(window=window).mean()
                df['rolling_std'] = df[main_attenuation].rolling(window=window).std()
                df['rolling_max'] = df[main_attenuation].rolling(window=window).max()
                df['rolling_min'] = df[main_attenuation].rolling(window=window).min()
                features.extend(['rolling_mean', 'rolling_std', 'rolling_max', 'rolling_min'])
                
                # 如果有额外的衰减特征，也加入特征集
                for feat in attenuation_features:
                    if feat != main_attenuation:
                        features.append(feat)

            # 滞后征
            if 'lag' in params.get('feature_engineering', []):
                lag_features = params.get('lag_features', [1, 2, 3])
                for lag in lag_features:
                    col_name = f'attenuation_lag_{lag}'
                    df[col_name] = df[main_attenuation].shift(lag)
                    features.append(col_name)

            # 添加主要衰减特征
            features.append(main_attenuation)

            # 处理缺失值
            df = df.fillna(method='ffill').fillna(method='bfill')

            # 数据标准化
            scaler_type = params.get('scaler_type', 'robust')
            if scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'standard':
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()
            
            df[features] = scaler.fit_transform(df[features])

            return df, features

        except Exception as e:
            print(f"Detection preprocessing error: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            raise

    def preprocess_inversion_data(self, df, params):
        """降雨反演的数据预处理函数"""
        try:
            # 1. 提取时间特征
            df['time'] = pd.to_datetime(df['time'])
            df['hour'] = df['time'].dt.hour
            df['day_of_week'] = df['time'].dt.dayofweek
            df['month'] = df['time'].dt.month
            df['quarter'] = df['time'].dt.quarter

            # 2. 创建滞后特征
            for lag in range(1, 6):
                df[f'attenuation_lag_{lag}'] = df['attenuation_mean'].shift(lag)

            # 3. 计算统计特征
            df['attenuation_kurt'] = df['attenuation_mean'].rolling(window=5).apply(pd.Series.kurt, raw=False)
            df['attenuation_skew'] = df['attenuation_mean'].rolling(window=5).apply(pd.Series.skew, raw=False)
            df['attenuation_std'] = df['attenuation_mean'].rolling(window=5).std()

            # 4. 选择特征
            features = [
                'attenuation_max', 'attenuation_min', 'attenuation_mean',
                'hour', 'day_of_week', 'month', 'quarter',
                'attenuation_skew', 'attenuation_std'
            ] + [f'attenuation_lag_{lag}' for lag in range(1, 6)]

            # 5. 处理缺失值
            df = df.dropna()

            # 6. 数据标准化
            scaler = MinMaxScaler()
            X = df[features].values
            X_scaled = scaler.fit_transform(X)
            df[features] = X_scaled

            return df, features

        except Exception as e:
            print(f"Inversion preprocessing error: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            raise

    def train_xgboost_detection(self, X_train, X_test, y_train, y_test, params, training_id, dataset):
        """训练XGBoost分类模型"""
        try:
            # 创建XGBoost模型
            model = xgb.XGBClassifier(
                max_depth=params.get('max_depth', 5),
                learning_rate=params.get('learning_rate', 0.1),
                n_estimators=params.get('n_estimators', 100),
                min_child_weight=params.get('min_child_weight', 1),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                scale_pos_weight=params.get('scale_pos_weight', 1),
                gamma=params.get('gamma', 0),
                reg_alpha=params.get('reg_alpha', 0),
                reg_lambda=params.get('reg_lambda', 1),
                random_state=SEED
            )

            # 创建评估器用于记录训练过程
            eval_set = [(X_train, y_train), (X_test, y_test)]
            eval_metric = ['error', 'auc']
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric=eval_metric,
                verbose=False
            )

            # 获取训练历史
            results = model.evals_result()

            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # 计算评估指标
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred)),
                'f1_score': float(f1_score(y_test, y_pred))
            }

            # 保存模型
            save_dir = os.path.join(settings.MEDIA_ROOT, 'models', 'classification')
            os.makedirs(save_dir, exist_ok=True)
            model_filename = f'model_detection_xgboost_{dataset.id}.json'
            model_path = os.path.join(save_dir, model_filename)
            model.save_model(model_path)

            # 创建模型记录
            model_instance = RainClassificationModel.objects.create(
                name=f'XGBoost_Detection_{dataset.name}',
                description=f'Trained XGBoost model on dataset {dataset.name}',
                model_type='rain_detection',
                training_dataset=dataset,
                parameters={
                    'model_architecture': 'xgboost',
                    'max_depth': params.get('max_depth'),
                    'learning_rate': params.get('learning_rate'),
                    'n_estimators': params.get('n_estimators'),
                    'min_child_weight': params.get('min_child_weight'),
                    'subsample': params.get('subsample'),
                    'colsample_bytree': params.get('colsample_bytree'),
                    'scale_pos_weight': params.get('scale_pos_weight'),
                    'gamma': params.get('gamma'),
                    'reg_alpha': params.get('reg_alpha'),
                    'reg_lambda': params.get('reg_lambda')
                },
                model_file=os.path.join('models', 'classification', model_filename),
                created_by=dataset.created_by,
                **metrics  # 只包含模型字段中定义的指标
            )

            # 创建训练记录
            num_epochs = len(results['validation_0']['error'])
            for epoch in range(num_epochs):
                TrainingRecord.objects.create(
                    model_class=model_instance.__class__.__name__,
                    model_id=model_instance.id,
                    epoch=epoch,
                    loss=float(results['validation_0']['error'][epoch]),
                    metrics={
                        'val_loss': float(results['validation_1']['error'][epoch]),
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1_score'],
                        'auc': float(roc_auc_score(y_test, y_pred_proba))  # 在metrics中包含AUC
                    }
                )

            # 预测所有数据用于可视化
            y_pred_all = model.predict(X_train)
            y_pred_proba_all = model.predict_proba(X_train)[:, 1]

            return Response({
                'message': '模型训练成功',
                'model_id': model_instance.id,
                'training_id': training_id,
                'metrics': {
                    **metrics,
                    'auc': float(roc_auc_score(y_test, y_pred_proba))  # 在响应中包含AUC
                },
                'predictions': [
                    {'true': float(true), 'pred': float(pred)}
                    for true, pred in zip(y_train, y_pred_proba_all)
                ]
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            print(f"XGBoost training error: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return Response(
                {'detail': f'XGBoost训练失败: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

    def train_dl_detection(self, X_train, X_test, y_train, y_test, model_architecture, params, training_id, dataset):
        """训练深度学习分类型"""
        try:
            # 重塑数据
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            input_shape = (1, X_train.shape[2])

            # 构建模型
            if model_architecture == 'transformer':
                # 为Transformer添加默认参数
                params.setdefault('num_layers', 2)
                params.setdefault('num_heads', 4)
                model = self.build_transformer_model(input_shape, params, 'rain_detection')
            elif model_architecture == 'lstm':
                model = self.build_lstm_model(input_shape, params, 'rain_detection')
            elif model_architecture == 'bilstm':
                model = self.build_bilstm_model(input_shape, params, 'rain_detection')
            elif model_architecture == 'gru':
                model = self.build_gru_model(input_shape, params, 'rain_detection')
            else:
                return Response(
                    {'detail': '不支持的模型架构'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 编译模型
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=float(params.get('learning_rate', 0.001))
                ),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # 创建回调函数列表
            callbacks = []
            
            # 添加WebSocket回调
            channel_layer = get_channel_layer()
            callbacks.append(
                TrainingCallback(
                    training_id=training_id,
                    channel_layer=channel_layer
                )
            )

            # 如果启用早停法
            if params.get('use_early_stopping', False):
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,  # 可以通过参数配置
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
                print("Early stopping enabled")  # 添加日志

            # 训练模型
            history = model.fit(
                X_train, y_train,
                epochs=params.get('epochs', 100),
                batch_size=params.get('batch_size', 32),
                validation_split=0.2,
                callbacks=callbacks,  # 使用回调函数列表
                verbose=2
            )

            # 测和评估
            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(float)

            # 计算评估指标
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred_binary)),
                'precision': float(precision_score(y_test, y_pred_binary)),
                'recall': float(recall_score(y_test, y_pred_binary)),
                'f1_score': float(f1_score(y_test, y_pred_binary))
            }

            # 保存模型
            save_dir = os.path.join(settings.MEDIA_ROOT, 'models', 'classification')
            os.makedirs(save_dir, exist_ok=True)
            model_filename = f'model_detection_{model_architecture}_{dataset.id}.h5'
            model_path = os.path.join(save_dir, model_filename)
            model.save_weights(model_path)

            # 创建模型记录
            model_instance = RainClassificationModel.objects.create(
                name=f'{model_architecture}_Detection_{dataset.name}',
                description=f'Trained {model_architecture} model on dataset {dataset.name}',
                model_type='rain_detection',
                training_dataset=dataset,
                parameters={
                    'model_architecture': model_architecture,
                    'input_shape': input_shape,
                    'lstm_layers': params.get('lstm_layers'),
                    'hidden_size': params.get('hidden_size'),
                    'dropout_rate': params.get('dropout_rate'),
                    'batch_size': params.get('batch_size'),
                    'learning_rate': params.get('learning_rate')
                },
                model_file=os.path.join('models', 'classification', model_filename),
                created_by=dataset.created_by,
                **metrics
            )

            # 创建训练记录
            for epoch in range(len(history.history['loss'])):
                TrainingRecord.objects.create(
                    model_class=model_instance.__class__.__name__,
                    model_id=model_instance.id,
                    epoch=epoch,
                    loss=float(history.history['loss'][epoch]),
                    metrics={
                        'val_loss': float(history.history['val_loss'][epoch]),
                        **{
                            k: float(history.history[k][epoch])
                            for k in history.history.keys()
                            if k not in ['loss', 'val_loss']
                        }
                    }
                )

            return Response({
                'message': '模型训练成功',
                'model_id': model_instance.id,
                'training_id': training_id,
                'metrics': metrics
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            print(f"Deep learning training error: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return Response(
                {'detail': f'深度学习训练失败: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

    def train_dl_inversion(self, X_train, X_test, y_train, y_test, model_architecture, params, training_id, dataset, epsilon):
        """训练深度学习回归模型"""
        try:
            # 设置随机种子
            SEED = 42
            tf.random.set_seed(SEED)
            np.random.seed(SEED)

            input_shape = (1, X_train.shape[2])
            
            # 构建模型，使用固定的隐藏层大小
            model = Sequential([
                Bidirectional(
                    LSTM(100, return_sequences=True),  # 固定第一层大小为100
                    input_shape=input_shape
                ),
                Dropout(params.get('dropout_rate', 0.3)),
                
                Bidirectional(
                    LSTM(100, return_sequences=True)  # 固定第二层大小为100
                ),
                Dropout(params.get('dropout_rate', 0.3)),
                
                Bidirectional(
                    LSTM(50)  # 固定第三层大小为50
                ),
                Dropout(params.get('dropout_rate', 0.3)),
                
                Dense(1)
            ])

            # 编译模型
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # 定义回调函数
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )

            # 创建回调函数列表
            callbacks = []
            
            # 添加WebSocket回调
            channel_layer = get_channel_layer()
            callbacks.append(
                TrainingCallback(
                    training_id=training_id,
                    channel_layer=channel_layer
                )
            )

            # 如果启用早停法
            if params.get('use_early_stopping', False):
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
                print("Early stopping enabled")

            # 训练模型
            history = model.fit(
                X_train, y_train,
                epochs=params.get('epochs', 200),
                batch_size=params.get('batch_size', 32),
                validation_split=params.get('validation_split', 0.2),
                verbose=2,
                callbacks=[early_stopping, reduce_lr]
            )

            # 预测和评估
            y_pred = model.predict(X_test)

            # 反转对数变换
            y_test_original = np.expm1(y_test) - epsilon
            y_pred_original = np.expm1(y_pred) - epsilon

            # 将负值设为0
            y_test_original = np.maximum(y_test_original, 0)
            y_pred_original = np.maximum(y_pred_original, 0)

            # 计算评估指标
            mse = mean_squared_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_original, y_pred_original)
            r2 = r2_score(y_test_original, y_pred_original)
            correlation = pearsonr(y_test_original.ravel(), y_pred_original.ravel())[0]

            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2_score': float(r2),
                'correlation': float(correlation)
            }

            print("\n详细评估指标:")
            print(f"MSE: {mse:.6f}")
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"R²: {r2:.6f}")
            print(f"相关系数: {correlation:.6f}")

            # 保存模型
            save_dir = os.path.join(settings.MEDIA_ROOT, 'models', 'regression')
            os.makedirs(save_dir, exist_ok=True)
            model_filename = f'model_inversion_{model_architecture}_{dataset.id}.h5'
            model_path = os.path.join(save_dir, model_filename)
            model.save_weights(model_path)

            # 创建模型记录
            model_instance = RainRegressionModel.objects.create(
                name=f'{model_architecture}_Inversion_{dataset.name}',
                description=f'Trained {model_architecture} model on dataset {dataset.name}',
                model_type='rain_inversion',
                training_dataset=dataset,
                parameters={
                    'model_architecture': model_architecture,
                    'input_shape': input_shape,
                    'lstm_layers': params.get('lstm_layers'),
                    'hidden_size': params.get('hidden_size'),
                    'dropout_rate': params.get('dropout_rate'),
                    'batch_size': params.get('batch_size'),
                    'learning_rate': params.get('learning_rate'),
                    'epsilon': float(epsilon)  # 保存epsilon用于后预测
                },
                model_file=os.path.join('models', 'regression', model_filename),
                created_by=dataset.created_by,
                **metrics
            )

            # 创建训练记录
            for epoch in range(len(history.history['loss'])):
                TrainingRecord.objects.create(
                    model_class=model_instance.__class__.__name__,
                    model_id=model_instance.id,
                    epoch=epoch,
                    loss=float(history.history['loss'][epoch]),
                    metrics={
                        'val_loss': float(history.history['val_loss'][epoch]) if 'val_loss' in history.history else 0,
                        'mae': float(history.history['mae'][epoch]) if 'mae' in history.history else 0,
                        'val_mae': float(history.history['val_mae'][epoch]) if 'val_mae' in history.history else 0
                    }
                )

            return Response({
                'message': '模型训练成功',
                'model_id': model_instance.id,
                'training_id': training_id,
                'metrics': metrics
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            print(f"Training error: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return Response(
                {'detail': f'深度学习训练失败: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

class ModelListView(APIView):
    def get(self, request):
        try:
            # 获取查询参数
            model_type = request.query_params.get('type', 'all')

            # 获取当前用户的分类模型
            classification_models = RainClassificationModel.objects.filter(created_by=request.user)
            # 获取当前用户的回归模型
            regression_models = RainRegressionModel.objects.filter(created_by=request.user)

            # 根据类型过滤
            if model_type == 'rain_detection':
                regression_models = RainRegressionModel.objects.none()
            elif model_type == 'rain_inversion':
                classification_models = RainClassificationModel.objects.none()

            # 序列化
            classification_serializer = RainClassificationModelSerializer(classification_models, many=True)
            regression_serializer = RainRegressionModelSerializer(regression_models, many=True)

            return Response({
                'classification_models': classification_serializer.data,
                'regression_models': regression_serializer.data
            })
        except Exception as e:
            return Response(
                {'detail': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

class ModelDetailView(ViewSet):
    def retrieve(self, request, pk=None):
        try:
            # 先尝获取分类模型
            try:
                model = RainClassificationModel.objects.get(pk=pk)
                serializer = RainClassificationModelSerializer(model)
                return Response(serializer.data)
            except RainClassificationModel.DoesNotExist:
                pass
            
            # 如果不是分类模型，尝试获取回归模型
            try:
                model = RainRegressionModel.objects.get(pk=pk)
                serializer = RainRegressionModelSerializer(model)
                return Response(serializer.data)
            except RainRegressionModel.DoesNotExist:
                return Response({
                    'detail': '模型不存在'
                }, status=status.HTTP_404_NOT_FOUND)

        except Exception as e:
            return Response({
                'detail': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

    def list(self, request):
        try:
            classification_models = RainClassificationModel.objects.all()
            regression_models = RainRegressionModel.objects.all()

            classification_serializer = RainClassificationModelSerializer(classification_models, many=True)
            regression_serializer = RainRegressionModelSerializer(regression_models, many=True)

            return Response({
                'classification_models': classification_serializer.data,
                'regression_models': regression_serializer.data
            })
        except Exception as e:
            return Response({
                'detail': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        try:
            # 尝试删除分类模型
            try:
                model = RainClassificationModel.objects.get(pk=pk)
            except RainClassificationModel.DoesNotExist:
                # 如果不是分类模型，尝试删除回归模型
                model = RainRegressionModel.objects.get(pk=pk)

            # 删除模型文件
            if model.model_file:
                file_path = os.path.join(settings.MEDIA_ROOT, str(model.model_file))
                if os.path.exists(file_path):
                    os.remove(file_path)

            # 删除数据库记录
            model.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Exception as e:
            return Response({
                'detail': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

    def get_model_file(self, request, pk=None):
        try:
            # 尝试获取分类模型
            try:
                model = RainClassificationModel.objects.get(pk=pk)
            except RainClassificationModel.DoesNotExist:
                # 如果不是分类模型，尝试获取回归模型
                model = RainRegressionModel.objects.get(pk=pk)

            # 模型文件路径
            file_path = os.path.join(settings.MEDIA_ROOT, str(model.model_file))
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    response = HttpResponse(f.read(), content_type='application/octet-stream')
                    response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
                    return response
            else:
                return Response({
                    'detail': '模型文件不存在'
                }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'detail': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_model_predictions(request, model_id):
    try:
        # 尝试获取模型
        try:
            model = RainRegressionModel.objects.get(id=model_id)
            is_regression = True
        except RainRegressionModel.DoesNotExist:
            try:
                model = RainClassificationModel.objects.get(id=model_id)
                is_regression = False
            except RainClassificationModel.DoesNotExist:
                return Response(
                    {'detail': '模型不存在'},
                    status=status.HTTP_404_NOT_FOUND
                )

        # 获取训练记录
        training_records = TrainingRecord.objects.filter(
            model_class=model.__class__.__name__,
            model_id=model_id
        ).order_by('epoch')

        # 读取数据集
        dataset = model.training_dataset
        if not dataset:
            return Response(
                {'detail': '训练数据集不存在'},
                status=status.HTTP_404_NOT_FOUND
            )

        # 读取数据集文件
        df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, str(dataset.file)))
        
        # 数据预处理
        model_view = ModelTrainView()
        if is_regression:
            df, features = model_view.preprocess_inversion_data(df, model.parameters)
            y = df['rainfall_intensity'].values
            
            # 重塑数据
            X = df[features].values
            X = X.reshape((X.shape[0], 1, X.shape[1]))
            epsilon = model.parameters.get('epsilon', 1e-6)
            y = np.log1p(y + epsilon)

            # 加载模型
            model_path = os.path.join(settings.MEDIA_ROOT, str(model.model_file))
            input_shape = (1, len(features))
            loaded_model = model_view.build_model(
                model.parameters.get('model_architecture'),
                input_shape,
                model.parameters
            )
            loaded_model.load_weights(model_path)
            
            # 进行预测
            y_pred = loaded_model.predict(X)
            # 反变换预测结果
            y_pred = np.expm1(y_pred.flatten()) - epsilon
            y = np.expm1(y) - epsilon
        else:  # RainClassificationModel
            # 使用与训练时相同的参数进行预处理
            params = model.parameters.copy()
            # 确保包含所有必要的特征工程参数
            params.setdefault('feature_engineering', ['time_features', 'statistical_features', 'lag'])
            params.setdefault('rolling_window', 5)
            params.setdefault('lag_features', [1, 2, 3])
            params.setdefault('scaler_type', 'robust')
            
            df, features = model_view.preprocess_detection_data(df, params)
            y = (df['rainfall_intensity'].values > 0.1).astype(np.float32)
            X = df[features].values

            # 加载模型
            model_path = os.path.join(settings.MEDIA_ROOT, str(model.model_file))
            if model.parameters.get('model_architecture') == 'xgboost':
                loaded_model = xgb.XGBClassifier()
                loaded_model.load_model(model_path)
                # 使用原始的分类标签
                y_pred = loaded_model.predict(X)
            else:
                input_shape = (1, len(features))
                loaded_model = model_view.build_model(
                    model.parameters.get('model_architecture'),
                    input_shape,
                    model.parameters,
                    'rain_detection'
                )
                loaded_model.load_weights(model_path)
                X = X.reshape((X.shape[0], 1, X.shape[1]))
                y_pred = (loaded_model.predict(X).flatten() > 0.5).astype(np.float32)

        # 返回预测结果和训练记录
        return Response({
            'predictions': [
                {'true': float(true), 'pred': float(pred)}
                for true, pred in zip(y, y_pred)
            ],
            'training_records': [
                {
                    'epoch': record.epoch,
                    'loss': float(record.loss),
                    'metrics': record.metrics or {}
                }
                for record in training_records
            ]
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return Response(
            {'detail': f'获取预测结果失败: {str(e)}'},
            status=status.HTTP_400_BAD_REQUEST
        )

def build_lstm_detection_model(self, input_shape, params):
    """构建晴雨区分的LSTM模型"""
    model = tf.keras.Sequential([
        # 第一个LSTM层，增加单元数
        tf.keras.layers.LSTM(
            units=params.get('hidden_size', 128),  # 增加默认单元数
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)  # 添加L2正则化
        ),
        tf.keras.layers.BatchNormalization(),  # 添加批标准化
        tf.keras.layers.Dropout(params.get('dropout_rate', 0.3)),
        
        # 第二个LSTM层
        tf.keras.layers.LSTM(
            units=params.get('hidden_size', 128),
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(params.get('dropout_rate', 0.3)),
        
        # 全连接层
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        
        # 输出层
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_bilstm_detection_model(self, input_shape, params):
    """构建晴雨区分的BiLSTM模型"""
    model = tf.keras.Sequential([
        # 第一个BiLSTM层
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=params.get('hidden_size', 128),
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            input_shape=input_shape
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(params.get('dropout_rate', 0.3)),
        
        # 第二个BiLSTM层
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=params.get('hidden_size', 128),
                return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            )
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(params.get('dropout_rate', 0.3)),
        
        # 注意力层
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(1, activation='softmax'),
        tf.keras.layers.Multiply(),
        
        # 全连接层
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        
        # 输出层
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_gru_detection_model(self, input_shape, params):
    """构建晴雨区分的GRU模型"""
    model = tf.keras.Sequential([
        # 第一个GRU层
        tf.keras.layers.GRU(
            units=params.get('hidden_size', 128),
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            recurrent_dropout=0.1  # 添加循环层dropout
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(params.get('dropout_rate', 0.3)),
        
        # 第二个GRU层
        tf.keras.layers.GRU(
            units=params.get('hidden_size', 128),
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            recurrent_dropout=0.1
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(params.get('dropout_rate', 0.3)),
        
        # 第三个GRU层
        tf.keras.layers.GRU(
            units=params.get('hidden_size', 128),
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            recurrent_dropout=0.1
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(params.get('dropout_rate', 0.3)),
        
        # 全连接层
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        
        # 输出层
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model
