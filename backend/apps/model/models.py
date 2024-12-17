from django.db import models
from django.conf import settings
from apps.data.models import Dataset

# Create your models here.

class BaseModel(models.Model):
    """模型基类"""
    MODEL_TYPE_CHOICES = (
        ('classification', '晴雨判别'),
        ('regression', '降雨反演'),
    )

    STATUS_CHOICES = (
        ('success', '训练成功'),
        ('failed', '训练失败'),
        ('training', '训练中'),
    )

    name = models.CharField(max_length=100, verbose_name='模型名称')
    description = models.TextField(blank=True, verbose_name='模型描述')
    model_type = models.CharField(max_length=20, choices=MODEL_TYPE_CHOICES, verbose_name='模型类型')
    training_dataset = models.ForeignKey(
        Dataset, 
        on_delete=models.SET_NULL,
        null=True,
        related_name='%(class)s_training_set',
        verbose_name='训练数据集'
    )
    validation_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.SET_NULL,
        null=True,
        related_name='%(class)s_validation_set',
        verbose_name='验证数据集'
    )
    parameters = models.JSONField(default=dict, verbose_name='模型参数')
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        verbose_name='创建者'
    )
    created_time = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_time = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    training_status = models.CharField(
        max_length=20, 
        choices=STATUS_CHOICES, 
        default='training',
        verbose_name='训练状态'
    )

    class Meta:
        abstract = True

class RainClassificationModel(BaseModel):
    """晴雨判别模型"""
    accuracy = models.FloatField(default=0.0, verbose_name='准确率')
    precision = models.FloatField(default=0.0, verbose_name='精确率')
    recall = models.FloatField(default=0.0, verbose_name='召回率')
    f1_score = models.FloatField(default=0.0, verbose_name='F1分数')
    model_file = models.FileField(upload_to='models/classification/%Y/%m/', verbose_name='模型文件')

    class Meta:
        verbose_name = '晴雨判别模型'
        verbose_name_plural = verbose_name
        ordering = ['-created_time']

    def __str__(self):
        return f"{self.name} - {self.get_model_type_display()}"

class RainRegressionModel(BaseModel):
    """降雨反演模型"""
    mse = models.FloatField(default=0.0, verbose_name='均方误差')
    rmse = models.FloatField(default=0.0, verbose_name='均方根误差')
    r2_score = models.FloatField(default=0.0, verbose_name='R方分数')
    mae = models.FloatField(default=0.0, verbose_name='平均绝对误差')
    correlation = models.FloatField(default=0.0, verbose_name='相关系数')
    model_file = models.FileField(upload_to='models/regression/%Y/%m/', verbose_name='模型文件')

    class Meta:
        verbose_name = '降雨反演模型'
        verbose_name_plural = verbose_name
        ordering = ['-created_time']

    def __str__(self):
        return f"{self.name} - {self.get_model_type_display()}"

class TrainingRecord(models.Model):
    """模型训练记录"""
    MODEL_CLASS_CHOICES = (
        ('RainClassificationModel', '晴雨判别模型'),
        ('RainRegressionModel', '降雨反演模型'),
    )

    model_class = models.CharField(max_length=30, choices=MODEL_CLASS_CHOICES, verbose_name='模型类别')
    model_id = models.IntegerField(verbose_name='模型ID')
    epoch = models.IntegerField(default=0, verbose_name='训练轮次')
    loss = models.FloatField(verbose_name='损失值')
    metrics = models.JSONField(default=dict, verbose_name='评估指标')
    created_time = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        verbose_name = '训练记录'
        verbose_name_plural = verbose_name
        ordering = ['model_class', 'model_id', 'epoch']

    def __str__(self):
        return f"{self.get_model_class_display()} - Epoch {self.epoch}"
