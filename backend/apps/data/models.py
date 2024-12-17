from django.db import models
from django.conf import settings

# Create your models here.

class Dataset(models.Model):
    """数据集模型"""
    STATUS_CHOICES = (
        ('raw', '原始数据'),
        ('processed', '已处理'),
        ('training', '训练集'),
        ('validation', '验证集'),
        ('testing', '测试集'),
    )

    name = models.CharField(max_length=100, verbose_name='数据集名称')
    description = models.TextField(blank=True, verbose_name='数据集描述')
    file = models.FileField(upload_to='datasets/%Y/%m/', verbose_name='数据文件')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='raw', verbose_name='数据状态')
    columns = models.JSONField(default=dict, verbose_name='列信息')
    row_count = models.IntegerField(default=0, verbose_name='行数')
    missing_values = models.JSONField(default=dict, verbose_name='缺失值信息')
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        verbose_name='创建者'
    )
    created_time = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_time = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '数据集'
        verbose_name_plural = verbose_name
        ordering = ['-created_time']

    def __str__(self):
        return self.name

class DataProcessingRecord(models.Model):
    """数据处理记录"""
    PROCESSING_TYPE_CHOICES = (
        ('clean', '数据清洗'),
        ('normalize', '归一化'),
        ('standardize', '标准化'),
        ('split', '数据集划分'),
    )

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, verbose_name='数据集')
    processing_type = models.CharField(max_length=20, choices=PROCESSING_TYPE_CHOICES, verbose_name='处理类型')
    parameters = models.JSONField(default=dict, verbose_name='处理参数')
    result = models.JSONField(default=dict, verbose_name='处理结果')
    processed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        verbose_name='处理者'
    )
    processed_time = models.DateTimeField(auto_now_add=True, verbose_name='处理时间')

    class Meta:
        verbose_name = '数据处理记录'
        verbose_name_plural = verbose_name
        ordering = ['-processed_time']

    def __str__(self):
        return f"{self.dataset.name} - {self.get_processing_type_display()}"
