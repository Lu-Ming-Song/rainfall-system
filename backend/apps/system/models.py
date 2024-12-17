from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _

# Create your models here.

class User(AbstractUser):
    """用户模型"""
    phone = models.CharField(max_length=11, blank=True, verbose_name='手机号')
    role = models.ForeignKey('Role', on_delete=models.SET_NULL, null=True, blank=True, verbose_name='角色')
    
    class Meta:
        verbose_name = '用户'
        verbose_name_plural = verbose_name
        ordering = ['-date_joined']

class Role(models.Model):
    """角色模型"""
    name = models.CharField(max_length=32, unique=True, verbose_name='角色名称')
    desc = models.CharField(max_length=128, blank=True, verbose_name='角色描述')
    permissions = models.ManyToManyField('Permission', blank=True, verbose_name='权限')
    created_time = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_time = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '角色'
        verbose_name_plural = verbose_name
        ordering = ['id']

class Permission(models.Model):
    """权限模型"""
    name = models.CharField(max_length=32, unique=True, verbose_name='权限名称')
    code = models.CharField(max_length=32, unique=True, verbose_name='权限代码')
    desc = models.CharField(max_length=128, blank=True, verbose_name='权限描述')
    created_time = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_time = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '权限'
        verbose_name_plural = verbose_name
        ordering = ['id']

class SystemSetting(models.Model):
    """系统设置模型"""
    key = models.CharField(max_length=32, unique=True, verbose_name='设置项')
    value = models.JSONField(verbose_name='设置值')
    desc = models.CharField(max_length=128, blank=True, verbose_name='设置描述')
    created_time = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_time = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '系统设置'
        verbose_name_plural = verbose_name
        ordering = ['id']
