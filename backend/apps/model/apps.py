from django.apps import AppConfig


class ModelConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.model'
    verbose_name = '模型管理'
