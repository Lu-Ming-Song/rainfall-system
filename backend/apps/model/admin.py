from django.contrib import admin
from .models import RainClassificationModel, RainRegressionModel, TrainingRecord

@admin.register(RainClassificationModel)
class RainClassificationModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'model_type', 'accuracy', 'created_by', 'created_time')
    list_filter = ('model_type', 'created_time')
    search_fields = ('name', 'description')

@admin.register(RainRegressionModel)
class RainRegressionModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'model_type', 'r2_score', 'created_by', 'created_time')
    list_filter = ('model_type', 'created_time')
    search_fields = ('name', 'description')

@admin.register(TrainingRecord)
class TrainingRecordAdmin(admin.ModelAdmin):
    list_display = ('model_class', 'model_id', 'epoch', 'loss', 'created_time')
    list_filter = ('model_class', 'created_time')
    search_fields = ('model_id',)
