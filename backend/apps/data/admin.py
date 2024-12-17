from django.contrib import admin
from .models import Dataset, DataProcessingRecord

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'status', 'row_count', 'created_by', 'created_time')
    list_filter = ('status', 'created_time')
    search_fields = ('name', 'description')

@admin.register(DataProcessingRecord)
class DataProcessingRecordAdmin(admin.ModelAdmin):
    list_display = ('dataset', 'processing_type', 'processed_by', 'processed_time')
    list_filter = ('processing_type', 'processed_time')
    search_fields = ('dataset__name',)
