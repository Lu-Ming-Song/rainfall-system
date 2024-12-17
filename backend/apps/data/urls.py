from django.urls import path
from .views import (
    DatasetUploadView, DatasetListView,
    DatasetDetailView, DatasetDataView,
    DatasetExportView
)

app_name = 'data'

urlpatterns = [
    path('datasets/upload/', DatasetUploadView.as_view()),
    path('datasets/', DatasetListView.as_view()),
    path('datasets/<int:pk>/', DatasetDetailView.as_view()),
    path('datasets/<int:pk>/data/', DatasetDataView.as_view()),
    path('datasets/<int:pk>/data/<int:row_id>/', DatasetDataView.as_view()),
    path('datasets/<int:pk>/export/', DatasetExportView.as_view()),
    path('export/', DatasetExportView.as_view()),
] 