from django.urls import path
from .views import (
    ModelTrainView,
    ModelListView,
    ModelDetailView,
    get_model_predictions
)

app_name = 'model'

urlpatterns = [
    path('train/', ModelTrainView.as_view()),
    path('models/', ModelListView.as_view()),
    path('models/<int:pk>/', ModelDetailView.as_view({'get': 'retrieve', 'delete': 'destroy'})),
    path('models/<int:pk>/download/', ModelDetailView.as_view({'get': 'get_model_file'})),
    path('models/<int:model_id>/predictions/', get_model_predictions, name='model-predictions'),
] 