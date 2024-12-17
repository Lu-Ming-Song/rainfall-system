from rest_framework import serializers
from .models import RainClassificationModel, RainRegressionModel, TrainingRecord
from apps.system.serializers import UserSerializer
from apps.data.serializers import DatasetSerializer

class TrainingRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingRecord
        fields = '__all__'

class BaseModelSerializer(serializers.ModelSerializer):
    created_by = UserSerializer(read_only=True)
    training_dataset = DatasetSerializer(read_only=True)
    training_records = serializers.SerializerMethodField()

    def get_training_records(self, obj):
        records = TrainingRecord.objects.filter(
            model_class=obj.__class__.__name__,
            model_id=obj.id
        ).order_by('epoch')
        return TrainingRecordSerializer(records, many=True).data

    class Meta:
        fields = [
            'id', 'name', 'description', 'model_type', 
            'training_dataset', 'parameters', 'created_by', 
            'created_time', 'updated_time', 'training_records'
        ]

class RainClassificationModelSerializer(BaseModelSerializer):
    class Meta(BaseModelSerializer.Meta):
        model = RainClassificationModel
        fields = BaseModelSerializer.Meta.fields + [
            'accuracy', 'precision', 'recall', 'f1_score'
        ]

class RainRegressionModelSerializer(BaseModelSerializer):
    class Meta(BaseModelSerializer.Meta):
        model = RainRegressionModel
        fields = BaseModelSerializer.Meta.fields + [
            'mse', 'rmse', 'mae', 'r2_score', 'correlation'
        ] 