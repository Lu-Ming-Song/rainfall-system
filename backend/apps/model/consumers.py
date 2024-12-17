import json
from channels.generic.websocket import AsyncWebsocketConsumer

class TrainingProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.training_id = self.scope['url_route']['kwargs']['training_id']
        self.group_name = f'training_{self.training_id}'

        # Join room group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        pass

    async def training_progress(self, event):
        # Send training progress to WebSocket
        try:
            await self.send(text_data=json.dumps({
                'type': 'training_progress',
                'data': event['data']
            }))
        except Exception as e:
            print(f"Error sending training progress: {str(e)}") 