import os

class Config:
    AZURE_INFERENCE_CREDENTIAL = os.environ.get('AZURE_INFERENCE_CREDENTIAL')
    SECRET_KEY = os.environ.get('SECRET_KEY')