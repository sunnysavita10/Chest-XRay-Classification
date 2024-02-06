import sys


from Xray.cloud_storage.s3_operation import S3Operation
#from Xray.constant.training_pipeline import *
from Xray.entity.artifact_entity import DataIngestionArtifact
from Xray.entity.config_entity import DataIngestionConfig
from Xray.exception import XRayException
from Xray.logger import logging

class DataIngestion:
    def __init__(self):
        pass
    def get_data_from_s3(self):
        try:
            pass
        except Exception as e:
            raise XRayException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            pass
        except Exception as e:
            raise XRayException(e,sys)