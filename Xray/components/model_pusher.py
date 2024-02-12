import os
import sys

from Xray.entity.artifact_entity import ModelPusherArtifact
from Xray.entity.config_entity import ModelPusherConfig
from Xray.exception import XRayException
from Xray.logger import logging


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config

    def build_and_push_bento_image(self):
        logging.info("Entered build_and_push_bento_image method of ModelPusher class")

        try:
            logging.info("Building the bento from bentofile.yaml")

            os.system("bentoml build")

            logging.info("Built the bento from bentofile.yaml")

            logging.info("Creating docker image for bento")

            os.system(
                f"bentoml containerize {self.model_pusher_config.bentoml_service_name}:latest -t 136566696263.dkr.ecr.us-east-1.amazonaws.com/{self.model_pusher_config.bentoml_ecr_image}:latest"
            )

            logging.info("Created docker image for bento")

            logging.info("Logging into ECR")

            os.system(
                "aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 136566696263.dkr.ecr.us-east-1.amazonaws.com"
            )

            logging.info("Logged into ECR")

            logging.info("Pushing bento image to ECR")

            os.system(
                f"docker push 136566696263.dkr.ecr.us-east-1.amazonaws.com/{self.model_pusher_config.bentoml_ecr_image}:latest"
            )

            logging.info("Pushed bento image to ECR")

            logging.info(
                "Exited build_and_push_bento_image method of ModelPusher class"
            )

        except Exception as e:
            raise XRayException(e, sys)
        


    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This method initiates model pusher.

        Output      :   Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            self.build_and_push_bento_image()

            model_pusher_artifact = ModelPusherArtifact(
                bentoml_model_name=self.model_pusher_config.bentoml_model_name,
                bentoml_service_name=self.model_pusher_config.bentoml_service_name,
            )

            logging.info("Exited the initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise XRayException(e, sys)
