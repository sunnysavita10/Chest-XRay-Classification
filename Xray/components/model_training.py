import os
import sys

import bentoml
import joblib
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from tqdm import tqdm

from Xray.constant.training_pipeline import *
from Xray.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from Xray.entity.config_entity import ModelTrainerConfig
from Xray.exception import XRayException
from Xray.logger import logging
from Xray.ml.model.arch import Net




class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config: ModelTrainerConfig = model_trainer_config

        self.data_transformation_artifact: DataTransformationArtifact = (
            data_transformation_artifact
        )

        self.model: Module = Net()




    def train(self, optimizer: Optimizer) -> None:
        """
        Description: To train the model

        input: model,device,train_loader,optimizer,epoch

        output: loss, batch id and accuracy
        """
        logging.info("Entered the train method of Model trainer class")

        try:
            self.model.train()

            pbar = tqdm(self.data_transformation_artifact.transformed_train_object)

            correct: int = 0

            processed = 0

            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(DEVICE), target.to(DEVICE)

                # Initialization of gradient
                optimizer.zero_grad()

                # In PyTorch, gradient is accumulated over backprop and even though thats used in RNN generally not used in CNN
                # or specific requirements
                ## prediction on data

                y_pred = self.model(data)

                # Calculating loss given the prediction
                loss = F.nll_loss(y_pred, target)

                # Backprop
                loss.backward()

                optimizer.step()

                # get the index of the log-probability corresponding to the max value
                pred = y_pred.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

                processed += len(data)

                pbar.set_description(
                    desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
                )

            logging.info("Exited the train method of Model trainer class")

        except Exception as e:
            raise XRayException(e, sys)
        



    def test(self) -> None:
        try:
            """
            Description: To test the model

            input: model, DEVICE, test_loader

            output: average loss and accuracy

            """
            logging.info("Entered the test method of Model trainer class")

            self.model.eval()

            test_loss: float = 0.0

            correct: int = 0

            with torch.no_grad():
                for (
                    data,
                    target,
                ) in self.data_transformation_artifact.transformed_test_object:
                    data, target = data.to(DEVICE), target.to(DEVICE)

                    output = self.model(data)

                    test_loss += F.nll_loss(output, target, reduction="sum").item()

                    pred = output.argmax(dim=1, keepdim=True)

                    correct += pred.eq(target.view_as(pred)).sum().item()

                test_loss /= len(
                    self.data_transformation_artifact.transformed_test_object.dataset
                )

                print(
                    "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                        test_loss,
                        correct,
                        len(
                            self.data_transformation_artifact.transformed_test_object.dataset
                        ),
                        100.0
                        * correct
                        / len(
                            self.data_transformation_artifact.transformed_test_object.dataset
                        ),
                    )
                )

            logging.info(
                "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                    test_loss,
                    correct,
                    len(
                        self.data_transformation_artifact.transformed_test_object.dataset
                    ),
                    100.0
                    * correct
                    / len(
                        self.data_transformation_artifact.transformed_test_object.dataset
                    ),
                )
            )

            logging.info("Exited the test method of Model trainer class")

        except Exception as e:
            raise XRayException(e, sys)
        

        

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(
                "Entered the initiate_model_trainer method of Model trainer class"
            )

            model: Module = self.model.to(self.model_trainer_config.device)

            optimizer: Optimizer = torch.optim.SGD(
                model.parameters(), **self.model_trainer_config.optimizer_params
            )

            scheduler: _LRScheduler = StepLR(
                optimizer=optimizer, **self.model_trainer_config.scheduler_params
            )

            for epoch in range(1, self.model_trainer_config.epochs + 1):
                print("Epoch : ", epoch)

                self.train(optimizer=optimizer)

                optimizer.step()

                scheduler.step()

                self.test()

            os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)

            torch.save(model, self.model_trainer_config.trained_model_path)

            train_transforms_obj = joblib.load(
                self.data_transformation_artifact.train_transform_file_path
            )

            bentoml.pytorch.save_model(
                name=self.model_trainer_config.trained_bentoml_model_name,
                model=model,
                custom_objects={
                    self.model_trainer_config.train_transforms_key: train_transforms_obj
                },
            )

            model_trainer_artifact: ModelTrainerArtifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path
            )

            logging.info(
                "Exited the initiate_model_trainer method of Model trainer class"
            )

            return model_trainer_artifact

        except Exception as e:
            raise XRayException(e, sys)