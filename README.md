# deeplearningproject

Lung X-ray Classification
Data link: https://drive.google.com/file/d/1pfIAlurfeqFTbirUZ5v_vapIoGPgRiXY/view?usp=sharing

Workflows
constants
config_enity
artifact_enity
components
pipeline
main
How to setup:
conda create -n lungs python=3.8 -y
conda activate lungs
pip install -r requirements.txt
setup AWS CLI
link: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
aws configure
AWS_ACCESS_KEY_ID=***

AWS_SECRET_ACCESS_KEY= ***

AWS_REGION = us-east-1
BentoML demo repo:
https://github.com/entbappy/bentoml-demo