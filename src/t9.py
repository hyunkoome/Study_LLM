import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])
import wandb

if __name__ == '__main__':
    wandb.login()
    wandb.init(project='t9')