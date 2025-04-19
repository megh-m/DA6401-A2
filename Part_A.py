import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms #To manage dataset
from torch.utils.data import DataLoader, Subset #To load and transform data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger # As pytorch.lightning allows direct logging into wandb
#from sklearn.model_selection import StratifiedShuffleSplit '''Check with TAs'''

class CNN(pl.LightningModule):
    def __init__(self,config):
        super().__init__() #using LightningModule to assign hyperparams
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        self.final_ch, self.final_size = self.calc_output_dim(config)

        self.build_nn(config)
    
    def build_nn(self,config):
        self.conv_blk = nn.ModuleList()
        input_ch = 3 #RGB images
        current_filters = config.filter_base #Filter Number & Strategies from sweep config
        for i in range(5): #Create five consecutive convolution blocks with configurable strategies
            self.conv_blk.append(
                nn.Conv2d(input_ch, current_filters, kernel_size = config.filter_size, padding = config.filter_size//2) #Use filter size from sweep config, padding = floor(0.5*filter)
            )
            self.conv_blk.append(self.actv(config.conv_actv))
            self.conv_blk.append(nn.MaxPool2d(kernel_size = config.pool_size))#, padding = config.pool_size//2))
            #Define organization logic for filters in subsequent layers from sweep config metric filter_org
            if config.filter_org == "double":
                input_ch = current_filters
                current_filters *= 2 #Double number of filters
            elif config.filter_org == "halve":
                input_ch = current_filters
                current_filters = max(8, current_filters//2) #Ensuring a minimum of 8 filters
            else: #No filter strategies
                input_ch = current_filters
          
        self.dense = nn.Sequential(
              nn.Linear(self.final_ch*self.final_size**2, config.dense_neurons),
              self.actv(config.dense_actv),
              nn.Dropout(config.dropout),
              nn.Linear(config.dense_neurons, 10)
          )

    #Calculate what the dimensions for the dense layer inputs would turn out to be
    def calc_output_dim(self,config):
          with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            input_ch = 3
            current_filters = config.filter_base
            spatial_size =224

            for i in range(5):
                conv = nn.Conv2d(
                    input_ch, 
                    current_filters,
                    kernel_size=config.filter_size,
                    padding=config.filter_size//2
                )
                dummy = conv(dummy)
                dummy = self.actv(config.conv_actv)(dummy)

                dummy = F.max_pool2d(dummy, kernel_size = config.pool_size)
                spatial_size = spatial_size // config.pool_size
                #Error handling due to multiple errors lol
                if spatial_size<1:
                  raise ValueError(
                      f"Pool Size {config.pool_size} wrong for 5 convs"
                      f"Max allowed: {224**(1/5):.0f}"
                  )

                # Update filter organization
                if config.filter_org == "double":
                    input_ch = current_filters
                    current_filters *= 2
                elif config.filter_org == "halve":
                    input_ch = current_filters
                    current_filters = max(8, current_filters // 2)
                else:
                    input_ch = current_filters
            print(dummy.shape[1], dummy.shape[2])        
            return dummy.shape[1], dummy.shape[2]
    def actv(self,name): #Choosing activation func
        actv={
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish()
        }
        return actv[name.lower()]

    def forward(self,x): #Explicit Lightning Module Method Forward
        for layer in self.conv_blk:
            x = layer(x)
        x = x.view(x.size(0), -1) #Flatten
        return self.dense(x) #Invoke Dense layers

    def training_step(self,batch,batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss",loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat,y)
        acc = (y_hat.argmax(dim=1) == y).float().mean() #Mean fractional accuracy
        self.log("val_loss",loss)
        self.log("val_acc", acc)
        return {"val_loss":loss, "val_acc":acc}

    def configure_optimizers(self): #Another pl method. We use Adam optim
        return torch.optim.Adam(self.parameters(), lr = self.config.eta) #have to use self.hparams ,if used, as referring config outside the __init__

class DataManager(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transform = self.tfms()

    def tfms(self): #Preprocess input data. Add (Augment) data to make model robust
        base_transform = [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ]
        if self.config.data_augmentation==True:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
                *base_transform
            ])
        else:
            train_transform = transforms.Compose(base_transform)
        
        return {
            "train": train_transform,
            "val": transforms.Compose(base_transform),
            "test": transforms.Compose(base_transform)
        }

    def setup(self, stage=None):
        full_data = datasets.ImageFolder(root="/kaggle/input/inaturalist/inaturalist_12K/train", transform=self.transform["train"])
        #Implementing a random 80-20 train-val split
        idx = list(range(len(full_data)))
        np.random.seed(42) 
        np.random.shuffle(idx)
        size=len(full_data)
        split = int(np.floor(0.8*size)) 
        train_idx = idx[:split]
        val_idx = idx[split:]

        self.train_dataset = Subset(full_data, train_idx)
        self.val_dataset = Subset(full_data, val_idx)
        self.test_dataset = datasets.ImageFolder(root = "/kaggle/input/inaturalist/inaturalist_12K/val", transform=self.transform["test"] )
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.config.batch_size, shuffle=True, num_workers = 3)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.config.batch_size, shuffle=False, num_workers = 3)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.config.batch_size, shuffle=True, num_workers = 3)

#Define Configuration for param sweep
sweep_config = {
    "method": "bayes",
    "metric": {"name":"val_acc", "goal":"maximize"},
    "parameters":{
        "filter_base":{"values":[32,64]},
        "filter_size":{"values":[3,5]},
        "filter_org":{"values":["same","double","halve"]},
        "conv_actv":{"values":["gelu","silu","mish"]},
        "pool_size":{"values":[2]},
        "dense_actv":{"values":["gelu"]},
        "dense_neurons":{"values":[512,1024]},
        "data_augmentation":{"values":[True]},
        "batch_norm":{"values":[False]},
        "dropout":{"values":[0.3,0.4]},
        "eta":{"values": [0.0001,0.001]},
        "batch_size":{"values":[64,128]}
    },
    "early_terminate":{
        "type":"hyperband",
        "min_iter": 3,
        "eta": 2
    },
    "command": [
        "python",
        "-W", "ignore",  # Disable warnings
        "-u",  # Unbuffered output
        "${program}"  # Required for Kaggle compatibility
    ]
}
from pytorch_lightning.callbacks import ModelCheckpoint
def train_sweep():
  import pickle
  wandb.init(project="DA6401_A2",settings=wandb.Settings(start_method="thread",_disable_stats=True))
  config = wandb.config
  data_manager = DataManager(config)
  model = CNN(config)
  chkpt_callback = ModelCheckpoint(
        monitor="val_acc",   # or other validation metric
        mode="max",          # "min" for loss
        save_top_k=1,
        filename="best_model"
    )
  with open("config.pkl", "wb") as f:
        pickle.dump(config, f)
  trainer = pl.Trainer(
      max_epochs = 10,
      logger = WandbLogger(),
      accelerator = "auto",
      callbacks = [chkpt_callback],
      enable_checkpointing = True,
      deterministic = True
  )
  torch.save(model.state_dict(), "model.pth")
  wandb.save("model.pth")
  trainer.fit(model, datamodule=data_manager)
  wandb.save(chkpt_callback.best_model_path)
  artifact = wandb.Artifact('best-model', type='model')
  artifact.add_file(chkpt_callback.best_model_path)
  wandb.log_artifact(artifact)

sweep_id = wandb.sweep(sweep_config, project = "DA6401_A2")
wandb.agent(sweep_id, function=train_sweep, count = 15)
