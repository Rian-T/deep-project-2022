from model.basic import MLPModel, CNNModel, ResNetModel, CnnLSTMModel
from data.MarioKart import MKFrameActionDataModule, MNISTDataModule, MKFrameSequenceActionDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping


models = {"MLP": MLPModel, "CNN": CNNModel, "RESNET": ResNetModel, "LSTM": CnnLSTMModel}

model_choosed = list(models.items())[2]

mk_ds = MKFrameActionDataModule(batch_size=256)
#mk_ds = MKFrameSequenceActionDataModule("datasets/MarioKartFrameSequence16", batch_size=64)
model = model_choosed[1]()
#model = CNNModel.load_from_checkpoint(f"checkpoints/CNN/deep-project-2022/2a5qk2g1/checkpoints/epoch=10-step=1166.ckpt")
#model = ResNetModel.load_from_checkpoint(f"checkpoints/RESNET/deep-project-2022/1ivbxl2j/checkpoints/epoch=6-step=2954.ckpt")

wandb_logger = WandbLogger(project="deep-project-2022")

trainer = Trainer(
    gpus=1,
    default_root_dir=f"checkpoints/{model_choosed[0]}",
    callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)],
    logger=wandb_logger,
)

trainer.fit(model, mk_ds)
results = trainer.test(model, mk_ds)
print(results)
