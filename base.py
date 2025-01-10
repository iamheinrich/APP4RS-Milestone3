import lightning as L


class BaseModel(L.LightningModule):
    def __init__(self, args, datamodule, network):
        super().__init__()
        self.args = args

        self.model = network
        self.save_hyperparameters('args')
        
        self.datamodule = datamodule
        self.criterion = self.init_criterion()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        output = None
        return output

    def validation_step(self, batch, batch_idx):
        output = None
        return output

    def test_step(self, batch, batch_idx):
        output = None
        return output

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    ########################
    # CRITERION & OPTIMIZER
    ########################

    def configure_optimizers(self):
        optimizer = None
        lr_scheduler = None
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def init_criterion(self):
        criterion = None
        return criterion

    #################
    # LOGGING MODULE
    #################

    # implement functionality for logging here

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
