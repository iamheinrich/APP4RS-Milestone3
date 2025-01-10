import lightning as L

#added imports
import torch

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
        x, y = batch
        #TODO check whether flattening of x necessary
        x_hat  = self.model(x) # TODO timm models seemingly only return logits not probs, verify
        batch_loss = self.criterion(x_hat, y)

        #turn logits to probabilities for logging
        if self.args.task == "slc":
            probabilities = torch.softmax(x_hat, dim=1)
        elif self.args.task == "mlc":
            probabilities = torch.sigmoid(x_hat)
        else:
            raise Exception("args.task not handled!")

        output = {"labels": y, "probabilities": probabilities, "loss": batch_loss}
        self.training_step_outputs.append(output)
        return batch_loss #TODO lightning needs loss as training step return to apply

    def validation_step(self, batch, batch_idx): #TODO check training_step todos
        x, y = batch
        x_hat  = self.model(x)
        batch_loss = self.criterion(x_hat, y)

        #turn logits to probabilities for logging
        if self.args.task == "slc":
            probabilities = torch.softmax(x_hat, dim=1)
        elif self.args.task == "mlc":
            probabilities = torch.sigmoid(x_hat)
        else:
            raise Exception("args.task not handled!")

        output = {"labels": y, "probabilities": probabilities, "loss": batch_loss}
        self.validation_step_outputs.append(output)
        return output

    def test_step(self, batch, batch_idx): #TODO check training_step todos
        x, y = batch
        x_hat  = self.model(x)
        batch_loss = self.criterion(x_hat, y)

        #turn logits to probabilities for logging
        if self.args.task == "slc":
            probabilities = torch.softmax(x_hat, dim=1)
        elif self.args.task == "mlc":
            probabilities = torch.sigmoid(x_hat)
        else:
            raise Exception("args.task not handled!")

        output = {"labels": y, "probabilities": probabilities, "loss": batch_loss}
        self.test_step_outputs.append(output)
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
        if self.args.task == "slc":
            criterion = torch.nn.CrossEntropyLoss()
        elif self.args.task == "mlc":
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise Exception("args.task not handled!")
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
