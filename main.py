from lightning.pytorch.cli import LightningCLI, instantiate_class
from lightning import LightningDataModule, LightningModule
from torch import nn
from torch.utils.data import DataLoader, Dataset
from functools import partial

class BaseModel(LightningModule):
    def __init__(self, network: nn.Module, solver: dict=None, weights=None):
        super().__init__()
        self.network = network
        self.solver = solver
        self.predictions = []
        
    def training_step(self, batch):
        losses = self.network(batch)
        for metric, value in losses.items():
            self.log(metric, value, sync_dist=False)
        # losses['loss'].backward()
        # for n, p in self.network.named_parameters():
        #     if p.requires_grad and p.grad is None:
        #         print(n)
        # assert False
        return losses
    
    def validation_step(self, batch, idx):
        predictions = self.network(batch)
        self.predictions.append(self.trainer.val_dataloaders.dataset.collect(predictions, batch))
        
    def on_validation_epoch_end(self,):
        predictions = self.trainer.val_dataloaders.dataset.accumulate(self.predictions)
        self.predictions.clear()
        if self.device.index == 0:
            metrics = self.trainer.val_dataloaders.dataset.evaluate(predictions)
            for metric, value in metrics.items():
                self.log(metric, value, sync_dist=False)
            return metrics

    def configure_optimizers(self):
        from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
        from detectron2.solver import get_default_optimizer_params
        if self.solver['lr_factors'] == 'vitdet':
            lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7) 
        else:
            def get_lr_factor(name, module_factor_dict=None):
                for key, value in module_factor_dict.items():
                    if name.startswith(key):
                        return value
                return 1
            lr_factor_func = partial(get_lr_factor, module_factor_dict=self.solver['lr_factors'])
        optimizer = instantiate_class(
            get_default_optimizer_params(
                self.network, 
                base_lr=self.solver['optimizer']['init_args']['lr'], 
                weight_decay_norm=0, 
                overrides=self.solver['overrides'] if 'overrides' in self.solver else None,
                lr_factor_func=lr_factor_func,
            ), self.solver['optimizer']
        )
        lr_scheduler = instantiate_class(optimizer, self.solver['lr_scheduler'])
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

class BaseData(LightningDataModule):
    def __init__(self, batch_size_per_gpu, num_workers, 
        dataset_train: Dataset=None, dataset_val: Dataset=None):
        super().__init__()
        self.batch_size_per_gpu = batch_size_per_gpu
        self.num_workers = num_workers  
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
    
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size_per_gpu, 
            collate_fn=self.dataset_train.collate_fn, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size_per_gpu, 
            collate_fn=self.dataset_val.collate_fn, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True, drop_last=True)

if __name__ == '__main__':
    LightningCLI(BaseModel, BaseData, parser_kwargs={"parser_mode": "omegaconf"})
