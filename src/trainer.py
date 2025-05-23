import wandb
from datetime import datetime
from typing import Dict, List, Union, Optional
from statistics import mean
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch import nn

from nlg_metrics import Metrics
from utils.custom_types import ModelType, OptimType, DeviceTye, DataIterType
from utils.custom_types import SchedulerType
from utils.train_utils import seed_everything


class TrackMetrics:

    def __init__(self) -> None:

        self.reset_running()
        self.metrics = self.init_metrics()

    def create_default_dict(self):

        metrics_dict = {
            "train": defaultdict(list, {}),
            "val": defaultdict(list, {})
        }

        return metrics_dict

    def reset_running(self):
        self.running = self.create_default_dict()

    def init_metrics(self):
        return self.create_default_dict()

    def update_running(self, metrics: Dict[str, float], phase: str) -> None:
        for name, value in metrics.items():
            self.running[phase][name].append(value)

    def update(self, phase: str):
        for name, values in self.running[phase].items():
            self.metrics[phase][name].append(mean(values))
        self.reset_running()


class Trainer():

    def __init__(self,
                 optims: List[OptimType],
                 schedulers: List[SchedulerType],
                 device: DeviceTye,
                 epochs: int,
                 val_interval: int,
                 early_stop: int,
                 lr_patience: int,
                 embedings_finetune: int,
                 grad_clip: float,
                 lambda_c: float,
                 checkpoints_path: str,
                 pad_id: int,
                 resume: Optional[str] = None,
                 use_wandb: bool = True,
                 enable_finetune: bool = False,
                 project_name: str = "image-captioning") -> None:

        # Some parameters
        self.train = True  # train or val
        self.device = device
        self.resume = resume
        self.epochs_num = epochs - 1  # epoch count start from 0
        self.epoch = 0
        self.val_interval = val_interval  # validate the model evey (n) epochs
        # stop trianing if the model doesn't improve for n-validation epochs
        self.stop = early_stop
        # number of validation epochs in which model doesn't improve
        self.bad_epochs_num = 0
        # number of validation epochs to wait before decreases the lr if model
        # does not improve
        self.lr_patience = lr_patience
        # start tune embeddings after n training epochs have beed passed
        self.finetune_embedding = embedings_finetune
        self.enable_finetune = enable_finetune
        self.pad_id = pad_id
        self.use_wandb = use_wandb
        self.batch_step = 0  # For batch-level metrics
        self.epoch_step = 0  # For epoch-level metrics

        # criterion, optims and schedulers
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)
        self.img_embed_optim = optims[0]
        self.transformer_optim = optims[1]
        self.image_scheduler = schedulers[0]
        self.transformer_scheduler = schedulers[1]

        # metrics
        # TODO:
        # - Make as configurable parameter. Setting the Metrics class and
        # metrics tracker with it.
        # - Move tracker to Metrics class.
        # metrics functions and tracker
        self.nlgmetrics = Metrics()
        self.metrics_tracker = TrackMetrics()
        self.best_metric = 0

        # Some coeffecient
        # coeffecient of Doubly stochastic attention regularization
        self.lc = lambda_c
        self.grad_clip_c = grad_clip  # gradient clip coeffecient

        if resume is None:
            time_tag = str(datetime.now().strftime("%d%m.%H%M"))
        else:
            time_tag = Path(resume).parent
        
        # Initialize wandb if enabled
        if self.use_wandb:
            # Configure WandB - you'll need to run wandb login first
            wandb.init(
                project=project_name,
                name=f"run_{time_tag}",
                config={
                    "epochs": epochs,
                    "val_interval": val_interval,
                    "early_stop": early_stop,
                    "lr_patience": lr_patience,
                    "embedings_finetune": embedings_finetune,
                    "enable_finetune": enable_finetune,
                    "grad_clip": grad_clip,
                    "lambda_c": lambda_c,
                    "device": str(device),
                    "resume": resume is not None
                }
            )
            
            # Define metrics with different step sequences
            wandb.define_metric("batch/*", step_metric="batch_step")
            wandb.define_metric("train/*", step_metric="epoch_step")
            wandb.define_metric("val/*", step_metric="epoch_step")
            wandb.define_metric("learning_rate/*", step_metric="epoch_step")

            wandb.define_metric("loss/*", step_metric="epoch")

        # make folder for the experment
        checkpoints_path = Path(checkpoints_path) / f"{time_tag}"  # type: Path
        checkpoints_path.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path = str(checkpoints_path)

    def loss_fn(self, logits: Tensor, targets: Tensor,
                attns: Tensor) -> Tensor:
        v_sz = logits.size()[-1]
        targets = targets.contiguous()
        loss = self.criterion(logits.view(-1, v_sz), targets.view(-1))

        # Doubly stochastic attention regularization:
        # "Show, Attend and Tell" - arXiv:1502.03044v3 eq(14)
        # change atten size to be
        # [layer_num, head_num, batch_size, max_len, encode_size^2]
        attns = attns.permute(0, 2, 1, 3, 4)
        ln, hn = attns.size()[:2]  # number of layers, number of heads

        # calc λ(1-∑αi)^2 for each pixel in each head in each layer
        # alphas [layer_num, head_num, batch_size*encode_size^2]
        # TODO:
        # Reduction: Would it make any difference if I sum across
        # (encode_size^2, and head) dimensions and average across batch and
        # layers?
        alphas = self.lc * (1. - attns.sum(dim=3).view(ln, hn, -1))**2
        alphas: Tensor
        dsar = alphas.mean(-1).sum()

        return loss + dsar

    def clip_gradient(self):
        for optim in [self.img_embed_optim, self.transformer_optim]:
            for group in optim.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-self.grad_clip_c,
                                               self.grad_clip_c)

    def remove_pad(
            self, tensor: Tensor, lens: Tensor,
            mask: Tensor) -> Union[List[List[List[str]]], List[List[str]]]:
        # format 3D tensor (References) to 2D tensor
        lens = lens.view(-1)
        max_len = tensor.size(1)
        is3d = len(tensor.size()) == 3
        if is3d:
            tensor = tensor.permute(0, 2, 1).contiguous().view(-1, max_len)
            mask = mask.permute(0, 2, 1).contiguous().view(-1, max_len)

        # Remove pads: select elements that are not equal to pad (into 1d
        # tensor) then split the formed 1d tensor according to lengthes
        tensor = torch.masked_select(tensor, mask=mask)
        tensor = torch.split(tensor, split_size_or_sections=lens.tolist())
        tensor = [[str(e.item()) for e in t] for t in tensor]

        # Get back into 3d (list of list of list)
        if is3d:
            tensor = [tensor[i:i + 5] for i in range(0, len(tensor), 5)]

        return tensor

    def get_metrics(self, gtruth: Tensor, lens: Tensor, preds: Tensor):
        # gtruth [B, lm - 1, cn=5]
        # lens [B, cn=5]
        # preds [B, lm]
        mask = gtruth != self.pad_id  # mask pad tokens
        refs = self.remove_pad(gtruth, lens, mask)
        hypos = self.remove_pad(preds, lens[:, 0], mask[:, :, 0])

        scores = self.nlgmetrics.calculate(refs, hypos, self.train)

        return scores

    def set_phase(self) -> None:
        if not self.train:
            self.train = True  # toggle if val
        else:
            # validate every "val_interval" epoch
            self.train = bool(self.epoch % self.val_interval)

    def check_improvement(self, metric: float):
        is_better = metric > self.best_metric
        reduce_lr = False
        es = False
        if is_better:
            self.best_metric = metric
            self.bad_epochs_num = 0
        else:
            self.bad_epochs_num += 1

        if self.bad_epochs_num > self.lr_patience:
            reduce_lr = True
            self.num_bad_epochs = 0

        if self.bad_epochs_num > self.stop:
            es = True  # early stop

        return is_better, reduce_lr, es

    def load_checkpoint(self):
        load_path = str(Path(self.checkpoints_path) / self.resume)

        # load checkopoint
        state = torch.load(load_path, map_location=torch.device("cpu"), weights_only=True)
        
        # Load optimizer and scheduler states
        image_optim_state = state["optims"][0]
        transformer_optim_state = state["optims"][1]
        image_scheduler_state = state["schedulers"][0]
        transformer_scheduler_state = state["schedulers"][1]

        # load state dicts
        self.img_embed_optim.load_state_dict(image_optim_state)
        self.transformer_optim.load_state_dict(transformer_optim_state)
        self.image_scheduler.load_state_dict(image_scheduler_state)
        self.transformer_scheduler.load_state_dict(transformer_scheduler_state)

        # set some parameters
        self.train = state["phase"]
        self.epoch = state["epoch"]
        self.bad_epochs_num = state["bad_epochs_num"]
        self.best_metric = state["best_metric"]
        self.metrics_tracker.running = state["running_metrics"]
        self.metrics_tracker.metrics = state["metrics"]

        # Load batch_step and epoch_step if available
        if "batch_step" in state:
            self.batch_step = state["batch_step"]
        else:
            self.batch_step = 0  # Fallback for older checkpoints
            
        if "epoch_step" in state:
            self.epoch_step = state["epoch_step"]
        else:
            self.epoch_step = self.epoch  # Fallback for older checkpoints

        self.set_phase()  # set train or vall phase
        self.epoch += 1 * self.train

        # Check if this is a combined model checkpoint (has only one model state)
        if len(state["models"]) == 1:
            return state["models"][0]
        else:
            image_model_state = state["models"][0]
            transformer_state = state["models"][1]
            return image_model_state, transformer_state


    def save_checkpoint(self, models: List[ModelType], is_best: bool):
        if len(models) == 1:
            # For combined model case
            model = models[0]
            model_state = model.state_dict()
            image_optim_state = self.img_embed_optim.state_dict()
            transformer_optim_state = self.transformer_optim.state_dict()
            image_scheduler_state = self.image_scheduler.state_dict()
            transformer_scheduler_state = self.transformer_scheduler.state_dict()
            
            state = {
                "models": [model_state],
                "model_type": "combined",
                "optims": [image_optim_state, transformer_optim_state],
                "schedulers": [image_scheduler_state, transformer_scheduler_state],
                "phase": self.train,
                "epoch": self.epoch,
                "bad_epochs_num": self.bad_epochs_num,
                "best_metric": self.best_metric,
                "running_metrics": self.metrics_tracker.running,
                "metrics": self.metrics_tracker.metrics,
                "batch_step": self.batch_step,
                "epoch_step": self.epoch_step
            }
        else:
            # Original case with separate encoder and decoder models
            image_model_state = models[0].state_dict()
            transformer_state = models[1].state_dict()
            image_optim_state = self.img_embed_optim.state_dict()
            transformer_optim_state = self.transformer_optim.state_dict()
            image_scheduler_state = self.image_scheduler.state_dict()
            transformer_scheduler_state = self.transformer_scheduler.state_dict()
            
            state = {
                "models": [image_model_state, transformer_state],
                "model_type": "standard",
                "optims": [image_optim_state, transformer_optim_state],
                "schedulers": [image_scheduler_state, transformer_scheduler_state],
                "phase": self.train,
                "epoch": self.epoch,
                "bad_epochs_num": self.bad_epochs_num,
                "best_metric": self.best_metric,
                "running_metrics": self.metrics_tracker.running,
                "metrics": self.metrics_tracker.metrics,
                "batch_step": self.batch_step,
                "epoch_step": self.epoch_step
            }

        # set save path
        file_name = "checkpoint"
        if is_best:
            file_name = f"{file_name}_best"
        save_path = Path(self.checkpoints_path) / f"{file_name}.pth.tar"

        torch.save(state, save_path)
        
        # Log checkpoint info to wandb if enabled
        if self.use_wandb and is_best:
            wandb.save(str(save_path))

    def record_data(self, phase, metrics_dict):
        # WandB logging if enabled
        if self.use_wandb:
            log_dict = {}
            for k, v in metrics_dict.items():
                if v:  # Check if list is not empty
                    log_dict[f"{phase}/{k}"] = v[-1]
                    if k == "loss":
                        log_dict[f"loss/{phase}"] = v[-1]
                        # Add epoch as step metric for loss
                        log_dict["epoch"] = self.epoch
            
            # Add learning rates
            if phase == "train":
                log_dict["learning_rate/encoder"] = self.image_scheduler.get_last_lr()[0]
                log_dict["learning_rate/transformer"] = self.transformer_scheduler.get_last_lr()[0]
                
            # Add epoch step metric for proper sequencing
            log_dict["epoch_step"] = self.epoch_step
            
            # Log with the dedicated step metric
            wandb.log(log_dict)
            
            # Increment epoch_step after logging epoch-level metrics
            self.epoch_step += 1

    def run(self, img_embeder: ModelType, transformer: ModelType,
            data_iters: DataIterType, SEED: int):
        # Sizes:
        # B:   batch_size
        # is:  image encode size^2: image seq len: [default=196]
        # vsc: vocab_size: vsz
        # lm:  max_len: [default=52]
        # cn:  number of captions: [default=5]
        # hn:  number of transformer heads: [default=8]
        # ln:  number of layers
        # k:   Beam Size

        # some preparations:
        phases = ["val", "train"]  # to determine the current phase
        seed_everything(SEED)
        if self.resume:
            model_state_dicts = self.load_checkpoint()
            img_embeder.load_state_dict(model_state_dicts[0])
            transformer.load_state_dict(model_state_dicts[1])

        # move models to device
        img_embeder = img_embeder.to(self.device)
        transformer = transformer.to(self.device)

        # start
        main_pb = tqdm(range(self.epochs_num))
        while self.epoch <= self.epochs_num:

            main_pb.set_description(f"epoch: {self.epoch:02d}")

            is_best = False
            es = False  # early stopping
            lr_r = False  # reduce lr flag

            if self.train:
                img_embeder.train()
                transformer.train()
                data_iter = data_iters[0]
                # Fine tune the embeddings layer after some epochs and add the
                # parameters to the optimizer
                if self.epoch == self.finetune_embedding and self.enable_finetune:
                    # Find embedding parameters that aren't already in the optimizer
                    embedding_params = list(transformer.decoder.cptn_emb.parameters())
                    existing_params = set()
                    
                    # Get existing parameters from optimizer
                    for group in self.transformer_optim.param_groups:
                        existing_params.update(group['params'])
                    
                    # Enable gradients for embedding parameters
                    for p in embedding_params:
                        p.requires_grad = True
                    
                    # Only add parameters that aren't already in the optimizer
                    new_params = [p for p in embedding_params if p not in existing_params]
                    if new_params:
                        self.transformer_optim.add_param_group({"params": new_params})
            else:
                img_embeder.eval()
                transformer.eval()
                data_iter = data_iters[1]

            # Iterate over data
            # prgress bar
            pb = tqdm(data_iter, leave=False, total=len(data_iter))
            pb.unit = "step"
            for step, (imgs, cptns_all, lens) in enumerate(pb):
                imgs: Tensor  # images [B, 3, 256, 256]
                cptns_all: Tensor  # all 5 captions [B, lm, cn=5]
                lens: Tensor  # lengthes of all captions [B, cn]

                # set progress bar description and metrics
                pb.set_description(f"{phases[self.train]}: Step-{step+1:<4d}")

                # move data to device, and random selected cptns
                imgs = imgs.to(self.device)
                # random selected cptns: [B, lm]
                idx = np.random.randint(0, cptns_all.size(-1))
                cptns = cptns_all[:, :, idx].to(self.device)

                # zero the parameter gradients
                self.img_embed_optim.zero_grad()
                self.transformer_optim.zero_grad()

                with torch.set_grad_enabled(self.train):
                    # embed images using CNN then get logits prediction using
                    # the transformer
                    imgs = img_embeder(imgs)
                    logits, attns = transformer(imgs, cptns[:, :-1])
                    logits: Tensor  # [B, lm - 1, vsz]
                    attns: Tensor  # [ln, B, hn, lm, is]

                    # loss calc, backward
                    loss = self.loss_fn(logits, cptns[:, 1:], attns)

                    # in train, gradient clip + update weights
                    if self.train:
                        loss.backward()
                        self.clip_gradient()
                        self.img_embed_optim.step()
                        self.transformer_optim.step()

                # get predections then alculate some metrics
                preds = torch.argmax(logits, dim=2).cpu()  # predections
                targets = cptns_all[:, 1:]  # remove < SOS >
                scores = self.get_metrics(targets, lens - 1, preds)
                scores["loss"] = loss.item()  # add loss to metrics scores
                self.metrics_tracker.update_running(scores, phases[self.train])

                # Log batch metrics to wandb if in training phase
                if self.use_wandb and self.train and (step % 50 == 0):  # Log every 50 steps
                    batch_log = {f"batch/{phases[self.train]}/{k}": v for k, v in scores.items()}
                    batch_log["batch_step"] = self.batch_step
                    wandb.log(batch_log)
                    self.batch_step += 1

            self.metrics_tracker.update(phases[self.train])  # save metrics
            
            if not self.train:
                checked_metric = self.metrics_tracker.metrics["val"]["bleu4"]
                is_best, lr_r, es = self.check_improvement(checked_metric[-1])

                if lr_r:  # reduce lr
                    self.image_scheduler.step()
                    self.transformer_scheduler.step()

            # save checkpoint
            if self.train or is_best:
                self.save_checkpoint(models=[img_embeder, transformer],
                                     is_best=is_best)

            # Record metrics
            phase = phases[self.train]
            metrics_dict = self.metrics_tracker.metrics[phase]
            self.record_data(phase, metrics_dict)

            # epoch ended
            self.set_phase()  # set train or vall phase
            self.epoch += 1 * self.train
            pb.close()  # close progress bar
            if self.train:
                main_pb.update(1)
            if es:  # early stopping
                main_pb.close()
                print(f"Early stop training at epoch {self.epoch}")
                if self.use_wandb:
                    wandb.log({"early_stopped": True, "stopped_epoch": self.epoch})
                break
                
        # Finish wandb run
        if self.use_wandb:
            wandb.finish()


    def run_combined(self, model: ModelType, data_iters: DataIterType, SEED: int):
        """Training method for combined model with hierarchical encoder"""
        # Sizes:
        # B:   batch_size
        # is:  image encode size^2: image seq len: [default=196]
        # vsc: vocab_size: vsz
        # lm:  max_len: [default=52]
        # cn:  number of captions: [default=5]
        # hn:  number of transformer heads: [default=8]
        # ln:  number of layers
        # k:   Beam Size

        # some preparations:
        phases = ["val", "train"]  # to determine the current phase
        seed_everything(SEED)
        
        if self.resume:
            model_state_dict = self.load_checkpoint()
            model.load_state_dict(model_state_dict)

        # move model to device
        model = model.to(self.device)

        # start
        main_pb = tqdm(range(self.epochs_num))
        while self.epoch <= self.epochs_num:
            main_pb.set_description(f"epoch: {self.epoch:02d}")

            is_best = False
            es = False  # early stopping
            lr_r = False  # reduce lr flag

            if self.train:
                model.train()
                data_iter = data_iters[0]
                # fine tune the embeddings layer after some epochs
                if self.epoch == self.finetune_embedding and self.enable_finetune:
                    # Find embedding parameters that aren't already in the optimizer
                    embedding_params = set(model.decoder.cptn_emb.parameters())
                    existing_params = set()
                    
                    # Get existing parameters from optimizer
                    for group in self.transformer_optim.param_groups:
                        existing_params.update(group['params'])
                    
                    # Only add parameters that aren't already in the optimizer
                    new_params = [p for p in embedding_params if p not in existing_params]
                    if new_params:
                        self.transformer_optim.add_param_group({"params": new_params})
                    
                    # Also start fine-tuning parts of the backbone
                    if hasattr(model.encoder, 'backbone'):
                        # For EfficientNet/ResNet, fine-tune the last few layers
                        backbone_layers = list(model.encoder.backbone.parameters())
                        # Fine-tune last 20% of layers
                        start_idx = int(len(backbone_layers) * 0.8)
                        
                        # Get existing parameters from encoder optimizer
                        encoder_existing_params = set()
                        for group in self.img_embed_optim.param_groups:
                            encoder_existing_params.update(group['params'])
                        
                        # Only add parameters that aren't already in the optimizer
                        new_backbone_params = [p for p in backbone_layers[start_idx:] 
                                            if p not in encoder_existing_params]
                        
                        if new_backbone_params:
                            self.img_embed_optim.add_param_group({"params": new_backbone_params})
            else:
                model.eval()
                data_iter = data_iters[1]

            # Iterate over data
            pb = tqdm(data_iter, leave=False, total=len(data_iter))
            pb.unit = "step"
            for step, (imgs, cptns_all, lens) in enumerate(pb):
                imgs: Tensor  # images [B, 3, 256, 256]
                cptns_all: Tensor  # all 5 captions [B, lm, cn=5]
                lens: Tensor  # lengthes of all captions [B, cn]

                # set progress bar description and metrics
                pb.set_description(f"{phases[self.train]}: Step-{step+1:<4d}")

                # move data to device, and random selected cptns
                imgs = imgs.to(self.device)
                # random selected cptns: [B, lm]
                idx = np.random.randint(0, cptns_all.size(-1))
                cptns = cptns_all[:, :, idx].to(self.device)

                # zero the parameter gradients
                self.img_embed_optim.zero_grad()
                self.transformer_optim.zero_grad()

                with torch.set_grad_enabled(self.train):
                    # Forward pass through combined model
                    logits, attns = model(imgs, cptns[:, :-1])
                    logits: Tensor  # [B, lm - 1, vsz]
                    attns: Tensor  # [ln, B, hn, lm, is]

                    # loss calc, backward
                    loss = self.loss_fn(logits, cptns[:, 1:], attns)

                    # in train, gradient clip + update weights
                    if self.train:
                        loss.backward()
                        self.clip_gradient()
                        self.img_embed_optim.step()
                        self.transformer_optim.step()

                # get predections then calculate some metrics
                preds = torch.argmax(logits, dim=2).cpu()  # predictions
                targets = cptns_all[:, 1:]  # remove < SOS >
                scores = self.get_metrics(targets, lens - 1, preds)
                scores["loss"] = loss.item()  # add loss to metrics scores
                self.metrics_tracker.update_running(scores, phases[self.train])

                # Log batch metrics to wandb if in training phase
                if self.use_wandb and self.train and (step % 50 == 0):  # Log every 50 steps
                    batch_log = {f"batch/{phases[self.train]}/{k}": v for k, v in scores.items()}
                    batch_log["batch_step"] = self.batch_step
                    wandb.log(batch_log)
                    self.batch_step += 1

            self.metrics_tracker.update(phases[self.train])  # save metrics
            
            if not self.train:
                checked_metric = self.metrics_tracker.metrics["val"]["bleu4"]
                is_best, lr_r, es = self.check_improvement(checked_metric[-1])

                if lr_r:  # reduce lr
                    self.image_scheduler.step()
                    self.transformer_scheduler.step()

            # save checkpoint
            if self.train or is_best:
                self.save_checkpoint(models=[model], is_best=is_best)

            # Record metrics
            phase = phases[self.train]
            metrics_dict = self.metrics_tracker.metrics[phase]
            self.record_data(phase, metrics_dict)

            # epoch ended
            self.set_phase()  # set train or vall phase
            self.epoch += 1 * self.train
            pb.close()  # close progress bar
            if self.train:
                main_pb.update(1)
            if es:  # early stopping
                main_pb.close()
                print(f"Early stop training at epoch {self.epoch}")
                if self.use_wandb:
                    wandb.log({"early_stopped": True, "stopped_epoch": self.epoch})
                break
                
        # Finish wandb run
        if self.use_wandb:
            wandb.finish()