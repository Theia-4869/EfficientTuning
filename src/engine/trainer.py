#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
from torch.nn import LayerNorm
import os

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..models.vit_subset.vit_layernorm import FrontModule, BackModule, SideModule
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage

logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # solver related
        logger.info("Setting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, outputs

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels
    
    def get_gradient(self, train_loader):
        """
        Get the gradient in only 1 epoch
        """
        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        
        # Enable training mode
        self.model.train()
        self.optimizer.zero_grad()
        
        logger.info("Calculating gradient...")
        end = time.time()
        
        for idx, input_data in enumerate(train_loader):
            inputs, targets = self.get_input(input_data)
            
            # move data to device
            inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
            targets = targets.to(self.device, non_blocking=True)  # (batchsize, )
            
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)  # (batchsize, num_cls)
                loss = self.cls_criterion(outputs, targets, self.cls_weights)

                if loss == float('inf'):
                    logger.info(
                        "encountered infinite loss, skip gradient updating for this batch!"
                    )
                    continue
                elif torch.isnan(loss).any():
                    logger.info(
                        "encountered nan loss, skip gradient updating for this batch!"
                    )
                    continue

            loss.backward()
        
        logger.info("Gradient calculation time: {:.2f}".format(time.time() - end))
        
    def set_module(self, model, name, module):
        tokens = name.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)
            
    def select_subset_by_layer(self):
        """
        Select the subset per layer in model to update
        """
        logger.info("Selecting gradient...")
        grad_mean_abs = []
        for name, param in self.model.named_parameters():
            if "enc.transformer.encoder.layer" in name:
                if not self.cfg.MODEL.SUBSET.LN_GRAD and "norm" in name:
                    continue
                grad_mean_abs.append(torch.mean(torch.abs(param.grad)))
        
        thresh = torch.quantile(torch.tensor(grad_mean_abs), 1 - self.cfg.MODEL.SUBSET.PERCENTILE)
        requires_grad = set()
        for name, param in self.model.named_parameters():
            if "enc.transformer.encoder" in name:
                if not self.cfg.MODEL.SUBSET.LN_GRAD and "norm" in name:
                    param.requires_grad = False
                elif "encoder_norm" in name:
                    param.requires_grad = False
                elif torch.mean(torch.abs(param.grad)) > thresh:
                    requires_grad.add(".".join(name.split(".")[:-1]))            
                else:
                    param.requires_grad = False
            elif "enc.transformer.embeddings" in name:
                param.requires_grad = False
                
        if self.cfg.MODEL.SUBSET.WEIGHT_AND_BIAS:
            for name, param in self.model.named_parameters():
                for requires_name in requires_grad:
                    if requires_name in name:
                        param.requires_grad = True
        
        # for name, module in self.model.named_modules():
        #     if name in requires_grad:
        #         if self.cfg.MODEL.SUBSET.MODE == "front":
        #             self.set_module(self.model, name, FrontModule(module))
        #         elif self.cfg.MODEL.SUBSET.MODE == "back":
        #             self.set_module(self.model, name, BackModule(module))
        #         elif self.cfg.MODEL.SUBSET.MODE == "side":
        #             self.set_module(self.model, name, SideModule(module))
        # logger.info(f"Subset mode: {self.cfg.MODEL.SUBSET.MODE}")
        
        # self.model = self.model.cuda(device=self.device)
                
        logger.info("Show subset...")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(name)
    
    def select_subset_by_block(self):
        """
        Select the subset per block in model to update
        """
        logger.info("Selecting gradient...")
        grad_per_block = {}
        for name, param in self.model.named_parameters():
            if "enc.transformer.encoder.layer" in name:
                if not self.cfg.MODEL.SUBSET.LN_GRAD and "norm" in name:
                    continue
                idx = int(name.split(".")[4])
                if idx in grad_per_block:
                    grad_per_block[idx].append(torch.mean(torch.abs(param.grad)))
                else:
                    grad_per_block[idx] = [torch.mean(torch.abs(param.grad))]
                    
        grad_thresh = []
        for idx in grad_per_block:
            grad_thresh.append(torch.quantile(torch.tensor(grad_per_block[idx]), 1 - self.cfg.MODEL.SUBSET.PERCENTILE))
        
        requires_grad = []
        for name, param in self.model.named_parameters():
            if "enc.transformer.encoder" in name:
                if not self.cfg.MODEL.SUBSET.LN_GRAD and "norm" in name:
                    param.requires_grad = False
                elif "encoder_norm" in name:
                    param.requires_grad = False
                else:
                    idx = int(name.split(".")[4])
                    if torch.mean(torch.abs(param.grad)) > grad_thresh[idx]:
                        requires_grad.append(".".join(name.split(".")[:-1]))             
                    else:
                        param.requires_grad = False
            elif "enc.transformer.embeddings" in name:
                param.requires_grad = False
        
        if self.cfg.MODEL.SUBSET.WEIGHT_AND_BIAS:
            for name, param in self.model.named_parameters():
                for requires_name in requires_grad:
                    if requires_name in name:
                        param.requires_grad = True
                        
        # for name, module in self.model.named_modules():
        #     if name in requires_grad:
        #         if self.cfg.MODEL.SUBSET.MODE == "front":
        #             self.set_module(self.model, name, FrontModule(module))
        #         elif self.cfg.MODEL.SUBSET.MODE == "back":
        #             self.set_module(self.model, name, BackModule(module))
        #         elif self.cfg.MODEL.SUBSET.MODE == "side":
        #             self.set_module(self.model, name, SideModule(module))
                    
        # self.model = self.model.cuda(device=self.device)
        
        logger.info("Show subset...")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(name)
    
    def select_subset_by_prompt(self):
        """
        Select the subset in prompt model to update
        """
        logger.info("Selecting gradient...")
        grad_mean_abs = []
        for name, param in self.model.named_parameters():
            if "enc.transformer.encoder.layer" in name:
                if not self.cfg.MODEL.SUBSET.LN_GRAD and "norm" in name:
                    continue
                grad_mean_abs.append(torch.mean(torch.abs(param.grad)))
        
        thresh = torch.quantile(torch.tensor(grad_mean_abs), 1 - self.cfg.MODEL.SUBSET.PERCENTILE)
        requires_grad = set()
        for name, param in self.model.named_parameters():
            if "enc.transformer.encoder" in name:
                if not self.cfg.MODEL.SUBSET.LN_GRAD and "norm" in name:
                    param.requires_grad = False
                elif "encoder_norm" in name:
                    param.requires_grad = False
                elif torch.mean(torch.abs(param.grad)) > thresh:
                    requires_grad.add(".".join(name.split(".")[:-1]))            
                else:
                    param.requires_grad = False
            elif "enc.transformer.embeddings" in name:
                param.requires_grad = False
                
        if self.cfg.MODEL.SUBSET.WEIGHT_AND_BIAS:
            for name, param in self.model.named_parameters():
                for requires_name in requires_grad:
                    if requires_name in name:
                        param.requires_grad = True
        
        # for name, module in self.model.named_modules():
        #     if name in requires_grad:
        #         if self.cfg.MODEL.SUBSET.MODE == "front":
        #             self.set_module(self.model, name, FrontModule(module))
        #         elif self.cfg.MODEL.SUBSET.MODE == "back":
        #             self.set_module(self.model, name, BackModule(module))
        #         elif self.cfg.MODEL.SUBSET.MODE == "side":
        #             self.set_module(self.model, name, SideModule(module))
        # logger.info(f"Subset mode: {self.cfg.MODEL.SUBSET.MODE}")
        
        # self.model = self.model.cuda(device=self.device)
                
        logger.info("Show subset...")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        best_test = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break
                
                X, targets = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                train_loss, _ = self.forward_one_batch(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
             # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            if test_loader is not None:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            if test_loader is not None:
                t_name_ = "test_" + test_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
                if test_loader is not None:
                    curr_acc_ = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name_]["top1"]
                    if curr_acc_ > best_test:
                        best_test = curr_acc_
                        best_epoch_ = epoch + 1
                        logger.info(
                            f'Best test epoch {best_epoch_}: best test metric: {best_test:.3f}')
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

        # save the last checkpoints
        # if self.cfg.MODEL.SAVE_CKPT:
        #     Checkpointer(
        #         self.model,
        #         save_dir=self.cfg.OUTPUT_DIR,
        #         save_to_disk=True
        #     ).save("last_model")

    @torch.no_grad()
    def save_prompt(self, epoch):
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")
