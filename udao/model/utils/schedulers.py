from typing import Any, Callable, Dict, Optional, Type, Union

import lightning.pytorch as pl
import pytorch_warmup as warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from ..module import UdaoModule

SchedulerType = Union[LRScheduler, Callable[[pl.Trainer, UdaoModule], LRScheduler]]


class UdaoLRScheduler(pl.Callback):
    """Manual LR scheduler for pytorch-lightning.
    This is a callback that can be used to manually control the learning rate.
    It can be used to implement warmup or other learning rate schedules,
    in particular when needing access to data properties such
    as the number of training batches.

    Parameters
    ----------
    scheduler : SchedulerType
        Either a torch.optim.lr_scheduler.LRScheduler or
        a callable that instantiates a scheduler based on the trainer and module.
        This second option is useful when the scheduler needs access to data properties.
    warmup_cls : Type[warmup.BaseWarmup]
        A warmup class from pytorch-warmup. Will be instantiated with the optimizer.
        If None, no warmup will be used.
    scheduler_params : Optional[Dict[str, Any]], optional
        Parameters to pass at scheduler instantiation, by default None
    warmup_params : Optional[Dict[str, Any]], optional
        Parameters to pass at warmup instantiation, by default None
    """

    def __init__(
        self,
        scheduler: SchedulerType,
        warmup_cls: Type[warmup.BaseWarmup],
        scheduler_params: Optional[Dict[str, Any]] = None,
        warmup_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.scheduler_params = scheduler_params if scheduler_params else {}
        self.warmup_params = warmup_params if warmup_params else {}
        self.warmup_cls = warmup_cls
        self.scheduler = None
        self.scheduler_generator = None
        self.warmup = None
        if isinstance(scheduler, LRScheduler):
            self.scheduler = scheduler
        elif callable(scheduler):
            self.scheduler_generator = scheduler

    def on_train_start(self, trainer: pl.Trainer, module: UdaoModule) -> None:  # type: ignore
        if self.scheduler_generator:
            self.scheduler = self.scheduler_generator(
                trainer, module, **self.scheduler_params
            )
        if self.warmup_cls:
            self.warmup = self.warmup_cls(trainer.optimizers[0], **self.warmup_params)
        # Assuming the optimizer has already been configured in pl_module

    def on_train_batch_end(self, *args: Any) -> None:
        if self.scheduler is None:
            raise ValueError("Scheduler not initialized")
        if self.warmup is not None:
            with self.warmup.dampening():
                self.scheduler.step()
        else:
            self.scheduler.step()

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint["scheduler_state_dict"] = (
            self.scheduler.state_dict() if self.scheduler else None
        )

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


def setup_cosine_annealing_lr(
    trainer: pl.Trainer, pl_module: UdaoModule
) -> CosineAnnealingLR:
    """Instantiation of a CosineAnnealingLR scheduler
    from data size and number of epochs."""
    num_training_batches = len(trainer.train_dataloader)  # type: ignore
    max_steps = 0
    if trainer.max_steps:
        max_steps = trainer.max_steps
    elif trainer.max_epochs:
        max_steps = num_training_batches * trainer.max_epochs
    else:
        raise ValueError("Either max_steps or max_epochs must be set")
    optimizer = trainer.optimizers[0]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(max_steps),
        eta_min=pl_module.learning_params.min_lr,
    )
    return scheduler
