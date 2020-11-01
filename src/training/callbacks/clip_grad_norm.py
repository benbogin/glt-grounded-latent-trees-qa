import logging
from typing import TYPE_CHECKING

import torch
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer  # pylint:disable=unused-import

logger = logging.getLogger(__name__)


@Callback.register("clip_grad_norm")
class ClipGradNorm(Callback):
    """
    Applies gradient norm and/or clipping.

    # Parameters

    grad_norm : float, optional (default = None)
        If provided, we rescale the gradients before the optimization step.
    grad_clipping : float, optional (default = None)
        If provided, we use this to clip gradients in our model.
    """

    def __init__(
        self, max_grad_norm: float
    ) -> None:
        self.max_grad_norm = max_grad_norm

    @handle_event(Events.BACKWARD, priority=1000)
    def rescale_gradients(self, trainer: "CallbackTrainer"):
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), self.max_grad_norm)
