from typing import TYPE_CHECKING
import logging

from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer

logger = logging.getLogger(__name__)


@Callback.register("alternative_metric")
class AlternativeMetric(Callback):
    """
    Update metric `gen_accuracy` to be the maximum between `gen_accuracy` and `_gen_accuracy_alt` (see paper for CLOSURE
    ambiguity issue)
    """
    @handle_event(Events.EPOCH_END, priority=-1)
    def epoch_end_logging(self, trainer: 'CallbackTrainer'):
        trainer.val_metrics['gen_accuracy_1'] = trainer.val_metrics['gen_accuracy']
        trainer.val_metrics['gen_accuracy_2'] = trainer.val_metrics['_gen_accuracy_alt']
        trainer.val_metrics['gen_accuracy'] = max(
            trainer.val_metrics['gen_accuracy'],
            trainer.val_metrics['_gen_accuracy_alt'],
        )
        if '_closure_embed_mat_spa_alt' in trainer.val_metrics:
            trainer.val_metrics['_closure_embed_mat_spa_1'] = trainer.val_metrics['_closure_embed_mat_spa']
            trainer.val_metrics['_closure_embed_mat_spa_2'] = trainer.val_metrics['_closure_embed_mat_spa_alt']
            trainer.val_metrics['_closure_embed_mat_spa'] = max(
                trainer.val_metrics['_closure_embed_mat_spa_1'],
                trainer.val_metrics['_closure_embed_mat_spa_2'],
            )
