import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K, is_tf_keras
from bert4keras.snippets import is_string, string_matching
from bert4keras.snippets import is_one_of, insert_arguments
from bert4keras.backend import piecewise_linear
from bert4keras.backend import root_mean_square as rms
import re
from keras.optimizers import Adam
def export_to_custom_objects(base_extend_with):
    """装饰器，用来将优化器放到custom_objects中
    """
    def new_extend_with(BaseOptimizer, name=None):
        NewOptimizer = base_extend_with(BaseOptimizer)

        if is_string(name):
            NewOptimizer.__name__ = name

        name = NewOptimizer.__name__
        keras.utils.get_custom_objects()[name] = NewOptimizer

        return NewOptimizer

    return new_extend_with


def transformer_schedule(t, start_step, warmup_steps, d_model):
    t = K.cast(t, K.floatx())
    d_model = K.cast(d_model, K.floatx())
    warmup_steps = K.cast(warmup_steps, K.floatx())
    start_step = K.cast(start_step, K.floatx())
    step = t + start_step
    arg1 = 1.0 / K.sqrt(step)
    arg2 = step * K.pow(warmup_steps, -1.5)
    lr = 1.0 / K.sqrt(d_model) * K.minimum(arg1, arg2)
    return lr


@export_to_custom_objects
def extend_with_transformer_schedule(BaseOptimizer):
    """返回新的优化器类，加入分段线性学习率
    """
    class NewOptimizer(BaseOptimizer):
        """带有分段线性学习率的优化器
        其中schedule是形如{1000: 1, 2000: 0.1}的字典，
        表示0～1000步内学习率线性地从零增加到100%，然后
        1000～2000步内线性地降到10%，2000步以后保持10%
        """
        @insert_arguments(start_step=0., warmup_steps=4000., d_model=512.)
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        @K.symbolic
        def get_updates(self, loss, params):
            lr_multiplier = transformer_schedule(self.iterations,
                                                      self.start_step,
                                                      self.warmup_steps,
                                                      self.d_model)

            old_update = K.update

            def new_update(x, new_x):
                if is_one_of(x, params):
                    new_x = x + (new_x - x) * lr_multiplier
                return old_update(x, new_x)

            K.update = new_update
            updates = super(NewOptimizer, self).get_updates(loss, params)
            K.update = old_update

            return updates

        def get_config(self):
            config = {
                'lr_schedule': self.lr_schedule,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_transformer_schedule_v2(BaseOptimizer):
    """返回新的优化器类，加入分段线性学习率
    """
    class NewOptimizer(BaseOptimizer):
        """带有分段线性学习率的优化器
        其中schedule是形如{1000: 1, 2000: 0.1}的字典，
        表示0～1000步内学习率线性地从零增加到100%，然后
        1000～2000步内线性地降到10%，2000步以后保持10%
        """
        @insert_arguments(start_step=0., warmup_steps=4000., d_model=512.)
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def _decayed_lr(self, var_dtype):
            lr_multiplier = transformer_schedule(self.iterations,
                                                      self.start_step,
                                                      self.warmup_steps,
                                                      self.d_model)
            lr_t = super(NewOptimizer, self)._decayed_lr(var_dtype)
            return lr_t * K.cast(lr_multiplier, var_dtype)

        def get_config(self):
            config = {
                'lr_schedule': self.lr_schedule,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


if is_tf_keras:
    extend_with_transformer_schedule = extend_with_transformer_schedule_v2
