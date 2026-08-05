from typing import Optional, List, Dict

from neuralforecast.models import (MLP, NHITS, PatchTST)


class ModelsConfig:
    MODEL_CLASSES = {
        'MLP': MLP,
        'NHITS': NHITS,
        'PatchTST': PatchTST,
    }

    model_names = [*MODEL_CLASSES]

    @classmethod
    def create_model_instance(cls,
                              model_class: str,
                              model_config: Dict,
                              horizon: int,
                              input_size: int,
                              engine: bool = 'cpu',
                              limit_epochs: bool = False,
                              limit_val_batches: Optional[int] = None,
                              callbacks: Optional[List] = None,
                              alias: Optional[str] = None, ):

        input_multiplier = model_config.pop('input_size_multiplier')

        base_config = {'accelerator': engine,
                       'h': horizon,
                       'input_size': input_size * input_multiplier, }

        if 'inference_input_size_multiplier' in model_config:
            inference_input_size_multiplier = model_config.pop('inference_input_size_multiplier')
            base_config['inference_input_size'] = input_size * inference_input_size_multiplier

        config = {**model_config, **base_config}

        if limit_epochs:
            config['max_steps'] = 2

        if limit_val_batches is not None:
            config['limit_val_batches'] = limit_val_batches

        if callbacks is not None:
            config['callbacks'] = callbacks

        if alias is not None:
            config['alias'] = alias

        model_instance = cls.MODEL_CLASSES[model_class](**config)

        return model_instance
