from typing import Optional, List, Union, Dict

from neuralforecast.models import (GRU,
                                   KAN,
                                   NBEATS,
                                   TiDE,
                                   NLinear,
                                   MLP,
                                   LSTM,
                                   DLinear,
                                   NHITS,
                                   Autoformer,
                                   Informer,
                                   DeepAR,
                                   PatchTST,
                                   TFT,
                                   DeepNPTS,
                                   DeepAR,
                                   TCN,
                                   DilatedRNN)


class ModelsConfig:
    MODEL_CLASSES = {
        # 'KAN': KAN,
        'MLP': MLP,
        'NHITS': NHITS,
        # 'TFT': TFT,
        'PatchTST': PatchTST,
        # 'GRU': GRU,
        # 'Autoformer': Autoformer,
        # 'Informer': Informer,

        # 'DLinear': DLinear,
        # 'DeepNPTS': DeepNPTS,
        # 'NBEATS': NBEATS,
        # 'TiDE': TiDE,
        # 'NLinear': NLinear,
        # 'DeepAR': DeepAR,
        # 'LSTM': LSTM,
        # 'DilatedRNN': DilatedRNN,
        # 'TCN': TCN,
    }

    model_names = [*MODEL_CLASSES]

    NEED_CPU = [
        # 'GRU',
        # 'DeepNPTS',
        # 'TFT',
        # 'PatchTST',
        # 'DeepAR',
        # 'LSTM',
        # 'TiDE',
        # 'NLinear',
        # 'KAN',
        # 'DilatedRNN',
        # 'TCN'
    ]

    @classmethod
    def create_model_instance(cls,
                              model_class: str,
                              model_config: Dict,
                              horizon: int,
                              input_size: int,
                              try_mps: bool = True,
                              limit_epochs: bool = False,
                              limit_val_batches: Optional[int] = None,
                              callbacks: Optional[List] = None,
                              alias: Optional[str] = None,):

        accelerator = 'mps' if try_mps else 'cpu'

        input_multiplier = model_config.pop('input_size_multiplier')

        base_config = {'accelerator': accelerator,
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
