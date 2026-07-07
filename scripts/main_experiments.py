import os
import warnings
from pathlib import Path

import pandas as pd
from neuralforecast import NeuralForecast
from modelradar.evaluate.radar import ModelRadar
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from utilsforecast.losses import mase
from functools import partial

from src.neural.nf_arch import ModelsConfig
from src.config import N_SAMPLES, SEED, TRY_MPS, MAX_SAMPLES, CB_N_STEPS
from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.neural.param_samples import ConfigSampler
from src.ww import WeightWatcherCallback

from src.utils import load_dataset_splits

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
target = 'monash_m1_monthly'

train, test, horizon, n_lags, freq, seas_len = load_dataset_splits(target)
mase_func = partial(mase, seasonality=seas_len)

results_dir = Path('../assets/results')


if __name__ == '__main__':
    print(results_dir.absolute())

    for model_nm in ModelsConfig.model_names:
        print(model_nm)
        # model = 'NHITS'

        config_pool = NEURAL_CONFIG_POOL[model_nm]
        config_list = ConfigSampler.generate_samples(config_pool=config_pool, num_samples=N_SAMPLES, random_state=SEED)

        for config_sample in config_list:
            print(config_sample)
            # config_sample=config_list[0]
            cfg_id = config_sample.pop('config_id')

            # check if no of configs reaches MAX_SAMPLES
            config_pattern = f"{model_nm},{target}"
            config_files = list(results_dir.glob(f"{config_pattern},*cbs.csv"))
            n_configs = len(config_files)
            if n_configs >= MAX_SAMPLES:
                print(f"No of configs reached MAX_SAMPLES for {model_nm},{target}")
                break

            cbs_fp = results_dir / f'{model_nm},{target},{cfg_id},cbs.csv'
            cbd_fp = results_dir / f'{model_nm},{target},{cfg_id},cbd.csv'

            if cbs_fp.exists():
                print(f"Skipping {model_nm},{target},{cfg_id},cbs.csv -- Already exists")
                continue

            print(f"Running config {n_configs} / {MAX_SAMPLES}")

            ##---- model instance

            ww_callback = WeightWatcherCallback(every_n_steps=CB_N_STEPS)

            model = ModelsConfig.create_model_instance(model_class=model_nm,
                                                       model_config=config_sample,
                                                       horizon=horizon,
                                                       input_size=n_lags,
                                                       try_mps=TRY_MPS,
                                                       callbacks=[ww_callback])

            sf = StatsForecast(models=[SeasonalNaive(season_length=seas_len)], freq=freq, )

            ##----- fitting and fcsting

            nf = NeuralForecast(models=[model], freq=freq, )

            sf.fit(train)
            nf.fit(df=train)

            fcst_sf = sf.predict(h=horizon)
            fcst = nf.predict()
            # fixing ds due to issues, mostly in QE/QS freq
            fcst['ds'] = test['ds']
            fcst_sf['ds'] = test['ds']

            ##----- eval

            holdout_set = test.merge(fcst, how='left', on=['unique_id', 'ds'])
            holdout_set = holdout_set.merge(fcst_sf, how='left', on=['unique_id', 'ds'])

            radar = ModelRadar(
                cv_df=holdout_set,
                metrics=[mase_func],
                model_names=[model_nm, 'SeasonalNaive'],
                train_df=train,
                hardness_reference='SeasonalNaive',
                ratios_reference='SeasonalNaive',
            )

            err = radar.evaluate(keep_uids=False)

            ##----- prep callback results

            cb = WeightWatcherCallback.get_cb(nf)

            cbs_df = pd.DataFrame(cb.summaries)
            cbs_df['model'] = model_nm
            cbs_df['config_id'] = cfg_id
            cbs_df['dataset'] = target
            cbs_df['mase'] = err[model_nm]
            cbs_df['mase_sn'] = err['SeasonalNaive']

            cbd_df = pd.concat(cb.details).reset_index(drop=True)
            cbd_df['model'] = model_nm
            cbd_df['config_id'] = cfg_id
            cbd_df['dataset'] = target
            cbd_df['mase'] = err[model_nm]
            cbd_df['mase_sn'] = err['SeasonalNaive']

            ##----- serialization
            cbs_df.to_csv(cbs_fp, index=False)
            cbd_df.to_csv(cbd_fp, index=False)
