from __future__ import absolute_import, unicode_literals
import json
import logging
import os
import shutil
import numpy as np
import dddm
from .pymultinest import MultiNestSampler, convert_dic_to_savable

export, __all__ = dddm.exporter()

log = logging.getLogger()


@export
class CombinedInference(MultiNestSampler):
    allow_multiple_detectors = True

    def __init__(self, targets, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not np.all([t in dddm.experiment_registry for t in targets]):
            raise NotImplementedError(
                f'Insert tuple of sub-experiments. {targets} are incorrect format')
        if len(targets) < 2:
            self.log.warning(
                "Don't use this class for single experiments! Use NestedSamplerStatModel instead")
        self.log.debug(f'Register {targets}')
        self.sub_detectors = targets
        self.config['sub_sets'] = targets
        self.sub_classes = [
            MultiNestSampler(det)
            for det in self.sub_detectors
        ]
        self.log.debug(f'Sub detectors are set: {self.sub_classes}')

    def _print_before_run(self):
        for c in self.sub_classes:
            self.log.debug(f'Printing det config for {c}')
            c._print_before_run()

    def _fix_parameters(self):
        """Fix the parameters of the sub classes"""
        super()._fix_parameters(_do_evaluate_benchmark=False)
        self.copy_config('mw prior sigma halo_model spectrum_class'.split())
        for c in self.sub_classes:
            self.log.debug(f'Fixing parameters for {c}')
            c._fix_parameters()

    def _log_probability_nested(self, theta):
        return np.sum([c._log_probability_nested(theta)
                       for c in self.sub_classes])

    def copy_config(self, keys):
        for k in keys:
            if k not in self.config:
                raise ValueError(
                    f'One or more of keys not in config: '
                    f'{np.setdiff1d(keys, list(self.config.keys()))}')
        copy_of_config = {k: self.config[k] for k in keys}
        self.log.info(f'update config with {copy_of_config}')
        for c in self.sub_classes:
            self.log.debug(f'{c} with config {c.config}')
            c.config.update(copy_of_config)

    def save_sub_configs(self, force_index=False):
        save_dir = self.get_save_dir(force_index=force_index)
        self.log.info(
            f'CombinedInference::\tSave configs of sub_experiments to {save_dir}')
        # save the config
        save_dir = os.path.join(save_dir, 'sub_exp_configs')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for c in self.sub_classes:
            save_as = os.path.join(f'{save_dir}', f'{c.config["detector"]}_')
            with open(save_as + 'config.json', 'w') as file:
                json.dump(convert_dic_to_savable(c.config), file, indent=4)
            np.save(save_as + 'config.npy', convert_dic_to_savable(c.config))
            shutil.copy(c.config['logging'], save_as +
                        c.config['logging'].split('/')[-1])
            self.log.info('save_sub_configs::\tdone_saving')
