from __future__ import absolute_import, unicode_literals
import json
import os
import shutil
import numpy as np
import dddm
from .pymultinest import MultiNestSampler, convert_dic_to_savable
import typing as ty

export, __all__ = dddm.exporter()


@export
class CombinedInference(MultiNestSampler):
    allow_multiple_detectors = True

    def __init__(
            self,
            wimp_mass: ty.Union[float, int],
            cross_section: ty.Union[float, int],
            spectrum_class: ty.List[ty.Union[dddm.DetectorSpectrum, dddm.GenSpectrum]],
            prior: dict,
            tmp_folder: str,
            results_dir: str=None,
            fit_parameters=('log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density', 'k'),

            detector_name=None,
            verbose=False,
            notes='default',

    ):
        assert detector_name is not None
        # Make list explicit
        spectrum_classes = spectrum_class
        del spectrum_class

        super().__init__(wimp_mass=wimp_mass,
                         cross_section=cross_section,
                         spectrum_class=spectrum_classes,
                         prior=prior,
                         tmp_folder=tmp_folder,
                         fit_parameters=fit_parameters,
                         detector_name=detector_name,
                         verbose=verbose,
                         results_dir=results_dir,
                         notes=notes,
                         )
        if len(spectrum_classes) < 2:
            self.log.warning(
                "Don't use this class for single experiments! Use NestedSamplerStatModel instead")
        self.sub_detectors = spectrum_classes
        self.config['sub_sets'] = spectrum_classes
        self.sub_classes = [
            MultiNestSampler(wimp_mass=wimp_mass,
                             cross_section=cross_section,
                             spectrum_class=one_class,
                             prior=prior,
                             tmp_folder=tmp_folder,
                             fit_parameters=fit_parameters,
                             detector_name=one_class.detector_name,
                             verbose=verbose,
                             notes=notes,
                             )
            for one_class in self.sub_detectors
        ]
        self.log.debug(f'Sub detectors are set: {self.sub_classes}')

    def _print_before_run(self):
        for c in self.sub_classes:
            self.log.debug(f'Printing det config for {c}')
            c._print_before_run()

    def _fix_parameters(self):
        """Fix the parameters of the sub classes"""
        super()._fix_parameters(_do_evaluate_benchmark=False)
        self.copy_config('mw prior sigma _wimp_mass _cross_section'.split())
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
            values = [s.config[k] for s in self.sub_classes]
            values.append(self.config[k])
            assert len(set(values)) == 1, f'Got inconsistent configs {values}'
        # for k in keys:
        #     if k not in self.config:
        #         raise ValueError(
        #             f'One or more of keys not in config: '
        #             f'{np.setdiff1d(keys, list(self.config.keys()))}')
        # copy_of_config = {k: self.config[k] for k in keys}
        # self.log.info(f'update config with {copy_of_config}')
        # for c in self.sub_classes:
        #     self.log.debug(f'{c} with config {c.config}')
        #     c.config.update(copy_of_config)

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
