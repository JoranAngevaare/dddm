from __future__ import absolute_import, unicode_literals
import json
import os
import shutil
import numpy as np
import dddm
from .pymultinest import MultiNestSampler, convert_dic_to_savable
from .nestle import NestleSampler
import typing as ty

export, __all__ = dddm.exporter()


class _CombinedInference:
    allow_multiple_detectors = True

    def set_models(self):
        for c in self.sub_classes:
            self.log.debug(f'Printing det config for {c}')
            c.set_models()

    def _print_before_run(self):
        for c in self.sub_classes:
            self.log.debug(f'Printing det config for {c}')
            c._print_before_run()

    def _fix_parameters(self):
        """Fix the parameters of the sub classes"""
        for c in self.sub_classes:
            self.log.debug(f'Fixing parameters for {c}')
            c._fix_parameters()
        super()._fix_parameters(_do_evaluate_benchmark=False)

    def _log_probability_nested(self, theta):
        return np.sum([c._log_probability_nested(theta)
                       for c in self.sub_classes])

    def save_sub_configs(self, force_index=False):
        save_dir = self.get_save_dir(force_index=force_index)
        self.log.info(
            f'CombinedInference::\tSave configs of sub_experiments to {save_dir}')
        # save the config
        save_dir = os.path.join(save_dir, 'sub_exp_configs')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for c in self.sub_classes:
            if 'logging' not in c.config:
                raise ValueError(f'{c} does not have logging in config ({list(c.config.keys())})')
            save_as = os.path.join(f'{save_dir}', f'{c.config["detector"]}_')
            with open(save_as + 'config.json', 'w') as file:
                json.dump(convert_dic_to_savable(c.config), file, indent=4)
            np.save(save_as + 'config.npy', convert_dic_to_savable(c.config))
            shutil.copy(c.config['logging'], save_as +
                        c.config['logging'].split('/')[-1])
            self.log.info('save_sub_configs::\tdone_saving')


@export
class CombinedMultinest(_CombinedInference, MultiNestSampler):
    def __init__(
            self,
            wimp_mass: ty.Union[float, int],
            cross_section: ty.Union[float, int],
            spectrum_class: ty.List[ty.Union[dddm.DetectorSpectrum, dddm.GenSpectrum]],
            prior: dict,
            tmp_folder: str,
            results_dir: str = None,
            fit_parameters=('log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density', 'k'),

            detector_name=None,
            verbose=False,
            notes='default',
            nlive=1024,
            tol=0.1,
    ):
        assert detector_name is not None
        # Make list explicit
        spectrum_classes = spectrum_class
        del spectrum_class

        MultiNestSampler.__init__(self,
                                  wimp_mass=wimp_mass,
                                  cross_section=cross_section,
                                  spectrum_class=spectrum_classes,
                                  prior=prior,
                                  tmp_folder=tmp_folder,
                                  fit_parameters=fit_parameters,
                                  detector_name=detector_name,
                                  verbose=verbose,
                                  results_dir=results_dir,
                                  notes=notes,
                                  nlive=nlive,
                                  tol=tol,
                                  )
        if len(spectrum_classes) < 2:
            self.log.warning(
                "Don't use this class for single experiments! Use NestedSamplerStatModel instead")
        self.sub_detectors = spectrum_classes
        self.config['sub_sets'] = [str(sp) for sp in spectrum_classes]
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


@export
class CombinedNestle(_CombinedInference, NestleSampler):
    def __init__(
            self,
            wimp_mass: ty.Union[float, int],
            cross_section: ty.Union[float, int],
            spectrum_class: ty.List[ty.Union[dddm.DetectorSpectrum, dddm.GenSpectrum]],
            prior: dict,
            tmp_folder: str,
            results_dir: str = None,
            fit_parameters=('log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density', 'k'),

            detector_name=None,
            verbose=False,
            notes='default',
            nlive=1024,
            tol=0.1,
    ):
        assert detector_name is not None
        # Make list explicit
        spectrum_classes = spectrum_class
        del spectrum_class

        NestleSampler.__init__(self,
                               wimp_mass=wimp_mass,
                               cross_section=cross_section,
                               spectrum_class=spectrum_classes,
                               prior=prior,
                               tmp_folder=tmp_folder,
                               fit_parameters=fit_parameters,
                               detector_name=detector_name,
                               verbose=verbose,
                               results_dir=results_dir,
                               notes=notes,
                               nlive=nlive,
                               tol=tol,
                               )
        if len(spectrum_classes) < 2:
            self.log.warning(
                "Don't use this class for single experiments! Use NestedSamplerStatModel instead")
        self.sub_detectors = spectrum_classes
        self.config['sub_sets'] = [str(sp) for sp in spectrum_classes]
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
