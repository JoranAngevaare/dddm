from .pymultinest import MultiNestSampler
import ultranest
import datetime
import os
import dddm
import warnings

export, __all__ = dddm.exporter()


@export
class UltraNestSampler(MultiNestSampler):
    """
    Skeleton of ultranest sampler
    """
    def run(self):
        warnings.warn(f'Ultranest sampler is not completely implemented yet')
        self._fix_parameters()
        self._print_before_run()

        n_dims = len(self.config["fit_parameters"])
        tol = self.config['tol']  # the stopping criterion
        save_at = self.get_save_dir()

        self.log.warning(f'start_fit for {n_dims} parameters')

        start = datetime.datetime.now()

        # Multinest saves output to a folder. First write to the tmp folder,
        # move it to the results folder later
        _tmp_folder = self.get_save_dir()
        save_at_temp = os.path.join(_tmp_folder, 'ultra_nest')

        sampler = ultranest.ReactiveNestedSampler(
            param_names=list(self.config["fit_parameters"]),
            loglike=self._log_probability_nested,
            transform=self._log_prior_transform_nested,  # SafePrior,
            log_dir=_tmp_folder,
            resume='resume',
        )
        result = sampler.run(
            min_num_live_points=self.config['nlive'],
            min_ess=self.config['nlive'],
            dlogz=tol,
        )
        sampler.print_results()
        self.result_file = save_at_temp

        # Open a save-folder after successful running multinest. Move the
        # multinest results there.
        dddm.utils.check_folder_for_file(save_at)
        end = datetime.datetime.now()
        dt = (end - start).total_seconds()
        self.log.info(f'fit_done in {dt} s ({dt / 3600} h)')
        self.log_dict['did_run'] = True
        # release the config
        self.config = dddm.utils._immutable_to_dict(self.config)
        self.config['fit_time'] = dt

        self.log.info('Finished with running Multinest!')
        return result, sampler

    def get_summary(self):
        raise NotImplementedError