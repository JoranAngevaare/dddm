"""
Submodules of the SuperCDMS migdal classes using the darkelf package
"""
from . import super_cdms


class DarkElfSuperCdmsHvGeMigdal(super_cdms.SuperCdmsHvGeMigdal):
    detector_name = f'{super_cdms.SuperCdmsHvGeMigdal.detector_name}_darkelf'
    interaction_type = 'migdal_SI_darkelf_grid'

    def __init__(self, e_min_kev=0, e_max_kev=0.1, **kwargs):
        super().__init__(e_min_kev=e_min_kev, e_max_kev=e_max_kev, **kwargs)


class DarkElfIbeSuperCdmsHvGeMigdal(super_cdms.SuperCdmsHvGeMigdal):
    detector_name = f'{super_cdms.SuperCdmsHvGeMigdal.detector_name}_darkelf_ibe'
    interaction_type = 'migdal_SI_darkelf_ibe'

    def __init__(self, e_min_kev=0, e_max_kev=0.1, **kwargs):
        super().__init__(e_min_kev=e_min_kev, e_max_kev=e_max_kev, **kwargs)


class DarkElfSuperCdmsHvSiMigdal(super_cdms.SuperCdmsHvSiMigdal):
    detector_name = f'{super_cdms.SuperCdmsHvSiMigdal.detector_name}_darkelf'
    interaction_type = 'migdal_SI_darkelf_grid'

    def __init__(self, e_min_kev=0, e_max_kev=0.1, **kwargs):
        super().__init__(e_min_kev=e_min_kev, e_max_kev=e_max_kev, **kwargs)


class DarkElfIbeSuperCdmsHvSiMigdal(super_cdms.SuperCdmsHvSiMigdal):
    detector_name = f'{super_cdms.SuperCdmsHvSiMigdal.detector_name}_darkelf_ibe'
    interaction_type = 'migdal_SI_darkelf_ibe'

    def __init__(self, e_min_kev=0, e_max_kev=0.1, **kwargs):
        super().__init__(e_min_kev=e_min_kev, e_max_kev=e_max_kev, **kwargs)


class DarkElfSuperCdmsIzipGeMigdal(super_cdms.SuperCdmsIzipGeMigdal):
    detector_name = f'{super_cdms.SuperCdmsIzipGeMigdal.detector_name}_darkelf'
    interaction_type = 'migdal_SI_darkelf_grid'

    def __init__(self, e_min_kev=0, e_max_kev=0.1, **kwargs):
        super().__init__(e_min_kev=e_min_kev, e_max_kev=e_max_kev, **kwargs)


class DarkElfIbeSuperCdmsIzipGeMigdal(super_cdms.SuperCdmsIzipGeMigdal):
    detector_name = f'{super_cdms.SuperCdmsIzipGeMigdal.detector_name}_darkelf_ibe'
    interaction_type = 'migdal_SI_darkelf_ibe'

    def __init__(self, e_min_kev=0, e_max_kev=0.1, **kwargs):
        super().__init__(e_min_kev=e_min_kev, e_max_kev=e_max_kev, **kwargs)


class DarkElfSuperCdmsIzipSiMigdal(super_cdms.SuperCdmsIzipSiMigdal):
    detector_name = f'{super_cdms.SuperCdmsIzipSiMigdal.detector_name}_darkelf'
    interaction_type = 'migdal_SI_darkelf_grid'

    def __init__(self, e_min_kev=0, e_max_kev=0.1, **kwargs):
        super().__init__(e_min_kev=e_min_kev, e_max_kev=e_max_kev, **kwargs)


class DarkElfIbeSuperCdmsIzipSiMigdal(super_cdms.SuperCdmsIzipSiMigdal):
    interaction_type = 'migdal_SI_darkelf_ibe'
    detector_name = f'{super_cdms.SuperCdmsIzipSiMigdal.detector_name}_darkelf_ibe'

    def __init__(self, e_min_kev=0, e_max_kev=0.1, **kwargs):
        super().__init__(e_min_kev=e_min_kev, e_max_kev=e_max_kev, **kwargs)
