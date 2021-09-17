import setuptools


def open_requirements(path):
    with open(path) as f:
        requires = [
            r.split('/')[-1] if r.startswith('git+') else r
            for r in f.read().splitlines()]
    return requires


readme = open('README.md').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
requirements = open_requirements('requirements.txt')

setuptools.setup(
    name='DirectDmTargets',
    version='2.0.1',
    description='Probing the complementarity of several targets used in '
                'Direct Detection Experiments for Dark Matter',
    long_description=readme + '\n\n' + history,
    author='Joran Angevaare',
    url='https://github.com/jorana/DD_DM_targets',
    packages=setuptools.find_packages() + ['extra_requirements'],
    package_dir={'DirectDmTargets': 'DirectDmTargets',
                 'extra_requirements': 'extra_requirements'},
    package_data={'DirectDmTargets': ['data/*'],
                  'extra_requirements': ['requirements-tests.txt'],
                  },
    setup_requires=['pytest-runner'],
    install_requires=requirements,
    python_requires=">=3.8",
    tests_require=requirements + ['pytest',
                                  'hypothesis-numpy'],
    scripts=['scripts/run_combined_multinest',
             'scripts/run_dddm_emcee',
             ],
    keywords='todo',
    classifiers=['Intended Audience :: Science/Research',
                 'Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 3.8',
                 'Natural Language :: English',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: Implementation :: CPython',
                 'Topic :: Scientific/Engineering :: Physics',
                 ],
    zip_safe=False)
