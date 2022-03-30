import setuptools


def open_requirements(path):
    with open(path) as f:
        requires = [
            r.split('/')[-1] if r.startswith('git+') else r
            for r in f.read().splitlines()]
    return requires


readme = open('README.md').read()
history = open('HISTORY.md').read()
requirements = open_requirements('requirements.txt')

setuptools.setup(
    name='dddm',
    version='3.0.2',
    description='Direct Detection of Dark Matter: '
                'Probing the complementarity of several targets for dark matter detection',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    author='Joran R. Angevaare',
    url='https://github.com/jorana/DD_DM_targets',
    packages=setuptools.find_packages() + ['extra_requirements'],
    package_dir={'dddm': 'dddm',
                 'extra_requirements': 'extra_requirements'},
    package_data={'dddm': ['data/*'],
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
                 'Programming Language :: Python :: 3.10',
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: Implementation :: CPython',
                 'Topic :: Scientific/Engineering :: Physics',
                 ],
    zip_safe=False)
