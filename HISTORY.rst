.. :changelog:

3.0.0 / 2022-01-30
------------------
major:
 - Refactor dddm (#158)
patch:
 - pipy install (#175)
 - Write documentation(#169)


2.1.1 / 2022-01-30
------------------
patch:
 - try upping coverage (#131)
 - Sourcery refactored master branch (#141)
 - fix line endings (#142)
 - Increase testing stability (#145)


2.1.0 / 2021-11-23
------------------
minor:
 - Fix name change of package (#109)
 - Add seaborn copies for extracting confidence regions (#119, #130)

patch: 
 - Readme updates (000057cb1e90bd77a5a733eb134ac36641173ef9, 0776ec9d6f35c87c5ae755d8c080a5c6675bb95f, e366fabba589eb7779a65adc73bd657ec55ef102, a9623b8092c0d800e21066c0c3207fb6927fde7e)
 - add fixed priors (#129)


2.0.1 / 2021-09-17
------------------
patch:
 - First apply smearing, then the threshold (#92)
 - Fix kwargs setting for scatter plots (#107)

2.0.0 / 2021-08-25
------------------
major:
 - Fix galactic and det spectrum (#87, #90)

minor:
 - Don't use save-intermediate or emax for run_combined_multinest (#51)
 -  Use 1T low-er resolution (#52)
 - Fix Ge-iZIP background rate (#53)
 - Make 5 keV consistently emax (#56)
 - Fix #54 - Update XENONnT (#84)
 - Sdd result plotting (#83)

patch:
 - Make requirements file pinned (#57)
 - Add a logger with nice formatting (#85)
 - Save canvas to pickle (#50)
 - Restore autopep8 (#88)
 - remove old notebooks (#91, 794adfb )


1.0.0 / 2021-06-22
------------------
major:
 - Restructure code, get ready for release (#16)
 - Restructure dddm and improve CI (#15)
 - Debugging DirectDmTargets (#12)

minor:
 - Update to run locally (#37)
 - Small tweaks to context and verne interfacing (#30)
 - Detector configurations and config passing (#13)

patch:
 - Delete run_dddm_multinest (#46)
 - Use dependabot for actions (#34, #35)
 - Pending issues work in development (#32)
 - use workflows for testing (#14)
 - Fix the tests (#19)
 - Flag files for computation before continuing (#10)

0.4.0 / 2020-04-23
------------------
- Working fully for three optimizers:
    - multinest
    - emcee
    - nestle

0.1.0 / 2019-11-14
------------------
- Initial release
