#!/usr/bin/python
# J. Angevaare 21-10-2019 <j.angevaare@nikhef.nl>

# from DirectDmTargets import context
import subprocess
import os
import argparse
print("write_script.py::\tstart writing script and submit it to the queue")

#
# basic options
#
base_dir = "/project/xenon/jorana/software/DD_DM_targets/"
log_dir = "/data/xenon/joranang/log_files/dddm/"
prof_dir = base_dir + "/submit_to_stoomboot/profiles/"
default_conda = "/project/xenon/jorana/software/miniconda3/bin:$PATH"
default_envr = "dddm2"

#
# parse the arguments
#
parser = argparse.ArgumentParser(
    description="Stoomboot scripts submit manager")
parser.add_argument(
    '--target',
    type=str,
    help="the python script to be submitted to the batch queue")
parser.add_argument(
    '--arguments',
    type=str,
    default='',
    help="arguments for the python script (specified by --target)")
parser.add_argument(
    '--q',
    type=str,
    default='long',
    help="type of job, being either long or short")
parser.add_argument('--conda', type=str,
                    default=default_conda,
                    help="the anaconda path to be used by the script")
parser.add_argument('--base_dir', type=str,
                    default=base_dir,
                    help="Where to look for scripts")
parser.add_argument('--environment', type=str,
                    default=default_envr,
                    help="the anaconda environment to be sourced")
parser.add_argument('--walltime', type=str, default='false',
                    help="type of job, being either long or short")
parser.add_argument('--profiler', type=bool,
                    default=False, help='profile the python script.')
parser.add_argument(
    '--n_cores',
    type=int,
    default=1,
    help='number of cores for job. Runs multicore if n_cores>0')
parser.add_argument('--mem', type=int,
                    default=3, help='GB of ram per core')

args = parser.parse_args()

base_dir = args.base_dir
#
# where to write
#
_scriptfile = '%s_%s.sh' % ('dddm', args.arguments.replace("-", "_"))
_scriptfile = _scriptfile.replace(
    '_context_from_json /project/xenon/jorana/software/dddm_paper/dddm_context.json', '')
if 'multicore_hash' in args.arguments:
    print(f'write_script.py::\tChange scriptfile name')
    script_split = _scriptfile.strip('.sh').split('multicore_hash')
    _scriptfile = "".join([script_split[-1], script_split[0]]) + '.sh'
_scriptfile = _scriptfile.replace(" ", "").replace("__", "_")
if _scriptfile[-1] == "_":
    _scriptfile = _scriptfile[:-1]
log_file = 'log_%s_' % _scriptfile
prof_file = prof_dir + _scriptfile.replace('.sh', '.prof')
log_done = 'done_%s' % log_file
scriptfile = 'scripts/' + _scriptfile
#
# Check that the paths exist
#
if not os.path.exists(base_dir):
    raise FileNotFoundError("No such directory as:\n%s" % (base_dir))
# if not os.path.exists(base_dir+args.target):
#    raise FileNotFoundError("No such srcipt found at:\n%s"%(base_dir+args.target))
if not os.path.exists(log_dir):
    cmd = 'mkdir %s' % log_dir
    print("make %s since no folder was found there" % log_dir)
    subprocess.call(cmd, shell=False)
n_machines = 1
#
# Write the script
#
# print(f"write_script.py::\t start writing {scriptfile}")

fout = open(scriptfile, 'w')
fout.write('#!/bin/bash \n')
fout.write('# setup anaconda \n')
fout.write('export PATH=%s \n' % (args.conda))
fout.write('source activate %s\n' % (args.environment))
if os.path.exists(log_dir + log_done) or os.path.exists(log_dir + log_file):
    print(f'MAIN::remove {log_done} and/or {log_file}')
    if os.path.exists(log_dir + log_file):
        fout.write('rm -f ' + log_dir + log_file + ' \n')
    if os.path.exists(log_dir + log_done):
        fout.write('rm -f ' + log_dir + log_done + ' \n')

fout.write('exec 1> ' + log_dir + log_file + ' \n')
fout.write('exec 2>&1 \n')
if args.n_cores > 1:
    fout.write(
        'mpiexec -v -n %i python %s/%s %s\n' %
        (n_machines *
         args.n_cores,
         base_dir,
         args.target,
         args.arguments))
elif args.profiler:
    fout.write('python -m cProfile -o %s %s/%s %s\n' %
               (prof_file, base_dir, args.target, args.arguments))
else:
    fout.write('python %s/%s %s\n' % (base_dir, args.target, args.arguments))
fout.write('cd ' + log_dir + ' \n')
fout.write('mv ' + log_file + ' ' + log_done + ' \n')
fout.close()

# print("write_script.py::\t done writing")

#
# submit the script
#
err_logs = log_dir + 'error_' + log_file
basic_options = '-W group_list=xenon -e %s -o %s -j eo ' % (err_logs, err_logs)
if args.walltime != 'false':
    basic_options = basic_options + ' -l walltime=' + args.walltime
basic_options += f' -l nodes={n_machines}:ppn={args.n_cores} -l pvmem={args.mem}gb'
cmd = 'qsub %s %s %s' % (basic_options, "-q " + args.q, scriptfile)
subprocess.call(cmd, shell=False)
sub_cmd = f'scripts/stbc/sub_{_scriptfile.split("_")[0]}.sh'
print(f'Write command to {sub_cmd}')
fout = open(sub_cmd, 'w')
fout.write(cmd + '\n')
fout.close()
print("write_script.py::\t job written and submitted. Bye bye")
