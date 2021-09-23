#-----------------------------------------------------------------------------------------
# Options:
#   -h  --help                    help message
#
# [ Compilation flags ]
#   -verbose                      enable verbose compilation mode
#   --build=<DIR>                 specify building directory
#   --bin=<DIR>                   specify directory for executables
#   -debug                        compile in `debug` mode
#   --compiler=<COMPILER>         compiler used (can be a valid path to the binary)
#
# [ Kokkos-specific flags ]
#   -kokkos                       compile with `Kokkos` support
#   --kokkos_devices=<DEV>        `Kokkos` devices
#   --kokkos_arch=<ARCH>          `Kokkos` architecture
#   --kokkos_options=<OPT>        `Kokkos` options
#   --kokkos_vector_length=<VLEN> `Kokkos` vector length
#   --kokkos_loop=[...]           `Kokkos` loop layout
#   --nvcc_wrapper_cxx=<COMPILER> `NVCC_WRAPPER_DEFAULT_COMPILER` flag for `Kokkos`
#   --kokkos_cuda_options=<COPT>  `Kokkos` Cuda options
# ----------------------------------------------------------------------------------------

import argparse
import glob
import re
import subprocess
import os
from pathlib import Path

# Global Settings
# ---------------
# Default values:
DEF_build_dir = 'build'
DEF_bin_dir = 'bin'
DEF_compiler = 'g++'
DEF_cppstandard = 'c++17'

# Set template and output filenames
makefile_input = 'Makefile.in'
makefile_output = 'Makefile'

# Options:
Kokkos_devices = dict(host=['Serial', 'OpenMP', 'PThreads'], device=['Cuda'])
Kokkos_arch = dict(host=["AMDAVX", "EPYC", "ARMV80", "ARMV81", "ARMV8_THUNDERX", "ARMV8_THUNDERX2", "WSM", "SNB", "HSW", "BDW", "SKX", "KNC", "KNL", "BGQ", "POWER7", "POWER8", "POWER9"], device=["KEPLER30", "KEPLER32", "KEPLER35", "KEPLER37", "MAXWELL50", "MAXWELL52", "MAXWELL53", "PASCAL60", "PASCAL61", "VOLTA70", "VOLTA72", "TURING75", "AMPERE80", "VEGA900", "VEGA906", "INTEL_GE"])
Kokkos_devices_options = Kokkos_devices["host"] + Kokkos_devices["device"]
Kokkos_arch_options = Kokkos_arch["device"] + Kokkos_arch["device"]
Kokkos_loop_options = ['default', '1DRange', 'MDRange', 'TP-TVR', 'TP-TTR', 'TP-TTR-TVR', 'for']

# . . . auxiliary functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . -->
def defineOptions():
  parser = argparse.ArgumentParser()
  # compilation
  parser.add_argument('-verbose', action='store_true', default=False, help='enable verbose compilation mode')
  parser.add_argument('--build', default=DEF_build_dir, help='specify building directory')
  parser.add_argument('--bin', default=DEF_bin_dir, help='specify directory for executables')
  parser.add_argument('--compiler', default=DEF_compiler, help='choose the compiler')
  parser.add_argument('-debug', action='store_true', default=False, help='compile in `debug` mode')

  # `Kokkos` specific
  parser.add_argument('-kokkos', action='store_true', default=False, help='compile with `Kokkos` support')
  parser.add_argument('--kokkos_devices', default='Serial', choices=Kokkos_devices_options, help='`Kokkos` devices')
  parser.add_argument('--kokkos_arch', default='SKX', choices=Kokkos_arch_options, help='`Kokkos` architecture')
  parser.add_argument('--kokkos_options', default='', help='`Kokkos` options')
  parser.add_argument('--kokkos_loop', default='default', choices=Kokkos_loop_options, help='`Kokkos` loop layout')
  parser.add_argument('--kokkos_cuda_options', default='', help='`Kokkos` CUDA options')
  parser.add_argument('--nvcc_wrapper_cxx', default='g++', help='Sets the `NVCC_WRAPPER_DEFAULT_COMPILER` flag for `Kokkos`')
  parser.add_argument('--kokkos_vector_length', default=-1, type=int, help='`Kokkos` vector length')
  return vars(parser.parse_args())

def configureKokkos(arg, mopt):
  if arg['kokkos']:
    # using Kokkos
    # custom flag to recognize that the code is compiled with `Kokkos`
    # check compatibility between arch and device
    is_on_host = (arg['kokkos_devices'] in Kokkos_devices['host']) and (arg['kokkos_arch'] in Kokkos_arch['host'])
    is_on_device = (arg['kokkos_devices'] in Kokkos_devices['device']) and (arg['kokkos_arch'] in Kokkos_arch['device'])
    assert is_on_host or is_on_device, "Incompatible device & arch specified"
    mopt['KOKKOS_DEVICES'] = arg['kokkos_devices']
    mopt['KOKKOS_ARCH'] = arg['kokkos_arch']

    mopt['KOKKOS_OPTIONS'] = arg['kokkos_options']
    if mopt['KOKKOS_OPTIONS'] != '':
      mopt['KOKKOS_OPTIONS'] += ','
    mopt['KOKKOS_OPTIONS'] += 'disable_deprecated_code'

    if arg['nvcc_wrapper_cxx'] != '':
      mopt['COMPILER'] = arg['nvcc_wrapper_cxx']
    mopt['KOKKOS_CUDA_OPTIONS'] = arg['kokkos_cuda_options']

    if 'Cuda' in mopt['KOKKOS_DEVICES']:
      # using Cuda
      mopt['KOKKOS_CUDA_OPTIONS'] = arg['kokkos_cuda_options']
      if mopt['KOKKOS_CUDA_OPTIONS'] != '':
        mopt['KOKKOS_CUDA_OPTIONS'] += ','
      mopt['KOKKOS_CUDA_OPTIONS'] += 'enable_lambda'

      # no MPI (TODO)
      mopt['COMPILER'] = f'NVCC_WRAPPER_DEFAULT_COMPILER={arg["nvcc_wrapper_cxx"]} ' + '${KOKKOS_PATH}/bin/nvcc_wrapper'
      # add with MPI here (TODO)

    if arg['kokkos_loop'] == 'default':
      arg['kokkos_loop'] = '1DRange' if 'Cuda' in arg['kokkos_devices'] else 'for'

    mopt['KOKKOS_VECTOR_LENGTH'] = '-1'
    if arg['kokkos_loop'] == '1DRange':
      mopt['KOKKOS_LOOP_LAYOUT'] = '-D MANUAL1D_LOOP'
    elif arg['kokkos_loop'] == 'MDRange':
      mopt['KOKKOS_LOOP_LAYOUT'] = '-D MDRANGE_LOOP'
    elif arg['kokkos_loop'] == 'for':
      mopt['KOKKOS_LOOP_LAYOUT'] = '-D FOR_LOOP'
    elif arg['kokkos_loop'] == 'TP-TVR':
      mopt['KOKKOS_LOOP_LAYOUT'] = '-D TP_INNERX_LOOP -D INNER_TVR_LOOP'
      mopt['KOKKOS_VECTOR_LENGTH'] = ('32' if arg['kokkos_vector_length'] == -1 else str(arg['kokkos_vector_length']))
    elif arg['kokkos_loop'] == 'TP-TTR':
      mopt['KOKKOS_LOOP_LAYOUT'] = '-D TP_INNERX_LOOP -D INNER_TTR_LOOP'
      mopt['KOKKOS_VECTOR_LENGTH'] = ('1' if arg['kokkos_vector_length'] == -1 else str(arg['kokkos_vector_length']))
    elif arg['kokkos_loop'] == 'TP-TTR-TVR':
      mopt['KOKKOS_LOOP_LAYOUT'] = '-D TPTTRTVR_LOOP'
      mopt['KOKKOS_VECTOR_LENGTH'] = ('32' if arg['kokkos_vector_length'] == -1 else str(arg['kokkos_vector_length']))

    mopt['KOKKOS_VECTOR_LENGTH'] = '-D KOKKOS_VECTOR_LENGTH=' + mopt['KOKKOS_VECTOR_LENGTH']

    settings = f'''
  `Kokkos`:
    {'Architecture':30} {arg['kokkos_arch'] if arg['kokkos_arch'] else '-'}
    {'Devices':30} {arg['kokkos_devices'] if arg['kokkos_devices'] else '-'}
    {'Options':30} {arg['kokkos_options'] if arg['kokkos_options'] else '-'}
    {'Loop':30} {arg['kokkos_loop']}
    {'Vector length':30} {arg['kokkos_vector_length']}
  '''
    return settings
  else:
    return ''

def createMakefile(m_in, m_out, mopt):
  with open(m_in, 'r') as current_file:
    makefile_template = current_file.read()
  for key, val in mopt.items():
    makefile_template = re.sub(r'@{0}@'.format(key), val, makefile_template)
  makefile_template = re.sub('# Template for ', '# ', makefile_template)
  with open(args['build'] + '/' + m_out, 'w') as current_file:
    current_file.write(makefile_template)
# <-- auxiliary functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

# Step 1. Prepare parser, add each of the arguments
args = defineOptions()

# Step 2. Set definitions and Makefile options based on above arguments

makefile_options = {}

# Settings
makefile_options['VERBOSE'] = ('y' if args['verbose'] else 'n')
makefile_options['DEBUGMODE'] = ('y' if args['debug'] else 'n')
makefile_options['USEKOKKOS'] = ('y' if args['kokkos'] else 'n')

# Compilation commands
makefile_options['COMPILER'] = args['compiler']
makefile_options['CXXSTANDARD'] = f'{DEF_cppstandard}'

# Paths
makefile_options['BUILD_DIR'] = args['build']
makefile_options['BIN_DIR'] = args['bin']
makefile_options['SRC_DIR'] = 'src'
makefile_options['EXTERN_DIR'] = 'extern'

Path(args['build']).mkdir(parents=True, exist_ok=True)

# `Kokkos` settings
Kokkos_details = configureKokkos(args, makefile_options)

# Configuration flags for the performance build (TODO: compiler specific)
makefile_options['RELEASE_CONF_FLAGS'] = "-O3 -Ofast"
makefile_options['RELEASE_PP_FLAGS'] = "-DNDEBUG"

# Configuration flags for the debug build (TODO: compiler specific)
makefile_options['DEBUG_CONF_FLAGS'] = ""
makefile_options['DEBUG_PP_FLAGS'] = "-O0 -g -DDEBUG"

# Warning flags (TODO: compiler specific)
makefile_options['WARNING_FLAGS'] = "-Wall -Wextra -pedantic"

# Step 3. Create new files, finish up
createMakefile(makefile_input, makefile_output, makefile_options)

#  Finish with diagnostic output
report = f'''
====================================================
Code has been configured with the following options:

Computational details ..............................
  {'Use `Kokkos` Library':32} {args['kokkos']}

Technical details ..................................
  {'Compiler':32} {makefile_options['COMPILER']}
  {'Debug mode':32} {args['debug']}
  {Kokkos_details}
====================================================
'''

print (report)

with open(args['build'] + "/REPORT", 'w') as reportfile:
  reportfile.write(report)
