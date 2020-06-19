project('libplacebo', ['c', 'cpp'],
  license: 'LGPL2.1+',
  default_options: ['c_std=c99', 'cpp_std=c++11', 'warning_level=2'],
  meson_version: '>=0.49',
  version: '2.80.0',
)

# Version number
version = meson.project_version()
version_pretty = 'v' + version
version_split = version.split('.')

majorver = version_split[0]
apiver = version_split[1]
fixver = version_split[2]

proj_name = meson.project_name()

# update `version_pretty` with `git describe` information if available
git = find_program('git', required: false)
if git.found()
  gitdesc = run_command(git, 'describe')
  if gitdesc.returncode() == 0
    version_pretty = gitdesc.stdout().strip()
  endif
endif


# Global build options
build_opts = [
  # Warnings
  '-Wundef', '-Wshadow', '-Wparentheses', '-Wpointer-arith',
]

link_args = []

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

c_opts = [
  '-D_ISOC99_SOURCE', '-D_GNU_SOURCE', '-D_XOPEN_SOURCE=700',
  '-U__STRICT_ANSI__', '-fvisibility=hidden', '-Wmissing-prototypes',

  # Warnings to ignore
  '-Wno-pointer-sign', '-Wno-sign-compare', '-Wno-unused-parameter',
  '-Wno-missing-field-initializers', '-Wno-type-limits',

  # Warnings to treat as errors
  '-Werror=implicit-function-declaration',
]

# glslang needs c++11
cpp_opts = [
  '-fvisibility=hidden',
]

if cc.has_argument('-Wincompatible-pointer-types')
  c_opts += ['-Werror=incompatible-pointer-types']
endif

# clang's version of -Wmissing-braces rejects the common {0} initializers
if cc.get_id() == 'clang'
  c_opts += ['-Wno-missing-braces']
endif

# don't leak library symbols if possible
vflag = '-Wl,--exclude-libs=ALL'
if cc.has_link_argument(vflag)
  link_args += [vflag]
endif

add_global_arguments(build_opts + c_opts, language: 'c')
add_global_arguments(build_opts + cpp_opts, language: 'cpp')
add_global_link_arguments(link_args, language: 'c')

subdir('src')
