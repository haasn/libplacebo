majorver = '0'
apiver   = '0'
fixver   = '0'

version = majorver + '.' + apiver + '.' + fixver

# Build options mostly taken from mpv
add_global_arguments([
  '-D_ISOC99_SOURCE', '-D_GNU_SOURCE',
  '-fvisibility=hidden',

  # Warnings
  '-Wall', '-Wundef', '-Wmissing-prototypes', '-Wshadow', '-Wparentheses',
  '-Wpointer-arith', '-Wno-pointer-sign',
], language: 'c')

cc = meson.get_compiler('c')

# Configuration
conf = configuration_data()
conf.set('majorver', majorver)
conf.set('apiver', apiver)
conf.set('fixver', fixver)

gitdesc = run_command('git', 'describe')
if gitdesc.returncode() == 0
  conf.set_quoted('version', gitdesc.stdout().strip())
else
  conf.set_quoted('version', 'v' + version)
endif

# Source files
headers = [
  'public/colorspace.h',
  'public/common.h',
  'public/context.h',
  'public/filters.h',
  'public/ra.h',
  'public/shaders.h',
]

sources = [
  'colorspace.c',
  'common.c',
  'context.c',
  'filters.c',
  'ra.c',
  'shaders.c',
  'spirv.c',

  # Helpers ported from mpv
  'bstr/bstr.c',
  'ta/ta.c',
  'ta/ta_utils.c',
  'ta/talloc.c',
]

# Optional components, in the format [ name, dependency, extra_sources ]
components = [
  [
    'shaderc',
    cc.find_library('shaderc_shared', required: false),
    'spirv_shaderc.c',
  ],
]

# Build process
bdeps = [ cc.find_library('m', required: false) ]

foreach c : components
  name = c[0]
  dep  = c[1]
  srcs = c[2]

  conf.set10('PL_HAVE_' + name.underscorify().to_upper(), dep.found())
  if dep.found()
    bdeps   += dep
    sources += srcs
  endif
endforeach

configure_file(
  input: 'config.h.in',
  output: 'config.h',
  install_dir: 'include/libplacebo',
  configuration: conf,
)

lib = library('placebo', sources,
  install: true,
  dependencies: bdeps,
  soversion: apiver,
)

# Install process
install_headers(headers, subdir: 'libplacebo')

pkg = import('pkgconfig')
pkg.generate(
  name: meson.project_name(),
  description: 'Reusable library for GPU-accelerated video/image rendering',
  libraries: lib,
  version: version,
)
