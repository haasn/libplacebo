majorver = '0'
apiver   = '0'
fixver   = '0'

# Build options mostly taken from mpv
add_global_arguments([
  '-D_ISOC99_SOURCE', '-D_GNU_SOURCE',
  '-fvisibility=hidden',

  # Warnings
  '-Wall', '-Wundef', '-Wmissing-prototypes', '-Wshadow', '-Wparentheses',
  '-Wpointer-arith', '-Wno-pointer-sign',
], language: 'c')

cc = meson.get_compiler('c')

# Dependencies
deps = [
  # libm
  cc.find_library('m', required: false),
]

# Configuration
conf = configuration_data()
conf.set('apiver', apiver)

configure_file(
  input: 'config.h.in',
  output: 'config.h',
  install_dir: 'include/libplacebo',
  configuration: conf,
)

# Source files
sources = [
  'colorspace.c',
  'context.c',
  'filters.c',
  'ra.c',
  'shaders.c',

  # Helpers ported from mpv
  'bstr/bstr.c',
  'ta/ta.c',
  'ta/ta_utils.c',
  'ta/talloc.c',
]

shared_library('placebo', sources,
  install: true,
  dependencies: deps,
  soversion: apiver,
)

headers = [
  'public/colorspace.h',
  'public/common.h',
  'public/context.h',
  'public/filters.h',
  'public/ra.h',
  'public/shaders.h',
]

install_headers(headers, subdir: 'libplacebo')

pkg = import('pkgconfig')
pkg.generate(
  name: meson.project_name(),
  description: 'Reusable library for GPU-accelerated video/image rendering',
  libraries: '-lplacebo',
  version: majorver + '.' + apiver + '.' + fixver,
)
