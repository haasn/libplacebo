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

# Source files
headers = [
  'public/colorspace.h',
  'public/common.h',
  'public/context.h',
  'public/filters.h',
  'public/ra.h',
  'public/shaders.h',
  'public/vulkan.h',
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

tests = [
  'context.c',
  'colorspace.c',
]

# Optional components, in the following format:
# [ name, dependency, extra_sources, extra_tests ]
components = [
  [
    'shaderc',
    cc.find_library('shaderc_shared', required: false),
    'spirv_shaderc.c',
  ], [
    'vulkan',
    dependency('vulkan', version: '>=1.0', required: false),
    [ 'vulkan/command.c',
      'vulkan/context.c',
      'vulkan/formats.c',
      'vulkan/malloc.c',
      'vulkan/ra_vk.c',
      'vulkan/utils.c',
    ]
  ]
]

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

# Build process
bdeps = [ cc.find_library('m', required: false) ]
defs = ''

foreach c : components
  name = c[0].underscorify().to_upper()
  dep  = c[1]
  defs += '#define PL_HAVE_@0@ @1@\n'.format(name, dep.found() ? 1 : 0)
  if dep.found()
    bdeps   += dep
    if (c.length() > 2)
      sources += c[2]
    endif
    if (c.length() > 3)
      tests += c[3]
    endif
  endif
endforeach

conf.set('extra_defs', defs)

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

# Tests

tdeps = declare_dependency(link_with: lib)

foreach t : tests
  e = executable(t, 'tests/' + t, dependencies: tdeps)
  test(t, e)
endforeach
