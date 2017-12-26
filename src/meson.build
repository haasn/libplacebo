majorver = '0'
apiver   = '2'
fixver   = '0'

version = majorver + '.' + apiver + '.' + fixver

# Build options mostly taken from mpv
build_opts = [
  '-D_ISOC99_SOURCE', '-D_GNU_SOURCE',
  '-fvisibility=hidden',

  # Warnings
  '-Wall', '-Wundef', '-Wmissing-prototypes', '-Wshadow', '-Wparentheses',
  '-Wpointer-arith', '-Wno-pointer-sign',
]

cc = meson.get_compiler('c')

# clang's version of -Wmissing-braces rejects the common {0} initializers
if cc.get_id() == 'clang'
  build_opts += ['-Wno-missing-braces']
endif

# Global dependencies
build_deps = [
  dependency('threads'),
  cc.find_library('m', required: false),
]

# Source files
sources = [
  'colorspace.c',
  'common.c',
  'context.c',
  'dither.c',
  'dispatch.c',
  'filters.c',
  'ra.c',
  'renderer.c',
  'shaders.c',
  'shaders/colorspace.c',
  'shaders/sampling.c',
  'spirv.c',

  # Helpers ported from mpv or other projects
  'bstr/bstr.c',
  'siphash.c',
  'ta/ta.c',
  'ta/ta_utils.c',
  'ta/talloc.c',
]

tests = [
  'context.c',
  'colorspace.c',
  'dither.c',
  'filters.c',
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
    ],
    'vulkan.c'
  ]
]

# OS-dependent files and feature checks
has_useloc_apple = cc.has_header_symbol('xlocale.h', 'uselocale')
has_useloc_win   = cc.has_header_symbol('locale.h', '_create_locale')
has_useloc_posix = cc.has_header_symbol('locale.h', 'uselocale',
  prefix: '#define _XOPEN_SOURCE 700'
)

## osdep/printf.h
if has_useloc_posix or has_useloc_apple
  sources += 'osdep/printf_useloc_posix.c'
elif has_useloc_win
  sources += 'osdep/printf_useloc_win.c'
  build_deps += cc.find_library('msvcr100', required: true)
else
  error('Could not find any suitable implementation of `osdep/printf.h`!')
endif

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
defs = ''

foreach c : components
  name = c[0]
  dep  = c[1]
  opt  = get_option(name)

  has_dep = opt == 'false' ? false : dep.found()
  if opt == 'true' and not has_dep
    error(('Dependency \'@0@\', marked as required, is not found. To build ' +
           'without support for @0@, use -D@0@=false.').format(name))
  endif

  pretty = name.underscorify().to_upper()
  defs += '#define PL_HAVE_@0@ @1@\n'.format(pretty, has_dep ? 1 : 0)
  if has_dep
    build_deps += dep
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

add_project_arguments(build_opts, language: 'c')
lib = library('placebo', sources,
  install: true,
  dependencies: build_deps,
  soversion: apiver,
)

# Install process
install_subdir('include/libplacebo', install_dir: get_option('includedir'))

pkg = import('pkgconfig')
pkg.generate(
  name: meson.project_name(),
  description: 'Reusable library for GPU-accelerated video/image rendering',
  libraries: lib,
  version: version,
)

# Tests

tdeps = [ declare_dependency(link_with: lib) ]

if get_option('tests')
  foreach t : tests
    e = executable('test.' + t, 'tests/' + t, dependencies: build_deps + tdeps)
    test(t, e)
  endforeach
endif
