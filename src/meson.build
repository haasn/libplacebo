majorver = '1'
apiver   = '7'
fixver   = '0'

version = majorver + '.' + apiver + '.' + fixver

c_opts = [
  '-D_ISOC99_SOURCE', '-D_GNU_SOURCE', '-D_XOPEN_SOURCE=700',
  '-U__STRICT_ANSI__', '-fvisibility=hidden',
  '-Wmissing-prototypes', '-Wno-pointer-sign'
]

# glslang needs c++11
cpp_opts = [
  '-std=c++11', '-fvisibility=hidden',
]

# Build options mostly taken from mpv
build_opts = [
  # Warnings
  '-Wall', '-Wundef', '-Wshadow', '-Wparentheses', '-Wpointer-arith',

  # Warnings to treat as errors
  '-Werror=implicit-function-declaration',
]

link_args = []

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

if cc.has_argument('-Wincompatible-pointer-types')
  build_opts += ['-Werror=incompatible-pointer-types']
endif

# clang's version of -Wmissing-braces rejects the common {0} initializers
if cc.get_id() == 'clang'
  build_opts += ['-Wno-missing-braces']
endif

vflag = '-Wl,--exclude-libs=ALL'
if cc.has_link_argument(vflag)
  link_args += [vflag]
endif

# Global dependencies
build_deps = [
  dependency('threads'),
  cc.find_library('m', required: false),
]

vulkan = dependency('vulkan', version: '>=1.0.42', required: get_option('vulkan'))

# Source files
sources = [
  'colorspace.c',
  'common.c',
  'context.c',
  'dither.c',
  'dispatch.c',
  'filters.c',
  'gpu.c',
  'renderer.c',
  'shaders.c',
  'shaders/av1.c',
  'shaders/colorspace.c',
  'shaders/sampling.c',
  'spirv.c',
  'swapchain.c',
  'utils/upload.c',

  # Helpers ported from mpv or other projects
  'bstr/bstr.c',
  'bstr/format.c',
  '3rdparty/siphash.c',
  'ta/ta.c',
  'ta/ta_utils.c',
  'ta/talloc.c',
]

tests = [
  'context.c',
  'colorspace.c',
  'dither.c',
  'filters.c',
  'utils.c',
]

# Work-arounds for glslang braindeath
glslang_combined = disabler()
glslang_min_ver = 2763
glslang_req = get_option('glslang')
glslang_deps = [
  cxx.find_library('glslang',     required: glslang_req),
  cxx.find_library('HLSL',        required: glslang_req),
  cxx.find_library('OGLCompiler', required: glslang_req),
  cxx.find_library('OSDependent', required: glslang_req),
  cxx.find_library('SPIRV',       required: glslang_req),
  cxx.find_library('SPVRemapper', required: glslang_req),
]

glslang_found = true
foreach d : glslang_deps
  glslang_found = glslang_found and d.found()
endforeach

if glslang_found
  glslang_ver = cxx.get_define('GLSLANG_PATCH_LEVEL',
      prefix: '#include <glslang/Include/revision.h>'
  ).to_int()

  if glslang_ver >= glslang_min_ver
    # glslang must be linked against pthreads on platforms where pthreads is
    # available. Because of their horrible architecture, gcc can't do it
    # automatically, and for some reason dependency('threads') (which uses
    # -pthread) doesn't work. We actually need -lpthreads for the glslang
    # object files to link, for whatever weird reason.
    pthread = cxx.find_library('pthread', required: false)
    glslang_combined = declare_dependency(dependencies: glslang_deps + [pthread])
  else
    error('glslang revision @0@ too old! Must be at least @1@'
          .format(glslang_ver, glslang_min_ver))
  endif
endif

# Optional dependencies / components
components = [
  {
    'name': 'lcms',
    'deps':  dependency('lcms2', version: '>=2.6', required: get_option('lcms')),
    'srcs': 'lcms.c',
  }, {
    'name': 'glslang',
    'deps': glslang_combined,
    'srcs': [ 'glsl/glslang.cc',
              'spirv_glslang.c',
            ],
  }, {
    'name': 'shaderc',
    'deps': cc.find_library('shaderc_shared', required: get_option('shaderc')),
    'srcs': 'spirv_shaderc.c',
  }, {
    'name': 'vulkan',
    'deps': vulkan,
    'srcs': [ 'vulkan/command.c',
              'vulkan/context.c',
              'vulkan/formats.c',
              'vulkan/gpu.c',
              'vulkan/malloc.c',
              'vulkan/swapchain.c',
              'vulkan/utils.c',
            ],
    'test': 'vulkan.c'
  }
]

# Configuration
conf = configuration_data()
conf.set('majorver', majorver)
conf.set('apiver', apiver)
conf.set('fixver', fixver)
conf.set_quoted('version', 'v' + version)

## Update PL_VERSION with `git describe` information if available
git = find_program('git', required: false)
if git.found()
  gitdesc = run_command(git, 'describe')
  if gitdesc.returncode() == 0
    conf.set_quoted('version', gitdesc.stdout().strip())
  endif
endif

# Build process
defs = ''
comps = configuration_data()

foreach c : components
  name = c['name']
  deps = c['deps']

  pretty = name.underscorify().to_upper()
  defs += '#define PL_HAVE_@0@ @1@\n'.format(pretty, deps.found() ? 1 : 0)

  if deps.found()
    comps.set(name, 1)
    build_deps += deps
    sources += c.get('srcs', [])
    tests += c.get('test', [])
  endif
endforeach

# Check to see if libplacebo built this way is sane
if not comps.has('vulkan')
  warning('Building without support for Vulkan. Currently, libplacebo only ' +
          'supports the Vulkan graphics API. libplacebo built this way still ' +
          'has some limited use (e.g. generating GLSL shaders), but most of ' +
          'its functionality will be missing or impaired!')
endif

if comps.has('vulkan') and not (comps.has('shaderc') or comps.has('glslang'))
  error('Building with support for Vulkan requires either `shaderc` or ' +
        '`glslang` to be of any use, otherwise libplacebo would fail to ' +
        'compile GLSL to SPIR-V (needed by the Vulkan API)!')
endif

# Generate the config.h file
conf.set('extra_defs', defs)

configure_file(
  input: 'config.h.in',
  output: 'config.h',
  install_dir: 'include/libplacebo',
  configuration: conf,
)

# Build process
add_project_arguments(build_opts + c_opts, language: 'c')
add_project_arguments(build_opts + cpp_opts, language: 'cpp')
inc = include_directories('./include')
lib = library('placebo', sources,
  install: true,
  dependencies: build_deps,
  soversion: apiver,
  include_directories: inc,
  link_args: link_args,
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
    e = executable('test.' + t, 'tests/' + t,
        dependencies: build_deps + tdeps,
        objects: lib.extract_all_objects(),
        include_directories: inc
    )

    test(t, e)
  endforeach
endif

if get_option('bench')
  if not vulkan.found()
    error('Compiling the benchmark suite requires vulkan support!')
  endif

  bench = executable('bench', 'tests/bench.c',
      dependencies: build_deps + tdeps,
      include_directories: inc
  )

  test('benchmark', bench, is_parallel: false, timeout: 600)
endif
