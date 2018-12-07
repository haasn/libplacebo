# Depedencies
cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

build_deps = [
  cc.find_library('m', required: false),
  subproject('xtalloc').get_variable('dep'),
  subproject('bstr').get_variable('dep'),
]

vulkan = dependency('vulkan', version: '>=1.0.42', required: get_option('vulkan'))

# configuration data
conf_public = configuration_data()
conf_internal = configuration_data()
conf_public.set('majorver', majorver)
conf_public.set('apiver', apiver)
conf_public.set('fixver', fixver)
conf_public.set_quoted('version', version_pretty)


# work-arounds for shaderc braindeath
shaderc_names = [
    'shaderc_shared',
    'shaderc_combined',
]

shaderc = disabler()
foreach n : shaderc_names
  if not shaderc.found()
    shaderc = cc.find_library(n, required: get_option('shaderc'))
  endif
endforeach

if shaderc.found()
  conf_internal.set('SHADERC_HAS_PERF', cc.has_header_symbol(
    'shaderc/shaderc.h',
    'shaderc_optimization_level_performance',
    dependencies: shaderc,
  ))
endif

# work-arounds for glslang braindeath
glslang_combined = disabler()
glslang_min_ver = 2763
glslang_req = get_option('glslang')

if glslang_req.auto() and shaderc.found()

  # we only need one or the other, and shaderc is preferred
  message('Skipping `glslang` because `shaderc` is available')
  glslang_req = false
  glslang_found = false

else

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

endif

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
]

tests = [
  'context.c',
  'colorspace.c',
  'dither.c',
  'filters.c',
  'utils.c',
]


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

# check to see if libplacebo built this way is sane
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


# Build process
conf_public.set('extra_defs', defs)

configure_file(
  input: 'config.h.in',
  output: 'config.h',
  install_dir: 'include/libplacebo',
  configuration: conf_public,
)

configure_file(
  output: 'config_internal.h',
  configuration: conf_internal
)

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
tdep = declare_dependency(
    link_with: lib,
    dependencies: build_deps,
    include_directories: inc,
)

if get_option('tests')
  foreach t : tests
    e = executable('test.' + t, 'tests/' + t,
        objects: lib.extract_all_objects(recursive: false),
        dependencies: tdep,
    )

    test(t, e)
  endforeach
endif

if get_option('bench')
  if not vulkan.found()
    error('Compiling the benchmark suite requires vulkan support!')
  endif

  bench = executable('bench', 'tests/bench.c', dependencies: tdep)
  test('benchmark', bench, is_parallel: false, timeout: 600)
endif
