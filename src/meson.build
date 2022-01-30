# Configuration data
conf_public = configuration_data()
conf_internal = configuration_data()
conf_public.set('majorver', majorver)
conf_public.set('apiver', apiver)
conf_internal.set('BUILD_API_VER', apiver)
conf_internal.set('BUILD_FIX_VER', fixver)

# Dependencies
prog_python = import('python').find_installation()
libm = cc.find_library('m', required: false)

if host_machine.system() == 'windows'
    threads = declare_dependency()
else
    threads = dependency('threads')
    conf_internal.set('PL_HAVE_PTHREAD', threads.found())
    conf_internal.set('PTHREAD_HAS_SETCLOCK', cc.has_header_symbol(
      'pthread.h',
      'pthread_condattr_setclock',
      dependencies: threads,
      args: c_opts,
    ))
endif

build_deps = [ libm, threads ]
test_deps = []

vulkan = dependency('vulkan', required: get_option('vulkan'))
opengl = dependency('epoxy', version: '>=1.4.0', required: get_option('opengl'))
shaderc = dependency('shaderc', version: '>=2019.1', required: get_option('shaderc'))
cross = dependency('spirv-cross-c-shared', version: '>=0.29.0', required: get_option('d3d11'))

if opengl.found()
  has_egl = opengl.get_variable(pkgconfig: 'epoxy_has_egl', default_value: '0')
  conf_internal.set('EPOXY_HAS_EGL', has_egl.to_int() == 1)
endif

vulkan_headers = vulkan
if vulkan.found() and not get_option('vulkan-link')
  vulkan_headers = vulkan.partial_dependency(includes: true, compile_args: true)
endif

if not vulkan_headers.found() and not get_option('vulkan').disabled()
  # Probe for <vulkan/vulkan.h> as a fallback for the vulkan headers
  if cc.check_header('vulkan/vulkan.h')
    vulkan_headers = declare_dependency()
  endif
endif

d3d11 = disabler()
d3d11_inc = ['d3d11_4.h', 'dxgi1_6.h']
d3d11_deps = [
  cross,
  cc.find_library('version', required: get_option('d3d11')),
]

d3d11_found = true
foreach h : d3d11_inc
  d3d11_found = d3d11_found and cc.check_header(h, required: get_option('d3d11'))
endforeach
foreach d : d3d11_deps
  d3d11_found = d3d11_found and d.found()
endforeach

if d3d11_found
  d3d11 = declare_dependency(dependencies: d3d11_deps)
  add_project_arguments(['-DCOBJMACROS', '-DINITGUID'], language: 'c')
endif

unwind = dependency('libunwind', required: get_option('unwind'))
has_execinfo = cc.has_header('execinfo.h')
conf_internal.set('PL_HAVE_UNWIND', unwind.found())
conf_internal.set('PL_HAVE_EXECINFO', has_execinfo)
if unwind.found()
  build_deps += [unwind, cc.find_library('dl', required : false)]
elif has_execinfo
  build_deps += cc.find_library('execinfo', required: false)
endif

# work-arounds for glslang braindeath
glslang_combined = disabler()
glslang_min_ver = '>=0.0.2763'
glslang_req = get_option('glslang')

if glslang_req.auto() and shaderc.found()

  # we only need one or the other, and shaderc is preferred
  message('Skipping `glslang` because `shaderc` is available')
  glslang_req = false
  glslang_found = false

else

  glslang_deps = [
    cxx.find_library('glslang',             required: glslang_req),
    cxx.find_library('MachineIndependent',  required: false),
    cxx.find_library('OSDependent',         required: glslang_req),
    cxx.find_library('HLSL',                required: glslang_req),
    cxx.find_library('OGLCompiler',         required: glslang_req),
    cxx.find_library('GenericCodeGen',      required: false),
    cxx.find_library('SPVRemapper',         required: glslang_req),
    cxx.find_library('SPIRV',               required: glslang_req),
    cxx.find_library('SPIRV-Tools-opt',     required: false),
    cxx.find_library('SPIRV-Tools',         required: false),
  ]

  glslang_found = glslang_deps[0].found()

endif

if glslang_found
  glslang_header_old = 'glslang/Include/revision.h'
  glslang_header_new = 'glslang/build_info.h'

  if cc.has_header(glslang_header_new)
    glslang_ver_major = cxx.get_define('GLSLANG_VERSION_MAJOR',
        prefix: '#include <' + glslang_header_new + '>'
    ).to_int()
    glslang_ver_minor = cxx.get_define('GLSLANG_VERSION_MINOR',
        prefix: '#include <' + glslang_header_new + '>'
    ).to_int()
    glslang_ver_patch = cxx.get_define('GLSLANG_VERSION_PATCH',
        prefix: '#include <' + glslang_header_new + '>'
    ).to_int()
  elif cc.has_header(glslang_header_old)
    # This is technically incorrect, but since we don't care about major
    # versions for this version range, it's an acceptable substitute
    glslang_ver_major = 0
    glslang_ver_minor = 0
    glslang_ver_patch = cxx.get_define('GLSLANG_PATCH_LEVEL',
        prefix: '#include <' + glslang_header_old+ '>'
    ).to_int()
  else
    error('No glslang version header found?')
  endif

    glslang_ver = '@0@.@1@.@2@'.format(
        glslang_ver_major,
        glslang_ver_minor,
        glslang_ver_patch,
    )

  if glslang_ver.version_compare(glslang_min_ver)
    # glslang must be linked against pthreads on platforms where pthreads is
    # available. Because of their horrible architecture, gcc can't do it
    # automatically, and for some reason dependency('threads') (which uses
    # -pthread) doesn't work. We actually need -lpthreads for the glslang
    # object files to link, for whatever weird reason.
    pthread = cxx.find_library('pthread', required: false)
    glslang_all_deps = glslang_deps + [pthread]
    glslang_combined = declare_dependency(dependencies: glslang_all_deps)

    # Work around a glslang include path bug w.r.t stuff previously namespaced
    # under /usr/include/SPIRV now being moved to /usr/include/glslang/SPIRV.
    extra_glslang_inc = [
      '/usr/include/glslang',
      get_option('prefix') / get_option('includedir') / 'glslang',
    ]

    foreach i : extra_glslang_inc
      add_project_arguments('-I' + i, language: 'cpp')
    endforeach

    conf_internal.set('GLSLANG_VERSION_MAJOR', glslang_ver_major)
    conf_internal.set('GLSLANG_VERSION_MINOR', glslang_ver_minor)
    conf_internal.set('GLSLANG_VERSION_PATCH', glslang_ver_patch)

  else
    error('glslang version @0@ too old! Must be at least @1@'
          .format(glslang_ver, glslang_min_ver))
  endif
endif

# Work around missing atomics on some (obscure) platforms
atomic_test = '''
#include <stdatomic.h>
#include <stdint.h>
int main(void) {
  _Atomic uint32_t x32;
  atomic_init(&x32, 0);
}'''

if not cc.links(atomic_test)
  build_deps += cc.find_library('atomic')
endif

# Source files
headers = [
  'colorspace.h',
  'common.h',
  'context.h',
  'dispatch.h',
  'dither.h',
  'dummy.h',
  'filters.h',
  'gpu.h',
  'log.h',
  'renderer.h',
  'shaders/colorspace.h',
  'shaders/custom.h',
  'shaders/film_grain.h',
  'shaders/lut.h',
  'shaders/sampling.h',
  'shaders.h',
  'swapchain.h',
  'tone_mapping.h',
  'utils/dav1d.h',
  'utils/dav1d_internal.h',
  'utils/frame_queue.h',
  'utils/libav.h',
  'utils/libav_internal.h',
  'utils/upload.h',
]

sources = [
  'colorspace.c',
  'common.c',
  'dither.c',
  'dispatch.c',
  'dummy.c',
  'filters.c',
  'format.c',
  'glsl/spirv.c',
  'glsl/utils.c',
  'gpu.c',
  'log.c',
  'pl_alloc.c',
  'pl_string.c',
  'renderer.c',
  'siphash.c',
  'shaders.c',
  'shaders/colorspace.c',
  'shaders/custom.c',
  'shaders/film_grain.c',
  'shaders/film_grain_av1.c',
  'shaders/film_grain_h274.c',
  'shaders/lut.c',
  'shaders/sampling.c',
  'swapchain.c',
  'tone_mapping.c',
  'utils/frame_queue.c',
  'utils/upload.c',
]

tests = [
  'colorspace.c',
  'common.c',
  'dither.c',
  'dummy.c',
  'lut.c',
  'filters.c',
  'string.c',
  'tone_mapping.c',
  'utils.c',
]

fuzzers = [
  'lut.c',
  'shaders.c',
  'user_shaders.c',
]


# Optional dependencies / components
components = [
  {
    'name': 'lcms',
    'deps':  dependency('lcms2', version: '>=2.6', required: get_option('lcms')),
    'srcs': 'shaders/icc.c',
    'headers': 'shaders/icc.h',
  }, {
    'name': 'glslang',
    'deps': glslang_combined,
    'srcs': [ 'glsl/glslang.cc',
              'glsl/spirv_glslang.c',
            ],
  }, {
    'name': 'shaderc',
    'deps': shaderc,
    'srcs': 'glsl/spirv_shaderc.c',
  }, {
    'name': 'vulkan',
    'deps': vulkan_headers,
    'srcs': [ 'vulkan/command.c',
              'vulkan/context.c',
              'vulkan/formats.c',
              'vulkan/gpu.c',
              'vulkan/gpu_buf.c',
              'vulkan/gpu_tex.c',
              'vulkan/gpu_pass.c',
              'vulkan/malloc.c',
              'vulkan/swapchain.c',
              'vulkan/utils.c',
            ],
    'headers': 'vulkan.h',
  }, {
    'name': 'opengl',
    'deps': opengl,
    'srcs': [ 'opengl/context.c',
              'opengl/formats.c',
              'opengl/gpu.c',
              'opengl/gpu_tex.c',
              'opengl/gpu_pass.c',
              'opengl/swapchain.c',
              'opengl/utils.c',
            ],
    'headers': 'opengl.h',
    'test': 'opengl_surfaceless.c',
  }, {
    'name': 'd3d11',
    'deps': d3d11,
    'srcs': [ 'd3d11/context.c',
              'd3d11/formats.c',
              'd3d11/gpu.c',
              'd3d11/gpu_buf.c',
              'd3d11/gpu_tex.c',
              'd3d11/gpu_pass.c',
              'd3d11/swapchain.c',
              'd3d11/utils.c',
            ],
    'headers': 'd3d11.h',
    'test': 'd3d11.c',
   }
]

defs = ''
pc_vars = []
comps = configuration_data()

foreach c : components
  name = c['name']
  deps = c['deps']

  pretty = name.underscorify().to_upper()

  if deps.found()
    defs += '#define PL_HAVE_@0@ 1\n'.format(pretty)
    pc_vars += 'pl_has_@0@=1'.format(pretty.to_lower())
    comps.set(name, 1)
    build_deps += deps
    sources += c.get('srcs', [])
    headers += c.get('headers', [])
    tests += c.get('test', [])
  else
    defs += '#undef PL_HAVE_@0@\n'.format(pretty)
    pc_vars += 'pl_has_@0@=0'.format(pretty.to_lower())
  endif
endforeach

# Extra checks/steps required for vulkan in particular
if comps.has('vulkan')
  registry_xml = get_option('vulkan-registry')
  sources += custom_target('vulkan boilerplate',
    input: 'vulkan/utils_gen.py',
    output: 'utils_gen.c',
    command: [prog_python, '@INPUT@', registry_xml, '@OUTPUT@']
  )

  if vulkan.found() and get_option('vulkan-link')
    defs += '#define PL_HAVE_VK_PROC_ADDR 1'
    pc_vars += 'pl_has_vk_proc_addr=1'
  else
    defs += '#undef PL_HAVE_VK_PROC_ADDR'
    pc_vars += 'pl_has_vk_proc_addr=0'
  endif
endif

# Check to see if libplacebo built this way is sane
if not (comps.has('vulkan') or comps.has('opengl') or comps.has('d3d11'))
  warning('Building without any graphics API. libplacebo built this way still ' +
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
subdir('./include/libplacebo') # generate config.h in the right location

configure_file(
  output: 'config_internal.h',
  configuration: conf_internal
)

sources += vcs_tag(
  command: ['git', 'describe'],
  fallback: version_pretty,
  replace_string: '@buildver@',
  input: 'version.h.in',
  output: 'version.h',
)

inc = include_directories('./include')
lib = library('placebo', sources,
  install: true,
  dependencies: build_deps,
  soversion: apiver,
  include_directories: inc,
  link_args: link_args,
)

libplacebo = declare_dependency(
  link_with: lib,
  include_directories: inc,
)

# Allows projects to build libplacebo by cloning into ./subprojects/libplacebo
meson.override_dependency('libplacebo', libplacebo)

# Install process
foreach h : headers
  parts = h.split('/')
  path = proj_name
  foreach p : parts
    if p != parts[-1]
      path = path / p
    endif
  endforeach

  install_headers('include' / proj_name / h, subdir: path)
endforeach

pkg = import('pkgconfig')
pkg.generate(
  name: proj_name,
  description: 'Reusable library for GPU-accelerated video/image rendering',
  libraries: lib,
  version: version,
  variables: pc_vars,
)


# Tests
tdep = [ declare_dependency(
    link_with: lib,
    dependencies: build_deps + test_deps,
    include_directories: inc,
) ]

if get_option('tests')
  dav1d = dependency('dav1d', required: false)
  if dav1d.found()
    tdep += dav1d
    tests += 'dav1d.c'
  endif

  lavu = dependency('libavutil', version: '>=55.74.100', required: false)
  lavc = dependency('libavcodec', required: false)
  lavf = dependency('libavformat', required: false)
  libav_found = lavu.found() and lavc.found() and lavf.found()
  if libav_found
    tdep += [lavu, lavc, lavf]
    tests += 'libav.c'
  endif

  if vulkan.found()
    tdep += vulkan
    tests += 'vulkan.c'
  endif

  foreach t : tests
    e = executable('test.' + t, 'tests/' + t,
        objects: lib.extract_all_objects(recursive: false),
        c_args: [ '-Wno-unused-function' ],
        dependencies: tdep,
    )

    test(t, e)
  endforeach

  # Ensure all headers compile
  foreach h : headers

    if (h.contains('internal') or
        h.contains('dav1d') and not dav1d.found() or
        h.contains('libav') and not libav_found)
      continue
    endif

    t = configure_file(
        input: 'tests/include_tmpl.c',
        output: 'include_@0@.c'.format(h.underscorify()),
        configuration: {
          'header': h
        },
    )

    executable('test.include.' + h.underscorify(), t,
        dependencies: tdep,
        c_args: [ '-Wno-unused-function' ],
    )
  endforeach
endif

if get_option('bench')
  if not vulkan.found()
    error('Compiling the benchmark suite requires vulkan support!')
  endif

  bench = executable('bench', 'tests/bench.c', dependencies: tdep)
  test('benchmark', bench, is_parallel: false, timeout: 600)
endif

if get_option('fuzz')
  foreach f : fuzzers
    executable('fuzz.' + f, 'tests/fuzz/' + f,
        objects: lib.extract_all_objects(recursive: false),
        dependencies: tdep,
    )
  endforeach
endif
