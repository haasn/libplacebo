# Configuration data
conf_public = configuration_data()
conf_internal = configuration_data()
conf_public.set('majorver', majorver)
conf_public.set('apiver', apiver)
conf_internal.set('BUILD_API_VER', apiver)
conf_internal.set('BUILD_FIX_VER', fixver)
conf_internal.set('PL_DEBUG_ABORT', get_option('debug-abort'))

# Dependencies
libm = cc.find_library('m', required: false)
libdl = cc.find_library('dl', required : false)
thirdparty = meson.project_source_root()/'3rdparty'
python = import('python').find_installation()
python_env = environment()
python_env.append('PYTHONPATH', thirdparty/'mako')
python_env.append('PYTHONPATH', thirdparty/'jinja'/'src')
python_env.append('PYTHONPATH', thirdparty/'markupsafe'/'src')
python_env.append('PYTHONPATH', thirdparty/'glad')

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

vulkan_opt = get_option('vulkan')
vulkan = dependency('vulkan', version: '>=1.2.0', required: vulkan_opt)
shaderc = dependency('shaderc', version: '>=2019.1', required: get_option('shaderc'))
cross = dependency('spirv-cross-c-shared', version: '>=0.29.0', required: get_option('d3d11'))

opengl = disabler()
if not get_option('opengl').disabled()
  opengl = declare_dependency()
endif

vulkan_headers = vulkan
if vulkan.found() and not get_option('vulkan-link')
  vulkan_headers = vulkan.partial_dependency(includes: true, compile_args: true)
endif

if not vulkan_headers.found() and not vulkan_opt.disabled()
  # Probe for <vulkan/vulkan.h> as a fallback for the vulkan headers
  if cc.has_header_symbol('vulkan/vulkan_core.h', 'VULKAN_1_2')
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
libexecinfo = cc.find_library('execinfo', required: false)
has_execinfo = cc.has_function('backtrace_symbols', dependencies: libexecinfo, prefix: '#include <execinfo.h>')
conf_internal.set('PL_HAVE_UNWIND', unwind.found())
conf_internal.set('PL_HAVE_EXECINFO', has_execinfo)
if unwind.found()
  build_deps += [unwind, cc.find_library('dl', required : false)]
elif has_execinfo
  build_deps += libexecinfo
endif

# work-arounds for glslang braindeath
glslang = disabler()
glslang_req = get_option('glslang')

if glslang_req.auto() and shaderc.found()

  # we only need one or the other, and shaderc is preferred
  message('Skipping `glslang` because `shaderc` is available')

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
    cxx.find_library('pthread',             required: false)
  ]

  if glslang_deps[0].found() and cc.has_header('glslang/build_info.h')
    glslang = declare_dependency(dependencies: glslang_deps)
  endif

endif

if not (glslang.found() or shaderc.found())
  spirv_msg = 'vulkan support requires either `shaderc` or `glslang`'
  if vulkan_opt.enabled()
    error(spirv_msg)
  elif vulkan_headers.found()
    warning(spirv_msg + ', disabling')
    vulkan_headers = disabler()
    vulkan = disabler()
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
  'dispatch.h',
  'dither.h',
  'dummy.h',
  'filters.h',
  'gpu.h',
  'log.h',
  'renderer.h',
  'shaders/colorspace.h',
  'shaders/custom.h',
  'shaders/deinterlacing.h',
  'shaders/film_grain.h',
  'shaders/icc.h',
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
  'shaders/deinterlacing.c',
  'shaders/film_grain.c',
  'shaders/film_grain_av1.c',
  'shaders/film_grain_h274.c',
  'shaders/icc.c',
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
    'deps':  dependency('lcms2', version: '>=2.9', required: get_option('lcms')),
    'test': 'icc.c',
  }, {
    'name': 'glslang',
    'deps': glslang,
    'srcs': [ 'glsl/glslang.cc',
              'glsl/glslang_resources.c',
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
              'opengl/loader_gl.c',
              'opengl/loader_egl.c',
              'opengl/gpu.c',
              'opengl/gpu_tex.c',
              'opengl/gpu_pass.c',
              'opengl/swapchain.c',
              'opengl/utils.c',
            ],
    'headers': 'opengl.h',
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
  summary(name, deps.found(), section: 'Components', bool_yn: true)

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
  datadir = get_option('prefix') / get_option('datadir')
  sources += custom_target('utils_gen.c',
    input: 'vulkan/utils_gen.py',
    output: 'utils_gen.c',
    command: [python, '@INPUT@', datadir, registry_xml, '@OUTPUT@'],
    env: python_env,
  )

  if vulkan.found() and get_option('vulkan-link')
    defs += '#define PL_HAVE_VK_PROC_ADDR 1\n'
    pc_vars += 'pl_has_vk_proc_addr=1'
  else
    defs += '#undef PL_HAVE_VK_PROC_ADDR\n'
    pc_vars += 'pl_has_vk_proc_addr=0'
  endif
endif

if comps.has('opengl')
  if libdl.found() and get_option('opengl-loader')
    build_deps += libdl
    tests += 'opengl_surfaceless.c'
    defs += '#define PL_HAVE_GL_PROC_ADDR 1\n'
    pc_vars += 'pl_has_gl_proc_addr=1'
    gl_loader = true
  else
    defs += '#undef PL_HAVE_GL_PROC_ADDR\n'
    pc_vars += 'pl_has_gl_proc_addr=0'
    gl_loader = false
  endif

  gl_extensions = [
    'GL_AMD_pinned_memory',
    'GL_ARB_buffer_storage',
    'GL_ARB_compute_shader',
    'GL_ARB_framebuffer_object',
    'GL_ARB_get_program_binary',
    'GL_ARB_invalidate_subdata',
    'GL_ARB_pixel_buffer_object',
    'GL_ARB_shader_image_load_store',
    'GL_ARB_shader_storage_buffer_object',
    'GL_ARB_sync',
    'GL_ARB_texture_float',
    'GL_ARB_texture_gather',
    'GL_ARB_texture_rg',
    'GL_ARB_timer_query',
    'GL_ARB_uniform_buffer_object',
    'GL_ARB_vertex_array_object',
    'GL_EXT_color_buffer_float',
    'GL_EXT_color_buffer_half_float',
    'GL_EXT_texture3D',
    'GL_EXT_texture_format_BGRA8888',
    'GL_EXT_texture_integer',
    'GL_EXT_texture_norm16',
    'GL_EXT_texture_rg',
    'GL_EXT_unpack_subimage',
    'GL_KHR_debug',
    'GL_OES_EGL_image',
    'GL_OES_EGL_image_external',
    'EGL_EXT_image_dma_buf_import',
    'EGL_EXT_image_dma_buf_import_modifiers',
    'EGL_EXT_platform_base',
    'EGL_KHR_debug',
    'EGL_KHR_image_base',
    'EGL_MESA_image_dma_buf_export',
    'EGL_MESA_platform_surfaceless',
  ]

  # Generate GL loader. Adding it to `build_deps` instead of `sources` makes
  # sure it also gets correctly pulled into the test framework
  subdir('include/glad')
  build_deps += declare_dependency(sources: glad)
endif

# Check to see if libplacebo built this way is sane
if not (comps.has('vulkan') or comps.has('opengl') or comps.has('d3d11'))
  warning('Building without any graphics API. libplacebo built this way still ' +
          'has some limited use (e.g. generating GLSL shaders), but most of ' +
          'its functionality will be missing or impaired!')
endif


# Build process
conf_public.set('extra_defs', defs)
subdir('./include/libplacebo') # generate config.h in the right location

sources += configure_file(
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
