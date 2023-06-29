### Common dependencies
unwind = dependency('libunwind', required: get_option('unwind'))
libexecinfo = cc.find_library('execinfo', required: false)
has_execinfo = cc.has_function('backtrace_symbols', dependencies: libexecinfo, prefix: '#include <execinfo.h>')
dbghelp = cc.check_header('dbghelp.h', prefix: '#include <windows.h>')
conf_internal.set('PL_HAVE_DBGHELP', dbghelp)
conf_internal.set('PL_HAVE_UNWIND', unwind.found())
conf_internal.set('PL_HAVE_EXECINFO', has_execinfo)
conf_internal.set('PL_STRIP_SHADERS', not get_option('debug'))
if dbghelp
  build_deps += cc.find_library('shlwapi', required: true)
elif unwind.found()
  build_deps += [unwind, cc.find_library('dl', required : false)]
elif has_execinfo
  build_deps += libexecinfo
endif

# Looks like meson in certain configuration returns ' ' instead of empty string
mingw32 = cc.get_define('__MINGW32__').strip()
if host_machine.system() == 'windows' and mingw32 != ''
  # MinGW-w64 math functions are significantly slower than the UCRT ones.
  # In particular powf is over 7 times slower than UCRT counterpart.
  # MinGW-w64 explicitly excludes some math functions from their ucrtbase def
  # file and replaces with own versions. To workaround the issue, generate the
  # import library and link it with UCRT versions of math functions.
  dlltool = find_program('llvm-dlltool', 'dlltool')
  ucrt_math = custom_target('ucrt_math.lib',
                            output : ['ucrt_math.lib'],
                            input : 'ucrt_math.def',
                            command : [dlltool, '-d', '@INPUT@', '-l', '@OUTPUT@'])
  build_deps += declare_dependency(link_with: ucrt_math)
  # MinGW-w64 inlines functions like powf, rewriting them to pow. We want to use
  # the powf specialization from UCRT, so disable inlining.
  add_project_arguments(['-D__CRT__NO_INLINE'], language: ['c', 'cpp'])
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


### Common source files
headers = [
  'colorspace.h',
  'common.h',
  'd3d11.h',
  'dispatch.h',
  'dither.h',
  'dummy.h',
  'filters.h',
  'gamut_mapping.h',
  'gpu.h',
  'log.h',
  'opengl.h',
  'renderer.h',
  'shaders/colorspace.h',
  'shaders/custom.h',
  'shaders/deinterlacing.h',
  'shaders/dithering.h',
  'shaders/film_grain.h',
  'shaders/icc.h',
  'shaders/lut.h',
  'shaders/sampling.h',
  'shaders.h',
  'swapchain.h',
  'tone_mapping.h',
  'utils/dav1d.h',
  'utils/dav1d_internal.h',
  'utils/dolbyvision.h',
  'utils/frame_queue.h',
  'utils/libav.h',
  'utils/libav_internal.h',
  'utils/upload.h',
  'vulkan.h',
]

sources = [
  'colorspace.c',
  'common.c',
  'dither.c',
  'dispatch.c',
  'dummy.c',
  'filters.c',
  'format.c',
  'gamut_mapping.c',
  'glsl/spirv.c',
  'gpu.c',
  'gpu/utils.c',
  'log.c',
  'pl_alloc.c',
  'pl_string.c',
  'renderer.c',
  'siphash.c',
  'shaders.c',
  'shaders/colorspace.c',
  'shaders/custom.c',
  'shaders/custom/mpv.c',
  'shaders/deinterlacing.c',
  'shaders/dithering.c',
  'shaders/film_grain.c',
  'shaders/film_grain_av1.c',
  'shaders/film_grain_h274.c',
  'shaders/icc.c',
  'shaders/lut.c',
  'shaders/sampling.c',
  'swapchain.c',
  'tone_mapping.c',
  'utils/dolbyvision.c',
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

components = configuration_data()


### Optional dependencies / components
subdir('glsl')
subdir('d3d11')
subdir('opengl')
subdir('vulkan')

lcms = dependency('lcms2', version: '>=2.9', required: get_option('lcms'))
components.set('lcms', lcms.found())
if lcms.found()
  build_deps += lcms
  tests += 'icc.c'
endif

# Check to see if libplacebo built this way is sane
if not (components.get('vulkan') or components.get('opengl') or components.get('d3d11'))
  warning('Building without any graphics API. libplacebo built this way still ' +
          'has some limited use (e.g. generating GLSL shaders), but most of ' +
          'its functionality will be missing or impaired!')
endif

has_spirv = components.get('shaderc') or components.get('glslang')
needs_spirv = components.get('vulkan') or components.get('d3d11')
if needs_spirv and not has_spirv
  warning('Building without any GLSL compiler (shaderc, glslang), but with ' +
          'APIs required that require one (vulkan, d3d11). This build is very ' +
          'likely to be very limited in functionality!')
endif

dovi = get_option('dovi')
components.set('dovi', dovi.allowed())

libdovi = dependency('dovi', version: '>=1.6.7', required: get_option('libdovi').require(dovi.allowed()))
components.set('libdovi', libdovi.found())
if libdovi.found()
  build_deps += libdovi
endif

# Generate configuration files
defs = ''
pc_vars = []

foreach comp : components.keys()
  found = components.get(comp)
  varname = comp.underscorify().to_upper()
  summary(comp, found, section: 'Optional features', bool_yn: true)
  defs += (found ? '#define PL_HAVE_@0@ 1\n' : '#undef PL_HAVE_@0@\n').format(varname)
  pc_vars += 'pl_has_@0@=@1@'.format(varname.to_lower(), found ? 1 : 0)
endforeach

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


### Main library build process
inc = include_directories('./include')
lib = library('placebo', sources,
  c_args: '-DPL_EXPORT',
  install: true,
  dependencies: build_deps + glad_dep,
  soversion: apiver,
  include_directories: inc,
  link_args: link_args,
  gnu_symbol_visibility: 'hidden',
  name_prefix: 'lib'
)

libplacebo = declare_dependency(
  include_directories: inc,
  compile_args: get_option('default_library') == 'static' ? ['-DPL_STATIC'] : [],
  link_with: lib,
  variables: pc_vars,
)


### Install process
proj_name = meson.project_name()
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

extra_cflags = []
if get_option('default_library') == 'static'
  extra_cflags = ['-DPL_STATIC']
elif get_option('default_library') == 'both'
  # meson doesn't support Cflags.private, insert it forcefully...
  extra_cflags = ['\nCflags.private:', '-DPL_STATIC']
endif

pkg = import('pkgconfig')
pkg.generate(
  name: proj_name,
  description: 'Reusable library for GPU-accelerated video/image rendering',
  libraries: lib,
  version: version,
  variables: pc_vars,
  extra_cflags: extra_cflags,
)


### Testing
tdep_static = declare_dependency(
    dependencies: build_deps,
    include_directories: [ inc, include_directories('.') ],
    compile_args: '-DPL_STATIC'
    # TODO: Define objects here once Meson 1.1.0 is ok to use
    # objects: lib.extract_all_objects(recursive: false)
  )

tdep_shared = declare_dependency(
    include_directories: [ inc, include_directories('.') ],
    compile_args: get_option('default_library') == 'static' ? ['-DPL_STATIC'] : [],
    link_with: lib,
  )

if get_option('tests')
  subdir('tests')
endif

if get_option('bench')
  if not components.get('vk-proc-addr')
    error('Compiling the benchmark suite requires vulkan support!')
  endif

  bench = executable('bench', 'tests/bench.c', dependencies: [tdep_shared, vulkan_headers])
  test('benchmark', bench, is_parallel: false, timeout: 600)
endif

if get_option('fuzz')
  foreach f : fuzzers
    executable('fuzz.' + f, 'tests/fuzz/' + f,
        objects: lib.extract_all_objects(recursive: false),
        dependencies: tdep_static,
    )
  endforeach
endif

pl_thread = declare_dependency(
  include_directories: include_directories('.'),
  dependencies: threads,
)
