vulkan_build = get_option('vulkan')
vulkan_link = get_option('vk-proc-addr')
vulkan_loader = dependency('vulkan', required: false)
vulkan_headers = vulkan_loader.partial_dependency(includes: true, compile_args: true)
registry_xml = get_option('vulkan-registry')

# Prefer our Vulkan headers for portability
vulkan_headers_dir = thirdparty/'Vulkan-Headers'
if fs.is_dir(vulkan_headers_dir/'include')
  vulkan_headers = declare_dependency()
  vulkan_headers_inc = include_directories('../../3rdparty/Vulkan-Headers/include')
  # Force the use of this vk.xml because it has to be in sync with the headers
  registry_xml = vulkan_headers_dir/'registry/vk.xml'
endif

vulkan_build = vulkan_build.require(
  cc.has_header_symbol('vulkan/vulkan_core.h', 'VK_VERSION_1_3',
                       include_directories: vulkan_headers_inc,
                       dependencies: vulkan_headers),
  error_message: 'vulkan.h was not found on the system, nor inside ' +
                 '`3rdparty/Vulkan-Headers`. Please run `git submodule update --init` ' +
                 'followed by `meson --wipe`.')
components.set('vulkan', vulkan_build.allowed())

vulkan_link = vulkan_link.require(vulkan_loader.found() and vulkan_build.allowed())
components.set('vk-proc-addr', vulkan_link.allowed())

build_deps += vulkan_headers

if vulkan_build.allowed()
  sources += [
    'vulkan/command.c',
    'vulkan/context.c',
    'vulkan/formats.c',
    'vulkan/gpu.c',
    'vulkan/gpu_buf.c',
    'vulkan/gpu_tex.c',
    'vulkan/gpu_pass.c',
    'vulkan/malloc.c',
    'vulkan/swapchain.c',
    'vulkan/utils.c',
  ]

  datadir = get_option('prefix') / get_option('datadir')
  sources += custom_target('utils_gen.c',
    input: 'utils_gen.py',
    output: 'utils_gen.c',
    command: [python, '@INPUT@', datadir, registry_xml, '@OUTPUT@'],
    env: python_env,
  )

  if vulkan_link.allowed()
    build_deps += vulkan_loader
    tests += 'vulkan.c'
  endif
else
  sources += 'vulkan/stubs.c'
endif
