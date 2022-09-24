vulkan_build = get_option('vulkan')
vulkan_link = get_option('vk-proc-addr')
vulkan_loader = dependency('vulkan', version: '>=1.2.0', required: vulkan_link)
vulkan_headers = disabler()
registry_xml = get_option('vulkan-registry')

if vulkan_loader.found()
  vulkan_headers = vulkan_loader.partial_dependency(includes: true, compile_args: true)
# Probe for <vulkan/vulkan.h> as a fallback for the vulkan headers
elif cc.has_header_symbol('vulkan/vulkan_core.h', 'VK_VERSION_1_2')
  vulkan_headers = declare_dependency()
elif fs.is_dir(thirdparty/'Vulkan-Headers')
  vulkan_inc = include_directories('../../3rdparty/Vulkan-Headers/include')
  vulkan_headers = declare_dependency(include_directories: vulkan_inc)
  # Force the use of this vk.xml because it has to be in sync with the headers
  registry_xml = thirdparty/'Vulkan-Headers/registry/vk.xml'
endif

if vulkan_build.enabled() and not vulkan_headers.found()
  error('vulkan.h was not found on the system, nor inside ' +
        '`3rdparty/Vulkan-Headers`. Please run `git submodule update --init` ' +
        'followed by `meson --wipe`.')
endif

vulkan_build = vulkan_build.require(vulkan_headers.found())
vulkan_link = vulkan_link.require(vulkan_loader.found())
components.set('vulkan', vulkan_build.allowed())
components.set('vk-proc-addr', vulkan_link.allowed())

if vulkan_build.allowed()
  build_deps += vulkan_headers
  headers += 'vulkan.h'
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
endif
