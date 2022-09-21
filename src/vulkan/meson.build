vulkan_build = get_option('vulkan')
vulkan_link = get_option('vk-proc-addr')
vulkan_loader = dependency('vulkan', version: '>=1.2.0', required: vulkan_link)
vulkan_headers = disabler()

if vulkan_loader.found()
  vulkan_headers = vulkan_loader.partial_dependency(includes: true, compile_args: true)
else
  # Probe for <vulkan/vulkan.h> as a fallback for the vulkan headers
  if cc.has_header_symbol('vulkan/vulkan_core.h', 'VULKAN_1_2')
    vulkan_headers = declare_dependency()
  endif
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

  registry_xml = get_option('vulkan-registry')
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
