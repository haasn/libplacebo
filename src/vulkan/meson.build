vulkan_opt = get_option('vulkan')
vulkan = dependency('vulkan', version: '>=1.2.0', required: vulkan_opt)

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

if not (glslang.found() or shaderc.found())
  spirv_msg = 'Vulkan support requires either `shaderc` or `glslang`'
  if vulkan_opt.enabled()
    error(spirv_msg)
  elif vulkan_headers.found()
    warning(spirv_msg + ', disabling')
    vulkan_headers = disabler()
    vulkan = disabler()
  endif
endif

components.set('vulkan', vulkan_headers.found())
if vulkan_headers.found()
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

  vk_loader = vulkan.found() and get_option('vulkan-link')
  components.set('vk-proc-addr', vk_loader)
endif

