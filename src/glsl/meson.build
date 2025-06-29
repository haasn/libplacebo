# shaderc
shaderc = dependency('shaderc', version: '>=2019.1', required: get_option('shaderc'))
components.set('shaderc', shaderc.found())
if shaderc.found()
  build_deps += shaderc
  sources += 'glsl/spirv_shaderc.c'

  # Version check for shaderc is not possible because everything after v2023.8
  # uses this version due to a malformed version line.
  # See https://github.com/google/shaderc/issues/1496
  if shaderc.type_name() == 'internal'
    conf_internal.set('PL_HAVE_SHADERC_VK_1_4', true)
    conf_internal.set('PL_HAVE_SHADERC_VK_1_3', true)
  else
    conf_internal.set('PL_HAVE_SHADERC_VK_1_4',
      cc.has_header_symbol('shaderc/shaderc.h', 'shaderc_env_version_vulkan_1_4', dependencies: shaderc))
    conf_internal.set('PL_HAVE_SHADERC_VK_1_3',
      cc.has_header_symbol('shaderc/shaderc.h', 'shaderc_env_version_vulkan_1_3', dependencies: shaderc))
  endif
endif

# glslang
glslang = disabler()
glslang_req = get_option('glslang')
if glslang_req.auto() and shaderc.found()

  # we only need one or the other, and shaderc is preferred
  message('Skipping `glslang` because `shaderc` is available')

elif not glslang_req.disabled()

  glslang_deps = [
    cxx.find_library('glslang-default-resource-limits', required: false)
  ]

  # meson doesn't respect generator expressions in INTERFACE_LINK_LIBRARIES
  # https://github.com/mesonbuild/meson/issues/8232
  # TODO: Use the following once it's fixed
  # glslang = dependency('glslang', method: 'cmake', modules: ['glslang::SPIRV'])

  prefer_static = get_option('prefer_static')
  found_lib = false
  foreach arg : [[prefer_static, false], [not prefer_static, glslang_req]]
    static   = arg[0]
    required = arg[1]

    spirv = cxx.find_library('SPIRV', required: required, static: static)

    if not spirv.found()
      continue
    endif

    glslang_deps += spirv

    # Glslang 15.0.0 moved some code around, add also linking to glslang, while
    # this is not needed for older versions, it will still work.
    glslang_deps += cxx.find_library('glslang', required: required, static: static)

    if static
      glslang_deps += [
        # Always required for static linking
        cxx.find_library('MachineIndependent', required: false, static: true),
        cxx.find_library('OSDependent',        required: false, static: true),
        cxx.find_library('OGLCompiler',        required: false, static: true),
        cxx.find_library('GenericCodeGen',     required: false, static: true),
        # SPIRV-Tools are required only if optimizer is enabled in glslang build
        cxx.find_library('SPIRV-Tools',        required: false, static: true),
        cxx.find_library('SPIRV-Tools-opt',    required: false, static: true),
      ]
    endif

    found_lib = true
    break
  endforeach

  if found_lib and cc.has_header('glslang/build_info.h')
    glslang = declare_dependency(dependencies: glslang_deps)
  endif

endif

components.set('glslang', glslang.found())
if glslang.found()
  build_deps += glslang
  sources += [
    'glsl/glslang.cc',
    'glsl/glslang_resources.c',
    'glsl/spirv_glslang.c',
  ]
endif
