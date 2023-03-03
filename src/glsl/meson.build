# shaderc
shaderc = dependency('shaderc', version: '>=2019.1', required: get_option('shaderc'))
components.set('shaderc', shaderc.found())
if shaderc.found()
  build_deps += shaderc
  sources += 'glsl/spirv_shaderc.c'
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

  glslang_shared = cxx.find_library('glslang', required: false, static: false)
  if glslang_shared.found()
    glslang_deps += glslang_shared
  else
    glslang_deps += [
      cxx.find_library('glslang',            required: glslang_req, static: true),
      cxx.find_library('MachineIndependent', required: glslang_req, static: true),
      cxx.find_library('OSDependent',        required: glslang_req, static: true),
      cxx.find_library('OGLCompiler',        required: glslang_req, static: true),
      cxx.find_library('GenericCodeGen',     required: glslang_req, static: true),
      cxx.find_library('SPVRemapper',        required: glslang_req, static: true),
    ]
  endif

  spirv_shared = cxx.find_library('SPIRV', required: false, static: false)
  if spirv_shared.found()
    glslang_deps += spirv_shared
  else
    glslang_deps += [
      cxx.find_library('SPIRV',           required: glslang_req, static: true),
      cxx.find_library('SPIRV-Tools',     required: glslang_req, static: true),
      cxx.find_library('SPIRV-Tools-opt', required: glslang_req, static: true),
    ]
  endif

  if glslang_deps[1].found() and cc.has_header('glslang/build_info.h')
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

