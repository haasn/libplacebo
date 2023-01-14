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

else

  glslang_deps = [
    cxx.find_library('glslang',             required: glslang_req),
    cxx.find_library('MachineIndependent',  required: false),
    cxx.find_library('OSDependent',         required: false),
    cxx.find_library('HLSL',                required: glslang_req),
    cxx.find_library('OGLCompiler',         required: false),
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

components.set('glslang', glslang.found())
if glslang.found()
  build_deps += glslang
  sources += [
    'glsl/glslang.cc',
    'glsl/glslang_resources.c',
    'glsl/spirv_glslang.c',
  ]
endif

