d3d11 = get_option('d3d11')
d3d11_header = cc.check_header('d3d11.h', required: false) # needed publicly
d3d11_headers_extra = [ # needed internally
  cc.check_header('d3d11_4.h', required: d3d11),
  cc.check_header('dxgi1_6.h', required: d3d11),
]
d3d11_deps = [
  dependency('spirv-cross-c-shared', version: '>=0.29.0', required: d3d11),
  cc.find_library('version', required: d3d11),
]

d3d11 = d3d11.require(d3d11_header)
foreach h : d3d11_headers_extra
  d3d11 = d3d11.require(h)
endforeach
foreach d : d3d11_deps
  d3d11 = d3d11.require(d.found())
endforeach

components.set('d3d11', d3d11.allowed())
if d3d11.allowed()
  add_project_arguments(['-DCOBJMACROS'], language: 'c')
  build_deps += declare_dependency(dependencies: d3d11_deps)
  tests += 'd3d11.c'
  sources += [
    'd3d11/context.c',
    'd3d11/formats.c',
    'd3d11/gpu.c',
    'd3d11/gpu_buf.c',
    'd3d11/gpu_tex.c',
    'd3d11/gpu_pass.c',
    'd3d11/swapchain.c',
    'd3d11/utils.c',
  ]
elif d3d11_header
  sources += 'd3d11/stubs.c'
endif
