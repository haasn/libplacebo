opengl_build = get_option('opengl')
opengl_link = get_option('gl-proc-addr')

if host_machine.system() == 'windows' or host_machine.system().endswith('bsd') or \
   host_machine.system() == 'dragonfly'
    libdl = declare_dependency()
else
    libdl = cc.find_library('dl', required : opengl_link)
endif
opengl_link = opengl_link.require(libdl.found())
components.set('opengl', opengl_build.allowed())
components.set('gl-proc-addr', opengl_link.allowed())

if opengl_build.allowed()
  sources += [
    'opengl/context.c',
    'opengl/formats.c',
    'opengl/loader_gl.c',
    'opengl/loader_egl.c',
    'opengl/gpu.c',
    'opengl/gpu_tex.c',
    'opengl/gpu_pass.c',
    'opengl/swapchain.c',
    'opengl/utils.c',
  ]

  if opengl_link.allowed()
    build_deps += libdl
    tests += 'opengl_surfaceless.c'
  endif

  gl_extensions = [
    'GL_AMD_pinned_memory',
    'GL_ARB_buffer_storage',
    'GL_ARB_compute_shader',
    'GL_ARB_framebuffer_object',
    'GL_ARB_get_program_binary',
    'GL_ARB_invalidate_subdata',
    'GL_ARB_pixel_buffer_object',
    'GL_ARB_program_interface_query',
    'GL_ARB_shader_image_load_store',
    'GL_ARB_shader_storage_buffer_object',
    'GL_ARB_sync',
    'GL_ARB_texture_float',
    'GL_ARB_texture_gather',
    'GL_ARB_texture_rg',
    'GL_ARB_timer_query',
    'GL_ARB_uniform_buffer_object',
    'GL_ARB_vertex_array_object',
    'GL_ARB_half_float_pixel',
    'GL_EXT_EGL_image_storage',
    'GL_EXT_color_buffer_float',
    'GL_EXT_texture3D',
    'GL_EXT_texture_format_BGRA8888',
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

  # Generate GL loader
  subdir('include/glad')
else
  glad_dep = []
  sources += 'opengl/stubs.c'
endif
