glfw = dependency('glfw3', required: false)
sdl = dependency('sdl2', required: false)
sdl_image = dependency('SDL2_image', required: false)

ffmpeg_deps = [
  dependency('libavcodec', required: false),
  dependency('libavformat', required: false),
  dependency('libavutil', required: false),
]

ffmpeg_found = true
foreach dep : ffmpeg_deps
  ffmpeg_found = ffmpeg_found and dep.found()
endforeach

nuklear = disabler()
nuklear_inc = include_directories('./3rdparty/nuklear')
if cc.has_header('nuklear.h', include_directories: nuklear_inc)
  nuklear_lib = static_library('nuklear',
    include_directories: nuklear_inc,
    c_args: ['-O2', '-Wno-missing-prototypes'],
    dependencies: [ libplacebo, libm ],
    sources: 'ui.c',
  )

  nuklear = declare_dependency(
    include_directories: nuklear_inc,
    link_with: nuklear_lib,
  )
else
  warning('Nuklear was not found in `demos/3rdparty`. Please run ' +
          '`git submodule update --init` followed by `meson --wipe`.')
endif

conf_demos = configuration_data()
conf_demos.set('HAVE_NUKLEAR', nuklear.found())
conf_demos.set('HAVE_EGL', cc.check_header('EGL/egl.h', required: false))

apis = []

# Enable all supported combinations of API and windowing system
if glfw.found()
  if components.get('vulkan')
    conf_demos.set('HAVE_GLFW_VULKAN', true)
    apis += static_library('glfw-vk',
      dependencies: [libplacebo, libm, glfw, vulkan_headers],
      sources: 'window_glfw.c',
      c_args: '-DUSE_VK',
    )
  endif

  if components.get('opengl')
    conf_demos.set('HAVE_GLFW_OPENGL', true)
    apis += static_library('glfw-gl',
      dependencies: [libplacebo, glfw],
      sources: 'window_glfw.c',
      c_args: '-DUSE_GL',
    )
  endif

  if components.get('d3d11')
    conf_demos.set('HAVE_GLFW_D3D11', true)
    apis += static_library('glfw-d3d11',
      dependencies: [libplacebo, glfw],
      sources: 'window_glfw.c',
      c_args: '-DUSE_D3D11',
    )
  endif
endif

if sdl.found()
  if components.get('vulkan')
    conf_demos.set('HAVE_SDL_VULKAN', true)
    apis += static_library('sdl-vk',
      dependencies: [libplacebo, sdl, vulkan_headers],
      sources: 'window_sdl.c',
      c_args: '-DUSE_VK',
    )
  endif

  if components.get('opengl')
    conf_demos.set('HAVE_SDL_OPENGL', true)
    apis += static_library('sdl-gl',
      dependencies: [libplacebo, sdl],
      sources: 'window_sdl.c',
      c_args: '-DUSE_GL',
    )
  endif
endif

configure_file(
  output: 'config_demos.h',
  configuration: conf_demos,
)

if apis.length() == 0
  warning('Demos enabled but no supported combination of windowing system ' +
          'and graphical APIs was found. Demo programs require either GLFW or ' +
          'SDL and either Vulkan or OpenGL to function.')
else

  dep = declare_dependency(
    dependencies: [ libplacebo, build_deps ],
    sources: ['window.c', 'utils.c'],
    link_with: apis,
  )

  # Graphical demo programs
  executable('colors', 'colors.c',
    dependencies: [ dep, libm ],
  )

  if sdl_image.found()
    executable('sdlimage', 'sdlimage.c',
      dependencies: [ dep, libm, sdl_image ],
    )
  endif

  if ffmpeg_found
    plplay_deps = [ dep, pl_thread ] + ffmpeg_deps
    if nuklear.found()
      plplay_deps += nuklear
    endif
    executable('plplay', 'plplay.c',
      dependencies: plplay_deps,
      install: true,
    )
  endif

endif

# Headless video filtering demo
if components.get('vk-proc-addr')
  executable('video-filtering', 'video-filtering.c',
    dependencies: [ libplacebo, vulkan_loader ],
    c_args: '-O2',
  )
endif
