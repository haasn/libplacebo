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

nuklear_inc = include_directories('./3rdparty/nuklear')
nuklear_found = cc.has_header('nuklear.h', include_directories: nuklear_inc)
nuklear = declare_dependency(
  sources: 'ui.c',
  include_directories: nuklear_inc,
  compile_args: ['-Wno-missing-prototypes', '-DHAVE_UI'],
  dependencies: libm,
)

apis = {}

# Enable all supported combinations of API and windowing system
if glfw.found()
  if vulkan.found()
    apis += {'glfw-vk': declare_dependency(
      sources: 'window_glfw.c',
      compile_args: '-DUSE_VK',
      dependencies: [libplacebo, vulkan, glfw],
    )}
  endif

  if opengl.found()
    apis += {'glfw-gl': declare_dependency(
      sources: 'window_glfw.c',
      compile_args: '-DUSE_GL',
      dependencies: [libplacebo, glfw],
    )}
  endif
endif

if sdl.found()
  if vulkan.found()
    apis += {'sdl-vk': declare_dependency(
      sources: 'window_sdl.c',
      compile_args: '-DUSE_VK',
      dependencies: [libplacebo, vulkan, sdl],
    )}
  endif

  if opengl.found()
    apis += {'sdl-gl': declare_dependency(
      sources: 'window_sdl.c',
      compile_args: '-DUSE_GL',
      dependencies: [libplacebo, sdl],
    )}
  endif
endif

if apis.keys().length() == 0
  warning('Demos enabled but no supported combination of windowing system ' +
          'and graphical APIs was found. Demo programs require either GLFW or ' +
          'SDL and either Vulkan or OpenGL.')
  subdir_done()
endif

if not nuklear_found
  warning('Nuklear was not found in `demos/3rdparty`. Please run ' +
          '`git submodule update --init` followed by `meson --wipe`. ' +
          'Disabling nuklear based demos.')
endif

# Compile demos for each combination
foreach api, dep : apis

  executable('colors-' + api, 'colors.c',
    dependencies: [ dep, libm ],
  )

  if sdl_image.found()
    executable('image-' + api, 'image.c',
      dependencies: [ dep, sdl_image ],
    )
  endif

  if ffmpeg_found
    executable('plplay-' + api, 'plplay.c',
      dependencies: [ dep ] + ffmpeg_deps,
    )
  endif

  if nuklear_found
    executable('nuklear-' + api, 'nuklear.c',
      dependencies: [ dep, nuklear ],
    )
  endif
endforeach

# Headless video filtering demo
if vulkan.found()
  executable('video-filtering', 'video-filtering.c',
    dependencies: libplacebo,
    c_args: '-O2',
  )
endif
