shader_sources = [
  'colorspace.c',
  'custom.c',
  'custom_mpv.c',
  'deinterlacing.c',
  'dithering.c',
  'film_grain.c',
  'film_grain_av1.c',
  'film_grain_h274.c',
  'icc.c',
  'lut.c',
  'sampling.c',
]

foreach s : shader_sources
  sources += custom_target(s,
    command: glsl_preproc,
    depend_files: glsl_deps,
    env: python_env,
    input: s,
    output: s,
  )
endforeach
