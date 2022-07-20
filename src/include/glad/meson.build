glad = custom_target('gl.h',
  output: 'gl.h',
  env: python_env,
  command: [
    python, '-m', 'glad', '--out-path=@OUTDIR@/../../',
    '--reproducible', '--merge', '--api=gl:core,gles2,egl',
    '--extensions=' + ','.join(gl_extensions),
    'c', '--header-only', '--mx', '--loader',
  ]
)
