glad_check = run_command([ python, '-c', 'import glad; print(glad.__version__)' ],
  env: python_env,
  capture: true,
  check: false,
)

glad_ver = glad_check.returncode() == 0 ? glad_check.stdout().strip() : 'none'
glad_req = '>= 2.0'

if not glad_ver.version_compare(glad_req)
  error(f'glad (required: @glad_req@, found: @glad_ver@) was not found in ' +
        'PYTHONPATH or `3rdparty`. Please run `git submodule update --init` ' +
        'followed by `meson --wipe`.')
endif

glad = custom_target('gl.h',
  output: 'gl.h',
  env: python_env,
  command: [
    python, '-m', 'glad', '--out-path=@OUTDIR@/../../',
    '--reproducible', '--merge', '--api=gl:core,gles2,egl',
    '--extensions=' + ','.join(gl_extensions), 'c', '--header-only', '--mx'
  ] + (opengl_link.allowed() ? ['--loader'] : [])
)
