ts = []

foreach t : tests
  deps = [tdep_static]
  if t == 'opengl_surfaceless.c'
    deps += glad_dep
  endif
  # TODO: Define objects in tdep_static once Meson 1.1.0 is ok to use
  ts += { 'source': t,
          'deps': deps,
          'objects': lib.extract_all_objects(recursive: false) }
endforeach

dav1d = dependency('dav1d', required: false)
if dav1d.found()
  ts += { 'source': 'dav1d.c', 'deps': [dav1d, tdep_shared] }
endif

lavu = dependency('libavutil', version: '>=55.74.100', required: false)
lavc = dependency('libavcodec', required: false)
lavf = dependency('libavformat', required: false)
libav_found = lavu.found() and lavc.found() and lavf.found()
if libav_found
  ts += { 'source': 'libav.c', 'deps': [lavu, lavc, lavf, tdep_shared] }
endif

foreach t : ts
  e = executable('test.' + t['source'], t['source'],
      objects: t.get('objects', []),
      c_args: [ '-Wno-unused-function' ],
      dependencies: t.get('deps', []),
      link_args: link_args,
      link_depends: link_depends,
  )

  test(t['source'], e)
endforeach

subdir('include')
