dav1d = dependency('dav1d', required: false)
if dav1d.found()
  tdep += dav1d
  tests += 'dav1d.c'
endif

lavu = dependency('libavutil', version: '>=55.74.100', required: false)
lavc = dependency('libavcodec', required: false)
lavf = dependency('libavformat', required: false)
libav_found = lavu.found() and lavc.found() and lavf.found()
if libav_found
  tdep += [lavu, lavc, lavf]
  tests += 'libav.c'
endif

foreach t : tests
  e = executable('test.' + t, t,
      objects: lib.extract_all_objects(recursive: false),
      c_args: [ '-Wno-unused-function' ],
      dependencies: tdep,
  )

  test(t, e)
endforeach

subdir('include')
