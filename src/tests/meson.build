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

include_tmpl_langs = ['c', 'cpp']

# Ensure all headers compile

test_include_sources = []
foreach h : headers

  if (h.contains('internal') or
      h.contains('dav1d') and not dav1d.found() or
      h.contains('libav') and not libav_found or
      h.contains('d3d11') and not d3d11_header)
    continue
  endif

  foreach lang : include_tmpl_langs

    test_include_sources += configure_file(
        input: 'include_tmpl.' + lang,
        output: 'include_@0@.@1@'.format(h.underscorify(), lang),
        configuration: {
          'header': h
        },
    )

  endforeach

endforeach

static_library('test_include', test_include_sources,
    dependencies: tdep,
    implicit_include_directories: false
)
