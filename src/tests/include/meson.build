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
    dependencies: [tdep_static, lavu, lavc, lavf],
    include_directories: [inc, vulkan_headers_inc],
    implicit_include_directories: false,
    c_args: ['-Wall', '-Wextra', '-Wpedantic'],
    cpp_args: ['-Wall', '-Wextra', '-Wpedantic'],
)
