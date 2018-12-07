project('xtalloc', 'c')

sources = [
  'ta.c',
  'ta_utils.c',
  'xtalloc.c',
]

deps = [
  dependency('threads'),
]

inc = include_directories('include')
lib = static_library('xtalloc',
  sources,
  include_directories: inc,
  dependencies: deps,
  install: false,
)

dep = declare_dependency(
  include_directories: inc,
  link_with: lib
)
