project('xtalloc', 'c')

sources = [
  'ta.c',
  'ta_utils.c',
  'xtalloc.c',
]

deps = [
  dependency('threads'),
]

lib = static_library('xtalloc',
  sources,
  dependencies: deps,
  install: false,
)

dep = declare_dependency(
  include_directories: include_directories('include'),
  link_with: lib
)
