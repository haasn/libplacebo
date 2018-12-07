project('bstr', 'c')

sources = [
  'bstr.c',
  'format.c',
  'siphash.c',
]

deps = [
  subproject('xtalloc').get_variable('dep'),
]

inc = include_directories('include')
lib = static_library('bstr',
  sources,
  include_directories: inc,
  dependencies: deps,
  install: false,
)

dep = declare_dependency(
  include_directories: inc,
  link_with: lib,
)
