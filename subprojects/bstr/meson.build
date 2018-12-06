project('bstr', 'c')

sources = [
  'bstr.c',
  'format.c',
  'siphash.c',
]

deps = [
  subproject('xtalloc').get_variable('dep'),
]

lib = static_library('bstr',
  sources,
  dependencies: deps,
  install: false,
)

dep = declare_dependency(
  include_directories: include_directories('include'),
  link_with: lib
)
