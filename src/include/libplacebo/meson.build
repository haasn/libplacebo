sources += configure_file(
  input: 'config.h.in',
  output: 'config.h',
  install_dir: get_option('includedir') / meson.project_name(),
  configuration: conf_public,
)
