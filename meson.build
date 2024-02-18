project('libplacebo', ['c', 'cpp'],
  license: 'LGPL2.1+',
  default_options: [
    'buildtype=debugoptimized',
    'warning_level=2',
    'c_std=c11',
    'cpp_std=c++20',
  ],
  meson_version: '>=0.63',
  version: '@0@.@1@.@2@'.format(
    # Major version
    7,
    # API version
    {
      '344': 'add PL_ALPHA_NONE',
      '343': 'add pl_map_avdovi_metadata and deprecate pl_frame_map_avdovi_metadata',
      '342': 'add pl_cache_signature',
      '341': 're-add pl_filter_function_{bicubic,bcspline,catmull_rom,mitchell,robidoux,robidouxsharp} as deprecated',
      '340': 'add pl_queue_params.drift_compensation, PL_QUEUE_DEFAULTS and pl_queue_pts_offset',
      '339': 'add pl_peak_detect_params.black_cutoff',
      '338': 'split pl_filter_nearest into pl_filter_nearest and pl_filter_box',
      '337': 'fix PL_FILTER_DOWNSCALING constant',
      '336': 'deprecate pl_filter.radius_cutoff in favor of pl_filter.radius',
      '335': 'remove {pl_render_params,pl_sample_filter_params}.{lut_entries,polar_cutoff}',
      '334': 'add pl_tex_transfer_params.no_import',
      '333': 'add pl_shader_sample_{hermite,gaussian}',
      '332': 'add pl_filter_function_hermite and pl_filter_hermite',
      '331': 'add pl_filter_function_cubic and remove bcspline family of filter functions',
      '330': 'add pl_frames_infer(_mix)',
      '329': 'add pl_frame_mix_current and pl_frame_mix_nearest',
      '328': 'remove pl_render_params.ignore_icc_profiles',
      '327': 'remove pl_render_params.icc_params',
      '326': 'add pl_frame.icc',
      '325': 'add pl_icc_update',
      '324': 'add pl_render_params.correct_subpixel_offsets',
      '323': 'deprecate pl_{dispatch,renderer}_{save,load}',
      '322': 'remove pl_pass_params.cached_program(_len)',
      '321': 'deprecate pl_icc_params.cache_{save,load,priv}',
      '320': 'add pl_gpu_set_cache',
      '319': 'add <libplacebo/cache.h>',
      '318': 'add pl_filter_ewa_lanczossharp and pl_filter_ewa_lanczos4sharpest',
      '317': 'add pl_filter_config.antiring',
      '316': 'remove pl_filter_config.filter_scale',
      '315': 'remove pl_tone_map_auto',
      '314': 'add pl_renderer_get_hdr_metadata',
      '313': 'remove pl_peak_detect_params.minimum_peak',
      '312': 'add pl_gamut_map_constants.perceptual_strength',
      '311': 'add pl_tone_map_constants, pl_tone_map_params.constants and pl_color_map_params.tone_constants',
      '310': 'add pl_gamut_map_constants, pl_gamut_map_params.constants and pl_color_map_params.gamut_constants',
      '309': 'add <libplacebo/options.h>',
      '308': 'add pl_hook_par.names',
      '307': 'add pl_filter.radius_zero',
      '306': 'add pl_filter_functions and pl_filter_configs',
      '305': 'add pl_filter_function.opaque and move pl_filter_oversample to filters.h',
      '304': 'add pl_filter_config.allowed/recommended',
      '303': 'refactor pl_filter_config and pl_filter_function',
      '302': 'change type of pl_icc_params.size_r/g/b to int',
      '301': 'add PL_COLOR_ADJUSTMENT_NEUTRAL and pl_color_adjustment()',
      '300': 'add pl_color_map_params.gamut_expansion',
      '299': 'add pl_primaries_compatible and pl_primaries_clip',
      '298': 'add pl_gamut_map_softclip',
      '297': 'add pl_tone_map_linear_light',
      '296': 'add pl_queue_estimate_fps/vps, pl_queue_num_frames and pl_queue_peek',
      '295': 'change pl_source_frame.pts and pl_queue_params.pts to double',
      '294': 'add pl_vulkan_swapchain_params.disable_10bit_sdr',
      '293': 'add pl_test_pixfmt_caps',
      '292': 'add pl_peak_detect_high_quality_params and pl_color_map_high_quality_params',
      '291': 'add PL_COLOR_HDR_BLACK, PL_COLOR_SDR_CONTRAST and PL_COLOR_HLG_PEAK',
      '290': 'remove pl_color_map_params.hybrid_mix',
      '289': 'remove pl_gamut_map_params.chroma_margin',
      '288': 'add pl_color_map_params.lut3d_tricubic',
      '287': 'add pl_transform2x2_bounds',
      '286': 'add PL_RENDER_ERR_CONTRAST_RECOVERY',
      '285': 'add pl_color_map_params.contrast_recovery/smoothness, ' +
             'pl_color_map_args.feature_map and pl_shader_extract_features',
      '284': 'add pl_color_map_args and pl_shader_color_map_ex',
      '283': 'add pl_render_params.distort_params',
      '282': 'add PL_HOOK_PRE_OUTPUT',
      '281': 'add pl_matrix2x2_rotation',
      '280': 'add pl_distortion_params and pl_shader_distort',
      '279': 'add pl_matrix2x2_scale/invert and pl_transform2x2_scale/invert',
      '278': 'switch pl_vulkan.(un)lock_queue to uint32_t',
      '277': 'add pl_render_params.corner_rounding',
      '276': 'add pl_get_mapped_avframe',
      '275': 'add pl_vulkan_params.extra_queues',
      '274': 'drop minimum vulkan version to 1.2',
      '273': 'add pl_vulkan_required_features and refactor pl_vulkan_recommended_features',
      '272': 'require vulkan version 1.3 minimum',
      '271': 'deprecate pl_vulkan.queues',
      '270': 'add pl_color_map_params.visualize_hue/theta',
      '269': 'refactor pl_color_map_params gamut mapping settings',
      '268': 'add <libplacebo/gamut_mapping.h>',
      '267': 'add pl_ipt_lms2rgb/rgb2lms and pl_ipt_lms2ipt/ipt2lms',
      '266': 'add pl_shader_info and change type of pl_dispatch_info.shader',
      '265': 'remove fields deprecated for libplacebo v4',
      '264': 'add pl_color_map_params.show_clipping',
      '263': 'add pl_peak_detect_params.percentile',
      '262': 'add pl_color_map_params.visualize_rect',
      '261': 'add pl_color_map_params.metadata',
      '260': 'add pl_tone_map_params.input_avg',
      '259': 'add pl_color_space_nominal_luma_ex',
      '258': 'add pl_hdr_metadata_type and pl_hdr_metadata_contains',
      '257': 'add pl_hdr_metadata.max_pq_y and avg_pq_y',
      '256': 'deprecate pl_peak_detect_params.overshoot_margin',
      '255': 'deprecate pl_get_detected_peak and add pl_get_detected_hdr_metadata',
      '254': 'deprecate pl_renderer_params.allow_delayed_peak_detect and add pl_peak_detect_params.allow_delayed',
      '253': 'remove pl_color_space.nominal_min/max and add pl_color_space_nominal_peak',
      '252': 'remove pl_swapchain.impl',
      '251': 'add `utils/dolbyvision.h` and `pl_hdr_metadata_from_dovi_rpu`',
      '250': 'add pl_frame_map_avdovi_metadata',
      '249': 'add `pl_render_error`, `pl_render_errors` and `pl_renderer_get_errors`',
      '248': 'add pl_hook.signature',
      '247': 'add pl_color_map_params.visualize_lut',
      '246': 'add `pl_tone_map_st2094_10` and `pl_tone_map_st2094_40`',
      '245': 'add `pl_tone_map_params.hdr`',
      '244': 'add `pl_map_hdr_metadata`',
      '243': 'add `pl_color_space.nominal_min/max`',
      '242': 'add `pl_hdr_metadata.scene_max/avg` and `pl_hdr_metadata.ootf`',
      '241': 'add `pl_plane_data.swapped`',
      '240': 'add `PL_COLOR_TRC_ST428`',
      '239': 'add `pl_fmt.planes` and `pl_tex.planes`',
      '238': 'add `pl_vulkan_wrap_params.aspect`',
      '237': 'add `pl_vulkan_hold_ex` and `pl_vulkan_release_ex`',
      '236': 'add `pl_vulkan_sem_create` and `pl_vulkan_sem_destroy`',
      '235': 'add `pl_vulkan.get_proc_addr`',
      '234': 'add `pl_gpu_limits.host_cached`',
      '233': 'add `pl_hook.parameters`, `struct pl_hook_par`',
      '232': 'add `pl_plane_data_from_comps`',
      '231': 'add `pl_tone_map_params_infer`',
      '230': 'add PL_COLOR_PRIM_ACES_AP0 and PL_COLOR_PRIM_ACES_AP1',
      '229': 'add pl_shader_sample_ortho2, deprecate pl_shader_sample_ortho',
      '228': 'add pl_icc_params.force_bpc',
      '227': 'refactor `pl_render_info.index` and add `pl_render_info.count`',
      '226': 'add `pl_dither_params.transfer` and `pl_render_params.disable_dither_gamma_correction`',
      '225': 'add `pl_render_params.error_diffusion`',
      '224': 'add `pl_shader_error_diffusion` and related functions',
      '223': 'add <libplacebo/shaders/dithering.h>',
      '222': 'add `pl_icc_params.cache_save/load`',
      '221': 'add `pl_source_frame.first_field`',
      '220': 'add deinterlacing-related fields to `pl_frame` and `pl_render_params`',
      '219': 'add pl_source_frame.duration, deprecating pl_queue_params.frame_duration',
      '218': 'add <libplacbeo/shaders/deinterlacing.h> and pl_shader_deinterlace',
      '217': 'add pl_color_space_infer_map',
      '216': 'add pl_deband_params.grain_neutral',
      '215': 'add pl_opengl_params.get_proc_addr_ex',
      '214': 'drop deprecated legacy C struct names',
      '213': 'add pl_opengl_params.get_proc_addr',
      '212': 'add pl_opengl.major/minor version numbers',
      '211': 'add pl_opengl.extensions and pl_opengl_has_ext',
      '210': 'add PL_HANDLE_MTL_TEX, PL_HANDLE_IOSURFACE, and pl_shared_mem.plane',
      '209': 'add pl_gpu_limits.array_size_constants',
      '208': 'add pl_filter_function.name and pl_filter_config.name',
      '207': 'add pl_render_params.plane_upscaler and plane_downscaler',
      '206': 'add new ICC profile API (pl_icc_open, ...)',
      '205': 'add pl_cie_from_XYZ and pl_raw_primaries_similar, fix pl_cie_xy_equal',
      '204': 'add pl_d3d11_swapchain_params.disable_10bit_sdr',
      '203': 'add pl_film_grain_from_av',
      '202': 'add pl_frame.acquire/release',
      '201': 'add pl_vulkan.(un)lock_queue',
      '200': 'always set pl_vulkan.queue_*',
      '199': 'add pl_plane.flipped',
      '198': 'remove PL_HOOK_PRE_OVERLAY',
      '197': 'add pl_overlay.coords, change type of pl_overlay_part.dst',
      '196': 'add pl_render_params.force_low_bit_depth_fbos',
      '195': 'change pl_log_create prototype to pl_log_create_${api_ver} to do linking time api check',
      '194': 'add pl_primaries_valid',
      '193': 'add pl_hook_params.orig_repr/color',
      '192': 'add pl_map_avframe_ex',
      '191': 'add pl_map_dovi_metadata',
      '190': 'add pl_color_map_params.gamut_mode, replacing gamut_clipping/warning',
      '189': 'refactor pl_color_space, merging it with pl_hdr_metadata',
      '188': 'refactor pl_color_map_params tone mapping settings',
      '187': 'add <libplacebo/tone_mapping.h>',
      '186': 'add pl_d3d11_swapchain_params.flags',
      '185': 'add PL_COLOR_SYSTEM_DOLBYVISION and reshaping',
      '184': 'add pl_map_avframe/pl_unmap_avframe, deprecate pl_upload_avframe',
      '183': 'relax pl_shared_mem.size > 0 requirement',
      '182': 'add pl_vulkan_get, pl_opengl_get, pl_d3d11_get',
      '181': 'add pl_shader_set_alpha, change alpha handling of pl_shader_decode_color',
      '180': 'add pl_gpu_limits.max_variable_comps',
      '179': 'add pl_render_params.skip_caching_single_frame',
      '178': 'add pl_gpu_limits.align_vertex_stride',
      '177': 'add debug_tag to pl_tex/buf_params',
      '176': 'revert vulkan 1.2 requirement',
      '175': 'require timeline semaphores for all vulkan devices',
      '174': 'deprecate pl_vulkan_params.disable_events',
      '173': 'remove VkAccessFlags from pl_vulkan_hold/release',
      '172': 'replace VkSemaphore by pl_vulkan_sem in pl_vulkan_hold/release',
      '171': 'make vulkan 1.2 the minimum version',
      '170': 'allow pl_queue_update on NULL',
      '169': 'refactor pl_pass_params.target_dummy into target_format',
      '168': 'refactor pl_tex_transfer.stride_w/h into row/depth_pitch',
      '167': 'expose pl_dispatch_reset_frame',
      '166': 'add pl_index_format',
      '165': 'add pl_fmt.signature',
      '164': 'support blending against tiles',
      '163': 'add pl_frame_copy_stream_props',
      '162': 'support rotation in pl_renderer',
      '161': 'make H.274 film grain values indirect',
      '160': 'add preprocessor macros for default params',
      '159': 'remove fields deprecated for libplacebo v3',
      '158': 'add support for H.274 film grain',
      '157': 'add pl_peak_detect_params.minimum_peak',
      '156': 'add pl_swapchain_colors_from_avframe/dav1dpicture',
      '155': 'refactor pl_swapchain_hdr_metadata into pl_swapchain_colorspace_hint',
      '154': 'add <libplacebo/d3d11.h>',
      '153': 'add pl_render_info callbacks',
      '152': 'add pl_dispatch_info callbacks',
      '151': 'pl_shader_res.description/steps',
      '150': 'add PL_FMT_CAP_READWRITE',
      '149': 'add pl_gpu_limits.buf_transfer',
      '148': 'refactor pl_gpu_caps',
      '147': 'add pl_color_space.sig_floor and black point adaptation',
      '146': 'add PL_COLOR_TRC_GAMMA20, GAMMA24 and GAMMA26',
      '145': 'add pl_render_params/pl_shader_params.dynamic_constants',
      '144': 'add support for pl_constant specialization constants',
      '143': 'add pl_color_space_infer_ref',
      '142': 'add pl_render_params.background_transparency and pl_frame_clear_rgba',
      '141': 'add pl_filter_oversample',
      '140': 'add pl_shader_sample_oversample',
      '139': 'make vulkan 1.1 the minimum vulkan version',
      '138': 're-add and properly deprecate pl_filter_haasnsoft',
      '137': 'change behavior of pl_image_mix.num_frames == 1',
      '136': 'add pl_fmt.gatherable',
      '135': 'add pl_queue_params.interpolation_threshold',
      '134': 'add pl_render_params.ignore_icc_profiles',
      '133': 'remove pl_shader_signature',
      '132': 'add pl_tex_clear_ex',
      '131': 'remove PL_PRIM_TRIANGLE_FAN',
      '130': 'provide typedefs for object types, e.g. const struct pl_tex * -> pl_tex',
      '129': 'rename pl_context to pl_log, move <libplacebo/context.h> to <libplacebo/log.h>',
      '128': 'add pl_opengl_params.make/release_current, for thread safety',
      '127': 'add pl_get_buffer2',
      '126': 'add pl_render_params.background_color',
      '125': 'allow calling pl_render_image on NULL',
      '124': 'make pl_queue_update return valid data even on PL_QUEUE_MORE',
      '123': 'refactor pl_overlay from pl_plane into pl_overlay_part',
      '122': 'make pl_gpu thread safe',
      '121': 'add pl_queue_push_block and refactor frame queue threading',
      '120': 'refactor pl_named_filter_config into pl_filter_preset',
      '119': 'add pl_color_adjustment.temperature',
      '118': 'add <libplacebo/utils/frame_queue.h>',
      '117': 'rename pl_filter_triangle/box to pl_filter_bilinear/nearest',
      '116': 'add pl_frame_recreate_from_avframe and pl_download_avframe',
      '115': 'add pl_dispatch_vertex',
      '114': 'add pl_pass_run_params.index_data',
      '113': 'add <libplacebo/shaders/lut.h>',
      '112': 'add <libplacebo/shaders/icc.h>, replacing existing 3dlut API',
      '111': 'add pl_fmt.modifiers for proper DRM format modifier support',
      '110': 'refactor pl_upload_dav1dpicture',
      '109': 'add support for host pointer imports on OpenGL',
      '108': 'add <libplacebo/utils/dav1d.h>',
      '107': 'add pl_render_image_mix',
      '106': 'add pl_shared_mem.stride_w/h',
      '105': 'add asynchronous texture transfers',
      '104': 'add pl_render_params.blend_params',
      '103': 'move pl_tex_sample_mode from pl_tex_params to pl_desc_binding',
      '102': 'add pl_tex_poll',
      '101': 'merge pl_image and pl_render_target into pl_frame',
      '100': 'add pl_render_target.planes',
      '99': 'add pl_sample_src.component_mask',
      '98': 'add pl_vulkan_params.disable_overmapping',
      '97': 'add pl_av1_grain_params.luma_comp',
      '96': 'add <libplacebo/utils/libav.h>',
      '95': 'add PL_COLOR_PRIM_EBU3213 and FILM_C',
      '94': 'add support for //!BUFFER to user shaders',
      '93': 'add pl_plane_data_align',
      '92': 'add more pl_var helper functions',
      '91': 'implement PL_HANDLE_DMA_BUF for EGL',
      '90': 'add pl_opengl_params.allow_software',
      '89': 'use uniform arrays instead of shader literals for LUTs',
      '88': 'add pl_shared_mem.drm_format_mod',
      '87': 'refactor pl_opengl_wrap',
      '86': 'add pl_pass_run_params.vertex_buf',
      '85': 'add PL_HANDLE_HOST_PTR',
      '84': 'add pl_buf_params.import/export_handle',
      '83': 'add pl_shader_custom',
      '82': 'add pl_gpu_is_failed',
      '81': 'add PL_GPU_CAP_SUBGROUPS',
      '80': 'add pl_color_map_params.gamut_clipping',
      '79': 'add pl_get_detected_peak',
      '78': 'add pl_buf_copy',
      '77': 'make all pl_buf_* commands implicitly synchronized',
      '76': 'add pl_vulkan_swapchain_params.prefer_hdr',
      '75': 'add pl_dispatch_save/load',
      '74': 'remove pl_image.signature',
      '73': 'add pl_memory_qualifiers',
      '72': 'generalize PL_SHADER_SIG_SAMPLER2D into PL_SHADER_SIG_SAMPLER',
      '71': 'add pl_opengl_wrap/unwrap',
      '70': 'add pl_tex_sampler_type',
      '69': 'add pl_peak_detect_params.overshoot_margin',
      '68': 'add PL_TONE_MAPPING_BT_2390',
      '67': 'add pl_image_set_chroma_location',
      '66': 'change pl_render_target.dst_rect from pl_rect2d to pl_rect2df',
      '65': 'add PL_SHADER_SIG_SAMPLER2D',
      '64': 'add pl_rect2df_aspect_* family of functions',
      '63': 'refactor pl_shader_av1_grain',
      '62': 'refactor PL_COLOR_REF_WHITE into PL_COLOR_SDR_WHITE and PL_COLOR_SDR_WHITE_HLG',
      '61': 'refactor pl_dispatch_finish etc. to support timers',
      '60': 'add pl_timer',
      '59': 'add pl_render_high_quality_params',
      '58': 'add <libplacebo/shaders/custom.h> and pl_hook',
      '57': 'add width/height fields to pl_dispatch_compute',
      '56': 'make pl_vulkan.features etc. extensible',
      '55': 'add pl_vulkan_params.features',
      '54': 'add pl_vulkan_import',
      '53': 'refactor pl_vulkan_wrap',
      '52': 'revert addition of pl_filter_nearest',
      '51': 'add pl_vulkan_hold_raw',
      '50': 'add pl_vulkan_params.device_uuid',
      '49': 'add pl_filter_nearest',
      '48': 'deprecate pl_image.width/height',
      '47': 'add more matrix math helpers to common.h',
      '46': 'add pl_vk_inst_params.debug_extra',
      '45': 'add pl_vulkan.api_version',
      '44': 'add pl_swapchain_hdr_metadata',
      '43': 'add pl_vulkan/opengl_params.max_glsl_version',
      '42': 'add pl_vk_inst_params.layers/opt_layers',
      '41': 'add PL_FMT_CAP_HOST_READABLE',
      '40': 'add PL_GPU_CAP_BLITTABLE_1D_3D',
      '39': 'add pl_render_params.disable_fbos',
      '38': 'add pl_render_params.force_dither',
      '37': 'add pl_color_levels_guess',
      '36': 'remove pl_opengl.priv leftover',
      '35': 'fix pl_vulkan_swapchain_suboptimal signature',
      '34': 'add <libplacebo/opengl.h>',
      '33': 'add pl_image.av1_grain',
      '32': 'refactor pl_grain_params',
      '31': 'add pl_vulkan_params.get_proc_addr',
      '30': 'add pl_gpu.pci',
      '29': 'add pl_vulkan_swapchain_params.allow_suboptimal',
      '28': 'eliminate void *priv fields from all object types',
      '27': 'add pl_vulkan_choose_device',
      '26': 'add PL_GPU_CAP_MAPPED_BUFFERS',
      '25': 'add pl_fmt.internal_size',
      '24': 'add pl_vulkan_params.disable_events',
      '23': 'add error checking to functions in <libplacebo/gpu.h>',
      '22': 'add pl_vulkan_params.blacklist_caps',
      '21': 'add pl_shader_params.glsl',
      '20': 'refactor pl_shader_alloc',
      '19': 'default to GLSL 130 instead of 110 if unspecified',
      '18': 'add pl_swapchain_resize',
      '17': 'add pl_context_update',
      '16': 'add pl_tex/buf_params.user_data',
      '15': 'add <libplacebo/dummy.h>',
      '14': 'remove ident from pl_shader_reset',
      '13': 'add pl_render_params.peak_detect_params',
      '12': 'add pl_shader_detect_peak',
      '11': 'add pl_var_int',
      '10': 'refactor pl_color_map_params desaturation fields',
      '9': 'add pl_tex_params.import/export_handle',
      '8': 'add pl_color_space.sig_scale',
      '7': 'initial major release',
      '6': '',
      '5': '',
      '4': '',
      '3': '',
      '2': '',
      '1': '',
    }.keys().length(),
    # Fix version
    0)
)

### Version number and configuration
version = meson.project_version()
version_pretty = 'v' + version
version_split = version.split('.')

majorver = version_split[0]
apiver = version_split[1]
fixver = version_split[2]

# Configuration data
conf_public = configuration_data()
conf_internal = configuration_data()
conf_public.set('majorver', majorver)
conf_public.set('apiver', apiver)
conf_internal.set('BUILD_API_VER', apiver)
conf_internal.set('BUILD_FIX_VER', fixver)
conf_internal.set('PL_DEBUG_ABORT', get_option('debug-abort'))


### Global build options
build_opts = [
  # Warnings
  '-Wundef', '-Wshadow', '-Wparentheses', '-Wpointer-arith',
  '-fno-math-errno',
]

link_args = []

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

c_opts = [
  '-D_ISOC99_SOURCE', '-D_ISOC11_SOURCE', '-D_GNU_SOURCE', '-U__STRICT_ANSI__',
  '-Wmissing-prototypes',

  # Warnings to ignore
  '-Wno-sign-compare', '-Wno-unused-parameter',
  '-Wno-missing-field-initializers', '-Wno-type-limits',

  # Warnings to treat as errors
  '-Werror=implicit-function-declaration',
]

if cc.has_argument('-Wincompatible-pointer-types')
  c_opts += ['-Werror=incompatible-pointer-types']
endif

# clang's version of -Wmissing-braces rejects the common {0} initializers
if cc.get_id() == 'clang'
  c_opts += ['-Wno-missing-braces']
endif

# For sanitizers to work/link properly some public symbols have to be available.
if get_option('b_sanitize') == 'none'
  # don't leak library symbols if possible
  vflag = '-Wl,--exclude-libs=ALL'
  # link and lld-link don't support this arg, but it only shows warning about
  # unsupported argument. Meson doesn't detect it, so manually exclude them.
  if cc.has_link_argument(vflag) and not ['lld-link', 'link'].contains(cc.get_linker_id())
    link_args += [vflag]
  endif
endif

# OS specific build options
if host_machine.system() == 'windows'
    build_opts += ['-D_WIN32_WINNT=0x0601',
                   '-D_USE_MATH_DEFINES',
                   '-DWIN32_LEAN_AND_MEAN',
                   '-DNOMINMAX',
                   '-D_CRT_SECURE_NO_WARNINGS']
    subdir('win32')
endif

add_project_arguments(build_opts + c_opts, language: ['c'])
add_project_arguments(build_opts, language: ['c', 'cpp'])
add_project_link_arguments(link_args, language: ['c', 'cpp'])


# Global dependencies
fs = import('fs')
libm = cc.find_library('m', required: false)
thirdparty = meson.project_source_root()/'3rdparty'
python = import('python').find_installation()
python_env = environment()
python_env.append('PYTHONPATH', thirdparty/'jinja/src')
python_env.append('PYTHONPATH', thirdparty/'markupsafe/src')
python_env.append('PYTHONPATH', thirdparty/'glad')

if host_machine.system() == 'windows'
    threads = declare_dependency()
else
    pthreads = dependency('threads')
    has_setclock = cc.has_header_symbol(
                     'pthread.h',
                     'pthread_condattr_setclock',
                     dependencies: pthreads,
                     args: c_opts,
                   )

    threads = declare_dependency(
      dependencies: pthreads,
      compile_args: [pthreads.found() ? '-DPL_HAVE_PTHREAD' : '',
                     has_setclock ? '-DPTHREAD_HAS_SETCLOCK' : '',]
    )
endif

build_deps = [ libm, threads ]

subdir('tools')
subdir('src')

if get_option('demos')
  subdir('demos')
endif

# Allows projects to build libplacebo by cloning into ./subprojects/libplacebo
meson.override_dependency('libplacebo', libplacebo)
