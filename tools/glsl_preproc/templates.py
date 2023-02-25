import jinja2
import os.path

TEMPLATEDIR = os.path.dirname(__file__) + '/templates'
TEMPLATES = jinja2.Environment(
    loader        = jinja2.FileSystemLoader(searchpath=TEMPLATEDIR),
    lstrip_blocks = True,
    trim_blocks   = True,
)

GLSL_BLOCK_TEMPLATE = TEMPLATES.get_template('glsl_block.c.j2')
FUNCTION_TEMPLATE   = TEMPLATES.get_template('function.c.j2')
CALL_TEMPLATE       = TEMPLATES.get_template('call.c.j2')
STRUCT_TEMPLATE     = TEMPLATES.get_template('struct.c.j2')
