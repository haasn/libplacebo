import re

from templates import GLSL_BLOCK_TEMPLATE
from variables import VarSet, slugify

VAR_PATTERN = re.compile(flags=re.VERBOSE, pattern=r'''
    # long form ${ ... } syntax
    \${ (?:\s*(?P<type>(?:              # optional type prefix
            ident                       # identifiers (always dynamic)
          | (?:(?:const|dynamic)\s+)?   # optional const/dynamic modifiers
                (?:float|u?int)         # base type
          | swizzle                     # swizzle mask
          | (?:b|i|u)?vecType           # vector type (for mask)
        )):)?
        (?P<expr>[^{}]+)
      }
|   \$(?P<name>\w+) # reference to captured variable
|   @(?P<var>\w+)   # reference to locally defined var
''')

class FmtSpec(object):
    def __init__(self, ctype='ident_t', fmtstr='_%hx',
                 wrap_expr=lambda name, expr: expr,
                 fmt_expr=lambda name: name):
        self.ctype     = ctype
        self.fmtstr    = fmtstr
        self.wrap_expr = wrap_expr
        self.fmt_expr  = fmt_expr

    @staticmethod
    def wrap_var(type, dynamic=False):
        if dynamic:
            return lambda name, expr: f'sh_var_{type}(sh, "{name}", {expr}, true)'
        else:
            return lambda name, expr: f'sh_const_{type}(sh, "{name}", {expr})'

    @staticmethod
    def wrap_fn(fn):
        return lambda name: f'{fn}({name})'

VAR_TYPES = {
    # identifiers: get mapped as-is
    'ident':            FmtSpec(),

    # normal variables: get mapped as shader constants
    'int':              FmtSpec(wrap_expr=FmtSpec.wrap_var('int')),
    'uint':             FmtSpec(wrap_expr=FmtSpec.wrap_var('uint')),
    'float':            FmtSpec(wrap_expr=FmtSpec.wrap_var('float')),

    # constant variables: get printed directly into the source code
    'const int':        FmtSpec(ctype='int',          fmtstr='%d'),
    'const uint':       FmtSpec(ctype='unsigned',     fmtstr='uint(%u)'),
    'const float':      FmtSpec(ctype='float',        fmtstr='float(%f)'),

    # dynamic variables: get loaded as shader variables
    'dynamic int':      FmtSpec(wrap_expr=FmtSpec.wrap_var('int', dynamic=True)),
    'dynamic uint':     FmtSpec(wrap_expr=FmtSpec.wrap_var('uint', dynamic=True)),
    'dynamic float':    FmtSpec(wrap_expr=FmtSpec.wrap_var('float', dynamic=True)),

    # component mask types
    'swizzle':          FmtSpec(ctype='uint8_t', fmtstr='%s', fmt_expr=FmtSpec.wrap_fn('sh_swizzle')),
    'bvecType':         FmtSpec(ctype='uint8_t', fmtstr='%s', fmt_expr=FmtSpec.wrap_fn('sh_bool_type')),
    'ivecType':         FmtSpec(ctype='uint8_t', fmtstr='%s', fmt_expr=FmtSpec.wrap_fn('sh_int_type')),
    'uvecType':         FmtSpec(ctype='uint8_t', fmtstr='%s', fmt_expr=FmtSpec.wrap_fn('sh_uint_type')),
    'vecType':          FmtSpec(ctype='uint8_t', fmtstr='%s', fmt_expr=FmtSpec.wrap_fn('sh_float_type')),
}

def stringify(value, strip):
    if strip:
        value = re.sub(r'(?:\/\*[^\*]*\*\/|\/\/[^\n]+|^\s*)', '', value)
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '\\n"'

def commentify(value, strip):
    if strip:
        return ''
    return '/*' + value.replace('/*', '[[').replace('*/', ']]') + '*/'

# Represents a statement + its enclosed variables
class Statement(object):
    def __init__(self, linenr=0):
        super().__init__()
        self.linenr = linenr
        self.vars = VarSet()

    def add_var(self, ctype, expr, name=None):
        return self.vars.add_var(ctype, expr, name, self.linenr)

    def render(self):
        raise NotImplementedError

    @staticmethod
    def parse(text_orig, **kwargs):
        raise NotImplementedError

# Represents a single line of GLSL
class GLSLLine(Statement):
    class GLSLVar(object): # variable reference
        def __init__(self, fmt, var):
            self.fmt = fmt
            self.var = var

    def __init__(self, text, strip=False, **kwargs):
        super().__init__(**kwargs)
        self.refs = []
        self.strip = strip

        # produce two versions of line, one for printf() and one for append()
        text = text.rstrip()
        self.rawstr = stringify(text, strip)
        self.fmtstr = stringify(re.sub(VAR_PATTERN, self.handle_var, text.replace('%', '%%')), strip)

    def handle_var(self, match):
        # local @var
        if match['var']:
            self.refs.append(match['var'])
            return '%d'

        # captured $var
        type = match['type']
        name = match['name']
        expr = match['expr'] or name
        name = name or slugify(expr)

        fmt = VAR_TYPES[type or 'ident']
        self.refs.append(fmt.fmt_expr(self.add_var(
            ctype = fmt.ctype,
            expr  = fmt.wrap_expr(name, expr),
            name  = name,
        )))

        if fmt.ctype == 'ident_t':
            return commentify(name, self.strip) + fmt.fmtstr
        else:
            return fmt.fmtstr

# Represents an entire GLSL block
class GLSLBlock(Statement):
    def __init__(self, line):
        super().__init__(linenr=line.linenr)
        self.lines = []
        self.refs  = []
        self.append(line)

    def append(self, line):
        assert isinstance(line, GLSLLine)
        self.lines.append(line)
        self.refs += line.refs
        self.vars.merge(line.vars)

    def render(self):
        return GLSL_BLOCK_TEMPLATE.render(block=self)

# Represents a statement which can either take a single line or a block
class BlockStart(Statement):
    def __init__(self, multiline=False, **kwargs):
        super().__init__(**kwargs)
        self.multiline = multiline

    def add_brace(self, text):
        if self.multiline:
            text += ' {'
        return text

# Represents an @if
class IfCond(BlockStart):
    def __init__(self, cond, inner=False, **kwargs):
        super().__init__(**kwargs)
        self.cond = cond if inner else self.add_var('bool', expr=cond)

    def render(self):
        return self.add_brace(f'if ({self.cond})')

# Represents an @else
class Else(BlockStart):
    def __init__(self, closing, **kwargs):
        super().__init__(**kwargs)
        self.closing = closing

    def render(self):
        text = '} else' if self.closing else 'else'
        return self.add_brace(text)

# Represents a normal (integer) @for loop, or an (unsigned 8-bit) bitmask loop
class ForLoop(BlockStart):
    def __init__(self, var, op, bound, **kwargs):
        super().__init__(**kwargs)
        self.comps = op == ':'
        self.bound = self.add_var('uint8_t' if self.comps else 'int', expr=bound)
        self.var   = var
        self.op    = op

    def render(self):
        if self.comps:
            loopstart = f'uint8_t _mask = {self.bound}, {self.var}'
            loopcond  = f'_mask && ({self.var} = __builtin_ctz(_mask), 1)'
            loopstep  = f'_mask &= ~(1u << {self.var})'
        else:
            loopstart = f'int {self.var} = 0'
            loopcond  = f'{self.var} {self.op} {self.bound}'
            loopstep  = f'{self.var}++'

        return self.add_brace(f'for ({loopstart}; {loopcond}; {loopstep})')

# Represents a @switch block
class Switch(Statement):
    def __init__(self, expr, **kwargs):
        super().__init__(**kwargs)
        self.expr = self.add_var('unsigned', expr=expr)

    def render(self):
        return f'switch ({self.expr}) {{'

# Represents a @case label
class Case(Statement):
    def __init__(self, label, **kwargs):
        super().__init__(**kwargs)
        self.label = label

    def render(self):
        return f'case {self.label}:'

# Represents a @default line
class Default(Statement):
    def render(self):
        return 'default:'

# Represents a @break line
class Break(Statement):
    def render(self):
        return 'break;'

# Represents a single closing brace
class EndBrace(Statement):
    def render(self):
        return '}'

# Shitty regex-based statement parser
PATTERN_IF  = re.compile(flags=re.VERBOSE, pattern=r'''
@\s*if\s*                       # '@if'
(?P<inner>@)?                   # optional leading @
\((?P<cond>.+)\)\s*             # (condition)
(?P<multiline>{)?\s*            # optional trailing {
$''')

PATTERN_ELSE = re.compile(flags=re.VERBOSE, pattern=r'''
@\s*(?P<closing>})?\s*          # optional leading }
else\s*                         # 'else'
(?P<multiline>{)?\s*            # optional trailing {
$''')

PATTERN_FOR = re.compile(flags=re.VERBOSE, pattern=r'''
@\s*for\s+\(                    # '@for' (
(?P<var>\w+)\s*                 # loop variable name
(?P<op>(?:\<=?|:))(?=[\w\s])\s* # '<', '<=' or ':', followed by \s or \w
(?P<bound>[^\s].*)\s*           # loop boundary expression
\)\s*(?P<multiline>{)?\s*       # ) and optional trailing {
$''')

PATTERN_SWITCH = re.compile(flags=re.VERBOSE, pattern=r'''
@\s*switch\s*                   # '@switch'
\((?P<expr>.+)\)\s*{            # switch expression
$''')

PATTERN_CASE = re.compile(flags=re.VERBOSE, pattern=r'''
@\s*case\s*                     # '@case'
(?P<label>[^:]+):?              # case label, optionally followed by :
$''')

PATTERN_BREAK   = r'@\s*break;?\s*$'
PATTERN_DEFAULT = r'@\s*default:?\s*$'
PATTERN_BRACE   = r'@\s*}\s*$'

PARSERS = {
    PATTERN_IF:         lambda r, **kw: IfCond(r['cond'], inner=r['inner'], multiline=r['multiline'], **kw),
    PATTERN_ELSE:       lambda r, **kw: Else(closing=r['closing'], multiline=r['multiline'], **kw),
    PATTERN_FOR:        lambda r, **kw: ForLoop(r['var'], r['op'], r['bound'], multiline=r['multiline'], **kw),
    PATTERN_SWITCH:     lambda r, **kw: Switch(r['expr'], **kw),
    PATTERN_CASE:       lambda r, **kw: Case(r['label'], **kw),
    PATTERN_BREAK:      lambda _, **kw: Break(**kw),
    PATTERN_DEFAULT:    lambda _, **kw: Default(**kw),
    PATTERN_BRACE:      lambda _, **kw: EndBrace(**kw),
}

def parse_line(text_orig, strip, **kwargs):
    # skip empty lines
    text = text_orig.strip()
    if not text:
        return None
    if text.lstrip().startswith('@'):
        # try parsing as statement
        for pat, fun in PARSERS.items():
            if res := re.match(pat, text):
                return fun(res, **kwargs)
        # return generic error for unrecognized statements
        raise SyntaxError('Syntax error in directive: ' + text.lstrip())
    else:
        # default to literal GLSL line
        return GLSLLine(text_orig, strip, **kwargs)

Statement.parse = parse_line
