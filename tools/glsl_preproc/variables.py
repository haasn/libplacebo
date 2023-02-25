import re

def slugify(value):
    value = re.sub(r'[^\w]+', '_', value.lower()).strip('_')
    if value[:1].isdigit():
        value = '_' + value
    return value

# A single variable (enclosed by the template)
class Var(object):
    STRUCT_NAME = 'vars'
    CSIZES = {
        # This array doesn't have to be exact, it's only used for sorting
        # struct members to save a few bytes of memory here and there
        'ident_t':  8,
        'int':      4,
        'unsigned': 4,
        'float':    4,
        'uint8_t':  1,
        'bool':     1,
    }

    def __init__(self, ctype, expr, name, csize=0, linenr=0):
        self.ctype  = ctype
        self.csize  = csize or Var.CSIZES[ctype]
        self.expr   = expr
        self.name   = name
        self.linenr = linenr

    def ref(self):
        return f'{Var.STRUCT_NAME}.{self.name}'

def is_literal(expr):
    return expr.isnumeric() or expr in ['true', 'false']

# A (deduplicated) set of variables
class VarSet(object):
    def __init__(self):
        self.varmap = {} # expr -> cvar

    def __iter__(self):
        # Sort from largest to smallest variable to optimize struct padding
        yield from sorted(self.varmap.values(),
            reverse=True,
            key=lambda v: v.csize,
        )

    def __bool__(self):
        return True if self.varmap else False

    # Returns a reference to the added var
    def add_var(self, ctype, expr, name=None, linenr=0):
        assert expr
        expr = expr.strip()
        if is_literal(expr):
            return expr
        name = name or slugify(expr)

        # re-use existing entry for identical expression/type pairs
        if var := self.varmap.get(expr):
            if ctype != var.ctype:
                raise SyntaxError(f'Conflicting types for expression {expr}, '
                                  f'got {ctype}, expected {var.ctype}')
            assert var.name == name
            return var

        var = Var(ctype, expr=expr, name=name, linenr=linenr)
        names = [ v.name for v in self.varmap.values() ]
        while var.name in names:
            var.name += '_'
        self.varmap[expr] = var
        return var.ref()

    def merge(self, other):
        for var in other:
            self.add_var(var.ctype, var.expr, var.name, var.linenr)

# Format specifier + corresponding C type and expression wrapper
class FmtSpec(object):
    def __init__(self, ctype='ident_t', fmtstr='_%hx', wrap_expr=lambda name, expr: expr):
        self.ctype     = ctype
        self.fmtstr    = fmtstr
        self.wrap_expr = wrap_expr

    @staticmethod
    def wrap_var(type, dynamic=False):
        if dynamic:
            return lambda name, expr: f'sh_var_{type}(sh, "{name}", {expr}, true)'
        else:
            return lambda name, expr: f'sh_const_{type}(sh, "{name}", {expr})'

# Current list of format types
class Fmt(object):
    IDENT       = FmtSpec()
    INT_CONST   = FmtSpec(ctype='int',      fmtstr='%d')
    UINT_CONST  = FmtSpec(ctype='unsigned', fmtstr='%u')
    FLOAT_CONST = FmtSpec(ctype='float',    fmtstr='%f')
    INT_VAR     = FmtSpec(wrap_expr=FmtSpec.wrap_var('int'))
    UINT_VAR    = FmtSpec(wrap_expr=FmtSpec.wrap_var('uint'))
    FLOAT_VAR   = FmtSpec(wrap_expr=FmtSpec.wrap_var('float'))
    INT_DYN     = FmtSpec(wrap_expr=FmtSpec.wrap_var('int', dynamic=True))
    UINT_DYN    = FmtSpec(wrap_expr=FmtSpec.wrap_var('uint', dynamic=True))
    FLOAT_DYN   = FmtSpec(wrap_expr=FmtSpec.wrap_var('float', dynamic=True))
