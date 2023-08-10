#!/usr/bin/env python3

import re

from variables import Var
from templates import *
from statement import *

PATTERN_PRAGMA  = re.compile(flags=re.VERBOSE, pattern=r'''
\s*\#\s*pragma\s+               # '#pragma'
(?P<pragma>(?:                  # pragma name
    GLSL[PHF]?
))\s*
(?P<rest>.*)$                   # rest of line (pragma body)
''')

# Represents a single #pragma macro
class Macro(object):
    PRAGMAS = {
        'GLSL':  'SH_BUF_BODY',
        'GLSLP': 'SH_BUF_PRELUDE',
        'GLSLH': 'SH_BUF_HEADER',
        'GLSLF': 'SH_BUF_FOOTER',
    }

    def __init__(self, linenr=0, type='GLSL'):
        self.linenr = linenr
        self.buf    = Macro.PRAGMAS[type]
        self.name   = '_glsl_' + str(linenr)
        self.body   = []    # list of statements
        self.last   = None  # previous GLSLBlock (if unterminated)
        self.vars   = VarSet()

    def needs_single_line(self):
        if not self.body:
            return False
        prev = self.body[-1]
        return isinstance(prev, BlockStart) and not prev.multiline

    def push_line(self, line):
        self.vars.merge(line.vars)

        if isinstance(line, GLSLLine):
            if self.last:
                self.last.append(line)
            elif self.needs_single_line():
                self.body.append(GLSLBlock(line))
            else:
                # start new GLSL block
                self.last = GLSLBlock(line)
                self.body.append(self.last)
        else:
            self.body.append(line)
            self.last = None

    def render_struct(self):
        return STRUCT_TEMPLATE.render(macro=self)

    def render_call(self):
        return CALL_TEMPLATE.render(macro=self)

    def render_fun(self):
        return FUNCTION_TEMPLATE.render(macro=self, Var=Var)

    # yields output lines
    @staticmethod
    def process_file(lines, strip=False):
        macro = None
        macros = []

        for linenr, line_orig in enumerate(lines, start=1):
            line = line_orig.rstrip()

            # Strip leading spaces, due to C indent. Skip first pragma line.
            if macro and leading_spaces is None:
                leading_spaces = len(line) - len(line.lstrip())

            # check for start of macro
            if not macro:
                leading_spaces = None
                if result := re.match(PATTERN_PRAGMA, line):
                    macro = Macro(linenr, type=result['pragma'])
                    line = result['rest'] # strip pragma prefix

            if macro:
                if leading_spaces:
                    line = re.sub(f'^\s{{1,{leading_spaces}}}', '', line)
                if more_lines := line.endswith('\\'):
                    line = line[:-1]
                if statement := Statement.parse(line, strip=strip, linenr=linenr):
                    macro.push_line(statement)
                if more_lines:
                    continue # stay in macro
                else:
                    yield macro.render_call()
                    yield '#line {}\n'.format(linenr + 1)
                    macros.append(macro)
                    macro = None
            else:
                yield line_orig

        if macros:
            yield '\n// Auto-generated template functions:'
        for macro in macros:
            yield macro.render_fun()
