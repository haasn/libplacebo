#!/usr/bin/env python3

import sys

from macros import Macro

assert len(sys.argv) == 3
with open(sys.argv[1]) as infile:
    with open(sys.argv[2], 'w') as outfile:
        for line in Macro.process_file(infile):
            outfile.write(line)
