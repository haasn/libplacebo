#!/usr/bin/env python3

import sys
import argparse

from macros import Macro

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('-s', '--strip', default=False, action='store_true')
args = parser.parse_args()

with open(args.input) as infile:
    with open(args.output, 'w') as outfile:
        for line in Macro.process_file(infile, strip=args.strip):
            outfile.write(line)
