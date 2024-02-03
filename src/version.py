#!/usr/bin/env python3

import sys
import subprocess

infilename, outfilename, source_dir, project_version_pretty = sys.argv[1:]

try:
    proc = subprocess.run(['git', 'describe', '--dirty'], cwd=source_dir,
                          capture_output=True, text=True)
    proc.check_returncode()
except (FileNotFoundError, subprocess.CalledProcessError):
    # No git or no repo. Hopefully a release tarball.
    version = project_version_pretty
else:
    version = '{} ({})'.format(project_version_pretty, proc.stdout.strip())

with open(infilename, 'r') as infile:
    output = infile.read().replace('@buildver@', version)
# Avoid touching file (triggering recompilation) if it's already up to date.
try:
    with open(outfilename, 'r') as outfile:
        write_output = outfile.read() != output
except FileNotFoundError:
    write_output = True
if write_output:
    with open(outfilename, 'w') as outfile:
        outfile.write(output)
