#!/usr/bin/env python3
#
# This file is part of libplacebo.
#
# libplacebo is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# libplacebo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.

import os.path
import sys
import xml.etree.ElementTree as ET

try:
    from mako.template import Template
except ModuleNotFoundError:
    print('Module \'mako\' not found, please install \'python3-mako\' or '
          'an equivalent package on your system!', file=sys.stderr)
    sys.exit(1)

TEMPLATE = Template("""
#define VK_ENABLE_BETA_EXTENSIONS
#include "vulkan/utils.h"

const char *vk_res_str(VkResult res)
{
    switch (res) {
%for res in vkresults:
    case ${res}: return "${res}";
%endfor

    default: return "unknown error";
    }
}

const char *vk_obj_str(VkObjectType obj)
{
    switch (obj) {
%for obj in vkobjects:
    case ${obj.enum}: return "${obj.name}";
%endfor

    default: return "unknown object";
    }
}

size_t vk_struct_size(VkStructureType stype)
{
    switch (stype) {
%for struct in vkstructs:
    case ${struct.stype}: return sizeof(${struct.name});
%endfor

    default: return 0;
    }
}
""")

class Obj(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_vkresults(registry):
    for e in registry.findall('enums[@name="VkResult"]/enum'):
        yield e.attrib['name']

def get_vkobjects(registry):
    for e in registry.findall('enums[@name="VkObjectType"]/enum'):
        if 'comment' in e.attrib:
            yield Obj(enum = e.attrib['name'],
                      name = e.attrib['comment'])

def get_vkstructs(registry):
    for e in registry.findall('types/type[@category="struct"]'):
        # Strings for platform-specific crap we want to blacklist as they will
        # most likely cause build failures
        blacklist_strs = [
            'ANDROID', 'Surface', 'Win32', 'D3D12', 'GGP'
        ]

        if any([ str in e.attrib['name'] for str in blacklist_strs ]):
            continue

        stype = None
        for m in e.findall('member'):
            if m.find('name').text == 'sType':
                stype = m
                break

        if stype and 'values' in stype.attrib:
            yield Obj(stype = stype.attrib['values'],
                      name = e.attrib['name'])

def find_registry_xml():
    registry_paths = [
        '/usr/share/vulkan/registry/vk.xml',
        '%VULKAN_SDK%/share/vulkan/registry/vk.xml',
        '$MINGW_PREFIX/share/vulkan/registry/vk.xml',
    ]

    for p in registry_paths:
        path = os.path.expandvars(p)
        if os.path.isfile(path):
            print('Found vk.xml: {0}'.format(path))
            return path

    print('Could not find the vulkan registry (vk.xml), please specify its '
          'location manually using the -Dvulkan-registry=/path/to/vk.xml '
          'option!', file=sys.stderr)
    sys.exit(1)

if __name__ == '__main__':
    assert len(sys.argv) == 3
    xmlfile = sys.argv[1]
    outfile = sys.argv[2]

    if not xmlfile or xmlfile == '':
        xmlfile = find_registry_xml()

    registry = ET.parse(xmlfile)
    with open(outfile, 'w') as f:
        f.write(TEMPLATE.render(
            vkresults = get_vkresults(registry),
            vkobjects = get_vkobjects(registry),
            vkstructs = get_vkstructs(registry),
        ))
