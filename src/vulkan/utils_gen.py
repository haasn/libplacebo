#!/usr/bin/env python3
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

import sys
import xml.etree.ElementTree as ET
from mako.template import Template

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

if __name__ == '__main__':
    assert len(sys.argv) == 3
    xmlfile = sys.argv[1]
    outfile = sys.argv[2]

    registry = ET.parse(xmlfile)
    with open(outfile, 'w') as f:
        f.write(TEMPLATE.render(
            vkresults = get_vkresults(registry),
            vkobjects = get_vkobjects(registry),
        ))
