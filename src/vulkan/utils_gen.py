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
          'an equivalent package on your system! Alternatively, run '
          '`git submodule update --init` followed by `meson --wipe`.',
          file=sys.stderr)
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

const char *vk_fmt_name(VkFormat fmt)
{
    switch (fmt) {
%for fmt in vkformats:
    case ${fmt}: return "${fmt}";
%endfor

    default: return "unknown format";
    }
}

const char *vk_csp_name(VkColorSpaceKHR csp)
{
    switch (csp) {
%for csp in vkspaces:
    case ${csp}: return "${csp}";
%endfor

    default: return "unknown color space";
    }
}

const char *vk_handle_name(VkExternalMemoryHandleTypeFlagBitsKHR handle)
{
    switch (handle) {
%for handle in vkhandles:
    case ${handle}: return "${handle}";
%endfor

    default: return "unknown handle type";
    }
}

const char *vk_alpha_mode(VkCompositeAlphaFlagsKHR alpha)
{
    switch (alpha) {
%for mode in vkalphas:
    case ${mode}: return "${mode}";
%endfor

    default: return "unknown alpha mode";
    }
}

const char *vk_surface_transform(VkSurfaceTransformFlagsKHR tf)
{
    switch (tf) {
%for tf in vktransforms:
    case ${tf}: return "${tf}";
%endfor

    default: return "unknown surface transform";
    }
}


const char *vk_obj_type(VkObjectType obj)
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

const VkAccessFlags vk_access_read = ${hex(vkaccess.read)}LLU;
const VkAccessFlags vk_access_write = ${hex(vkaccess.write)}LLU;
""")

class Obj(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def findall_enum(registry, name):
    for e in registry.iterfind('enums[@name="{0}"]/enum'.format(name)):
        if not 'alias' in e.attrib:
            yield e
    for e in registry.iterfind('.//enum[@extends="{0}"]'.format(name)):
        # ext 289 is a non-existing extension that defines some names for
        # proprietary downstream consumers, causes problems unless excluded
        if e.attrib.get('extnumber', '0') == '289':
            continue
        # some other extensions contain reserved identifiers that generally
        # translate to compile failures
        if 'RESERVED' in e.attrib['name']:
            continue
        if not 'alias' in e.attrib:
            yield e

def get_vkenum(registry, enum):
    for e in findall_enum(registry, enum):
        yield e.attrib['name']

def get_vkobjects(registry):
    for t in registry.iterfind('types/type[@category="handle"]'):
        if 'objtypeenum' in t.attrib:
            yield Obj(enum = t.attrib['objtypeenum'],
                      name = t.find('name').text)

def get_vkstructs(registry):
    for e in registry.iterfind('types/type[@category="struct"]'):
        # Strings for platform-specific crap we want to blacklist as they will
        # most likely cause build failures
        blacklist_strs = [
            'ANDROID', 'Surface', 'Win32', 'D3D12', 'GGP', 'FUCHSIA', 'Metal',
        ]

        if any([ str in e.attrib['name'] for str in blacklist_strs ]):
            continue

        stype = None
        for m in e.iterfind('member'):
            if m.find('name').text == 'sType':
                stype = m
                break

        if stype and 'values' in stype.attrib:
            yield Obj(stype = stype.attrib['values'],
                      name = e.attrib['name'])

def get_vkaccess(registry):
    access = Obj(read = 0, write = 0)
    for e in findall_enum(registry, 'VkAccessFlagBits'):
        if '_READ_' in e.attrib['name']:
            access.read |= 1 << int(e.attrib['bitpos'])
        if '_WRITE_' in e.attrib['name']:
            access.write |= 1 << int(e.attrib['bitpos'])
    return access

def find_registry_xml(datadir):
    registry_paths = [
        '{0}/vulkan/registry/vk.xml'.format(datadir),
        '%VULKAN_SDK%/share/vulkan/registry/vk.xml',
        '$VULKAN_SDK/share/vulkan/registry/vk.xml',
        '$MINGW_PREFIX/share/vulkan/registry/vk.xml',
        '/usr/share/vulkan/registry/vk.xml',
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
    assert len(sys.argv) == 4
    datadir = sys.argv[1]
    xmlfile = sys.argv[2]
    outfile = sys.argv[3]

    if not xmlfile or xmlfile == '':
        xmlfile = find_registry_xml(datadir)

    registry = ET.parse(xmlfile)
    with open(outfile, 'w') as f:
        f.write(TEMPLATE.render(
            vkresults = get_vkenum(registry, 'VkResult'),
            vkformats = get_vkenum(registry, 'VkFormat'),
            vkspaces  = get_vkenum(registry, 'VkColorSpaceKHR'),
            vkhandles = get_vkenum(registry, 'VkExternalMemoryHandleTypeFlagBits'),
            vkalphas  = get_vkenum(registry, 'VkCompositeAlphaFlagBitsKHR'),
            vktransforms = get_vkenum(registry, 'VkSurfaceTransformFlagBitsKHR'),
            vkobjects = get_vkobjects(registry),
            vkstructs = get_vkstructs(registry),
            vkaccess = get_vkaccess(registry),
        ))
