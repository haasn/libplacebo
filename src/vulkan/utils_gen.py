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
import re
import sys
import xml.etree.ElementTree as ET

try:
    import jinja2
except ModuleNotFoundError:
    print('Module \'jinja2\' not found, please install \'python3-Jinja2\' or '
          'an equivalent package on your system! Alternatively, run '
          '`git submodule update --init` followed by `meson --wipe`.',
          file=sys.stderr)
    sys.exit(1)

TEMPLATE = jinja2.Environment(
    loader = jinja2.FileSystemLoader(searchpath=os.path.dirname(__file__)),
    trim_blocks=True,
).get_template('utils_gen.c.j2')

class Obj(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class VkXML(ET.ElementTree):
    def blacklist_block(self, req):
        for t in req.iterfind('type'):
            self.blacklist_types.add(t.attrib['name'])
        for e in req.iterfind('enum'):
            self.blacklist_enums.add(e.attrib['name'])

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.blacklist_types = set()
        self.blacklist_enums = set()

        for f in self.iterfind('feature'):
            # Feature block for non-Vulkan API
            if not 'vulkan' in f.attrib['api'].split(','):
                for r in f.iterfind('require'):
                    self.blacklist_block(r)

        for e in self.iterfind('extensions/extension'):
            # Entire extension is unsupported on vulkan or platform-specifid
            if not 'vulkan' in e.attrib['supported'].split(',') or 'platform' in e.attrib:
                for r in e.iterfind('require'):
                    self.blacklist_block(r)
                continue

            # Only individual <require> blocks are API-specific
            for r in e.iterfind('require[@api]'):
                if not 'vulkan' in r.attrib['api'].split(','):
                    self.blacklist_block(r)

    def findall_enum(self, name):
        for e in self.iterfind('enums[@name="{0}"]/enum'.format(name)):
            if not 'alias' in e.attrib:
                if not e.attrib['name'] in self.blacklist_enums:
                    yield e
        for e in self.iterfind('.//enum[@extends="{0}"]'.format(name)):
            if not 'alias' in e.attrib:
                if not e.attrib['name'] in self.blacklist_enums:
                    yield e

    def findall_type(self, category):
        for t in self.iterfind('types/type[@category="{0}"]'.format(category)):
            name = t.attrib.get('name') or t.find('name').text
            if name in self.blacklist_types:
                continue
            yield t


def get_vkenum(registry, enum):
    for e in registry.findall_enum(enum):
        yield e.attrib['name']

def get_vkobjects(registry):
    for t in registry.findall_type('handle'):
        if 'objtypeenum' in t.attrib:
            yield Obj(enum = t.attrib['objtypeenum'],
                      name = t.find('name').text)

def get_vkstructs(registry):
    for t in registry.findall_type('struct'):
        stype = None
        for m in t.iterfind('member'):
            if m.find('name').text == 'sType':
                stype = m
                break

        if stype is not None and 'values' in stype.attrib:
            yield Obj(stype = stype.attrib['values'],
                      name = t.attrib['name'])

def get_vkaccess(registry):
    access = Obj(read = 0, write = 0)
    for e in registry.findall_enum('VkAccessFlagBits2'):
        if '_READ_' in e.attrib['name']:
            access.read |= 1 << int(e.attrib['bitpos'])
        if '_WRITE_' in e.attrib['name']:
            access.write |= 1 << int(e.attrib['bitpos'])
    return access

def get_vkexts(registry):
    for e in registry.iterfind('extensions/extension'):
        promoted_ver = None
        if res := re.match(r'VK_VERSION_(\d)_(\d)', e.attrib.get('promotedto', '')):
            promoted_ver = 'VK_API_VERSION_{0}_{1}'.format(res[1], res[2])
        yield Obj(name = e.attrib['name'],
                  promoted_ver = promoted_ver)

def get_vkfeatures(registry):
    structs = [];
    featuremap = {}; # features -> [struct]
    for t in registry.findall_type('struct'):
        sname = t.attrib['name']
        is_base = sname == 'VkPhysicalDeviceFeatures'
        extends = t.attrib.get('structextends', [])
        if is_base:
            sname = 'VkPhysicalDeviceFeatures2'
            stype = 'VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2'
        elif not 'VkPhysicalDeviceFeatures2' in extends:
            continue

        features = []
        for f in t.iterfind('member'):
            if f.find('type').text == 'VkStructureType':
                stype = f.attrib['values']
            elif f.find('type').text == 'VkBool32':
                fname = f.find('name').text
                if is_base:
                    fname = 'features.' + fname
                features.append(Obj(name = fname))

        core_ver = None
        if res := re.match(r'VkPhysicalDeviceVulkan(\d)(\d)Features', sname):
            core_ver = 'VK_API_VERSION_{0}_{1}'.format(res[1], res[2])

        struct = Obj(name       = sname,
                     stype      = stype,
                     core_ver   = core_ver,
                     is_base    = is_base,
                     features   = features)

        structs.append(struct)
        for f in features:
            featuremap.setdefault(f.name, []).append(struct)

    for s in structs:
        for f in s.features:
            f.replacements = featuremap[f.name]
            core_ver = next(( r.core_ver for r in f.replacements if r.core_ver ), None)
            for r in f.replacements:
                if not r.core_ver:
                    r.max_ver = core_ver

    yield from structs

def find_registry_xml(datadir):
    registry_paths = [
        '{0}/vulkan/registry/vk.xml'.format(datadir),
        '$MINGW_PREFIX/share/vulkan/registry/vk.xml',
        '%VULKAN_SDK%/share/vulkan/registry/vk.xml',
        '$VULKAN_SDK/share/vulkan/registry/vk.xml',
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

    tree = ET.parse(xmlfile)
    registry = VkXML(tree.getroot())
    with open(outfile, 'w') as f:
        f.write(TEMPLATE.render(
            vkresults = get_vkenum(registry, 'VkResult'),
            vkformats = get_vkenum(registry, 'VkFormat'),
            vkspaces  = get_vkenum(registry, 'VkColorSpaceKHR'),
            vkhandles = get_vkenum(registry, 'VkExternalMemoryHandleTypeFlagBits'),
            vkalphas  = get_vkenum(registry, 'VkCompositeAlphaFlagBitsKHR'),
            vktransforms = get_vkenum(registry, 'VkSurfaceTransformFlagBitsKHR'),
            vkdrivers = get_vkenum(registry, 'VkDriverId'),
            vkobjects = get_vkobjects(registry),
            vkstructs = get_vkstructs(registry),
            vkaccess = get_vkaccess(registry),
            vkexts = get_vkexts(registry),
            vkfeatures = get_vkfeatures(registry),
        ))
