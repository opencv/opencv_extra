# Generate test samples for protobuf parser module.
# Every test sample is a set of
#   1. compiled .proto in binary format
#   2. JSON file with data
#   2. protocol buffer in binary respresentation
# Expected call at opencv_extra/testdata/protobuf_parser.
import os
import fnmatch
import argparse
from google.protobuf import json_format

parser = argparse.ArgumentParser(description='Generate test samples for '
                                             'protobuf parser module')
parser.add_argument('-p', dest='protoc', default='protoc',
                    help='Path to protobuf compiler')
args = parser.parse_args()

# Compile .proto files into binary format and python modules.
dst = os.path.join('bin', '')

for name in os.listdir('.'):
    if fnmatch.fnmatch(name, '*.proto'):
        basename = name[:name.rfind('.')]
        os.system('%s --descriptor_set_out=%s.pb %s' % (args.protoc, dst + basename, name))
        os.system('%s --python_out=. %s' % (args.protoc, name))

# Import compiled modules
from test_pb2 import *
from test_package_pb2 import *
from test_proto3_pb2 import *

import json

def del_empty(d):
    for key, value in d.items():
        if isinstance(value, dict):
            if value:
                del_empty(value)
            else:
                del d[key]
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    del_empty(item)

def gen(prefix, msg):
    with open('test_data.json', 'rt') as f:
        obj = json.load(f)[prefix]
        del_empty(obj)
        json_format.Parse(json.dumps(obj), msg)
    with open(dst + prefix + '.pb', 'wb') as f:
        f.write(msg.SerializeToString())

gen('simple_values', SimpleValues())
gen('nested_message', HostMsg())
gen('default_values', DefaultValues())
gen('enums', EnumValues())
gen('packed_values', PackedValues())
gen('package', MessageTwo())
gen('map', Map())
