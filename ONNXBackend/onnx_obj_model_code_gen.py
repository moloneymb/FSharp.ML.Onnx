from onnx import defs
from onnx import AttributeProto
from onnx.defs import OpSchema, ONNX_DOMAIN, ONNX_ML_DOMAIN

import proto3_pb2
import python.onnx as onnx1

import onnx_schema.proto3_pb2 as onnx_schema

dir(onnx_schema)

import onnx
dir(onnx)
import onnx_schema

import onnx_schema.proto3_pb2

dir(onnx_schema)


import python.onnx_schema.proto3_pb2

#dir(onnx_schema)


import python.onnx_schema.proto3_pb2


schemas = defs.get_all_schemas_with_history()

names = [x.name for x in schemas]

schemas[0].file

def wrap(x,y):
    def f(z):
        return x + z + y
    return f


quote = wrap("\"","\"")
quote = wrap("\"","\"")
record = wrap("{","}")

def getBool(x):
    return "true" if x else "false"


def concatList(xs):
    return array(",".join([quote(str(x)) for x in xs]))

def getAttribute(name,x):
    attr = record(f' name = "{x.name}"; description = "{x.description}"; ``type`` = {x.type}; default_value = AttributeProto(); required = {getBool(x.required)}')
    return f'("{name}",{attr})'

def depth(i):
    return "    " * i

def getFormalParameter(x):
    return record(f"name = {quote(x.name)}\n\r types = {concatList(x.types)} typeStr = {quote(x.typeStr)} description ={quote(x.description)} option = {x.option} isHomogeneous = {formatBool(x.isHomogeneous)}")

def getTypeConstraintParam(x):
    return record(f'type_param_str = "{x.type_param_str}"; description = "{x.description}"; allowed_types_strs = {concatList(const.allowed_type_strs)}')

#def getOpSchema(x):
    "\n    ".join([

    ])

x = convSchema

print("\n".join(map(lambda x: depth(2) + x, )))

def mergeDepth(i,xs):
    return "\n".join(map(lambda x: depth(i) + x, xs))

print(y)

y = "\n".join([
        mergeDepth(1, [
                f'file = "{x.file}"',
                f'line = {x.line}',
                f'support_level = {"SupportType.COMMON" if convSchema.support_level == OpSchema.SupportType.COMMON else "SupportType.EXPEREMENTAL"}',
                f'doc = @"{x.doc}"',
                f'since_version = {x.since_version}',
                f'deprecated = {getBool(x.deprecated)}',
                f'domain = "{x.domain}"',
                f'name = "{x.name}"',
                f'min_input = {x.min_input}',
                f'max_input = {x.max_input}',
                f'min_output = {x.min_output}',
                f'max_output = {x.max_output}',
                f'attributes = [|'
            ]),
        mergeDepth(2,[getAttribute(x,y) for (x,y) in convSchema.attributes.items()]),
        mergeDepth(1,['|]']),
        mergeDepth(1,[f'inputs = [|']),
        mergeDepth(2,[getFormalParameter(x) for x in convSchema.inputs]),
        mergeDepth(1,[ f'|]', f'outputs = [|']),
        mergeDepth(2,[getFormalParameter(x) for x in convSchema.outputs]),
        mergeDepth(1,[f'|]']),
        mergeDepth(1,[ f'type_constraints = ', f'[|']),
        mergeDepth(2,[getTypeConstraintParam(x) for x in convSchema.type_constraints]),
        mergeDepth(1,[ f'|]']), 
        mergeDepth(1,[ f'has_type_and_shape_inference_function = {x.has_type_and_shape_inference_function}'])
        ])

print(y)

convSchema.outputs

"a" 
+ "b"

list(map(lambda x: depth(1) + x, ["1","1"]))

convSchema.SupportType.COMMON
convSchema.SupportType.EXPERIMETNAL



OpSchema.SupportType.EXPEREMENTAL

#convSchema.inputs[0].types
#convSchema.inputs[0].typeStr


OpSchmea.SupportType.COMMON


convSchema = next(x for x in schemas if x.name == "Conv")
convSchema.file
convSchema.line
convSchema.support_level #OpSchema.SupportType.COMMON
convSchema.doc
convSchema.since_version
convSchema.deprecated
convSchema.domain
convSchema.name
convSchema.min_input
convSchema.max_input
convSchema.min_output
convSchema.max_output

# Attribute
attr = convSchema.attributes["auto_pad"]
attr.name
attr.description
attr.type
attr.default_value
attr.required

input0 = convSchema.inputs[0]
# FormalParameters
input0.name
input0.types
input0.typeStr
input0.typeStr
input0.description
input0.option
input0.isHomogeneous

# convSchema.outputs # This is the same...
const = convSchema.type_constraints[0]
const.type_param_str
const.description
const.allowed_type_strs
getTypeConstraintParam(const)








