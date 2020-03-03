# NOTE: This was run in Linux, it may be difficult to get onnx to work in Windows
# NOTE: This is only API code generation, quotations transforms will need some other object model

# TODO: base64 encode default values

from onnx import defs
from onnx import AttributeProto
from onnx.defs import OpSchema, ONNX_DOMAIN, ONNX_ML_DOMAIN

AttrType = OpSchema.AttrType

# TODO filter out earlier version ops

def getSchemas():
    schemas = [x for x in defs.get_all_schemas_with_history() if not x.deprecated]
    max_version = {}
    for x in schemas:
        if x.name in max_version:
            max_version[x.name] = max(max_version[x.name],x.since_version)
        else:
            max_version[x.name] = x.since_version
    return [x for x in schemas if x.since_version == max_version[x.name]]

schemas = getSchemas()
unary_schemas = [schema for schema in schemas if (schema.min_input == 1) and (schema.max_input == 1) and (schema.min_output == 1) and (schema.max_output == 1)]
binary_schemas = [schema for schema in schemas if (schema.min_input == 2) and (schema.max_input == 2) and (schema.min_output == 1) and (schema.max_output == 1)]


#174
#93,28
len(unary_schemas)
len(binary_schemas)

[x.name for x in binary_schemas]
[x.name for x in unary_schemas]

[x for x in unary_schemas if x.name == "ReduceMax"][0].attributes['axes'].type

AttrType.INTS



#TODO all unary
#TODO all binary



schema = [x for x in schemas if x.name == "Softmax"][0]

#[x.name for x in schemas]

#convSchema.min_input
#convSchema.max_input
#convSchema.min_output
#convSchema.max_output


fo = open("/mnt/c/EE/Git/ONNXBackend/ONNXBackend/ONNXAPI.g.fs","w")

fo.write("module ONNXAPI\n")

def wrap(x,y):
    def f(z):
        return x + z + y
    return f

quote = wrap("\"","\"")
quote = wrap("\"","\"")
record = wrap("{","}")

def getBool(x):
    return "true" if x else "false"


def getStrings(xs):
    return wrap("[","]")(",".join([quote(str(x)) for x in xs]))


getStrings(["a","b"])

fo.flush()
fo.close()


# This is for diagnostics
def print_schema(x):
    print(x.name)
    print("(%i,%i) -> (%i,%i)" % (x.min_input, x.max_input, x.min_output, x.max_output))
    print("version %i" % x.since_version)
    print("Attrs")
    for (key,attr) in x.attributes.items():
        print("%s %s %s" % (attr.name, attr.type, ("r" if attr.required else "")))
    def print_formal_parameter(fp):
        print("%s %s %s %s %s" % (fp.name, getStrings(fp.types), fp.typeStr, fp.option, ("t" if fp.isHomogeneous else "f")))
    print("Inputs")
    for y in x.inputs:
        print_formal_parameter(y)
    print("Outputs")
    for y in x.outputs:
        print_formal_parameter(y)
    print("TypeConstraints")
    for y in x.type_constraints:
        print("%s %s" % (y.type_param_str,getStrings(y.allowed_type_strs)))

print_schema(schema)

#"tensor(uint8)"
#"tensor(uint16)"
#"tensor(uint32)"
#"tensor(uint64)"
#"tensor(int8)"
#"tensor(int16)"
#"tensor(int32)"
#"tensor(int64)"
#"tensor(float16)"
#"tensor(float)"
#"tensor(double)"
#"tensor(string)"
#"tensor(bool)"
#"tensor(complex64)"
#"tensor(complex128)"

unary
        
[len(x.type_constraints) for x in unary_schemas] 

[len(x.type_constraints) for x in unary_schemas]

[len(x.type_constraints) for x in binary_schemas][9]

#TODO handle seq, map, tensor
print_schema(unary_schemas[2])

print_schema(binary_schemas[9])

#special case SequenceAt

#len([x  for x in schemas if len(x.type_constraints) == 3])
#[x.name  for x in schemas if len(x.type_constraints) == 3]

def getSchema(name):
    return [x for x in schemas if x.name == name][0]

#print_schema(getSchema("QLinearConv"))
#print_schema(getSchema("MatMulInteger"))

for x in set(["(%i,%i) -> (%i,%i)" % (x.min_input, x.max_input, x.min_output, x.max_output) for x in schemas]):
    print(x)


#for now filter it down to uint8, int32, int64, float32, bool, string
# [x for x in schemas if x.min_output == 2 and x.max_output == 2]

#set([attr.type for x in schemas for (_,attr) in x.attributes.items()])

# TODO examine AttrType.TENSOR, AttrType.GRAPH, and AttrType.SPARSE_TENSOR

[x.name for x in schemas if x.min_output == 1 and x.max_output == 2]
[x.name for x in schemas if x.min_output == 3 and x.max_output == 3]
[x.name for x in schemas if x.max_output > 1000]
[x.name for x in schemas if x.max_input == 1 and x.max_output > 1000]


list(schemas[0].attributes.items())[0][1].default_value

defaults = [attr.default_value for x in schemas for (_,attr) in x.attributes.items() if attr.default_value.type != 0]

len(defaults)

#set([x.type for x in defaults])

#defaults only use {1, 2, 3, 7, 8}

#Note Attibutes only use FLOAT, INT, STRING, INTS, STRINGS

def filterAttrType(t):
    def f(x):
        return t in set([y.type for (_,y) in x.attributes.items()])
    return f


#[x.name for x in schemas if filterAttrType(AttrType.GRAPH)(x)]
#GRAPH is If, Scan, Loop
#[x.name for x in schemas if filterAttrType(AttrType.TENSOR)(x)]
#TENSOR is ConstantOfShape, Constant
#[x.name for x in schemas if filterAttrType(AttrType.SPARSE_TENSOR)(x)]
#SPARSE_TENSOR is Constant

#[x.attributes.items() for x in  schemas]
#['If', 'Scan', 'Loop', 'Split']

# -> (0,2) "GRU","RNN"
# -> (0,3) "LSTM"

#print_schema(getSchema('LSTM'))
#print_schema(getSchema('GRU'))
# (5,5) -> (1,5)
#print_schema(getSchema('BatchNormalization'))

# TODO figure out how optional inputs result in optional outputs
# AFAIK Optional outputs are always available but don't have to be used

# TODO, how do we deal with Optional outputs such as LSTM, it seems that it depends on the optional inputs. May need to special case it.


# NOTE: These are uncommon
#[x.name for x in schemas if x.min_output == 2 and x.max_output == 2]
#['TreeEnsembleClassifier', 'LinearClassifier', 'SVMClassifier', 'TopK']

print_schema([0])



#["(%i,%i) -> (%i,%i)" % (x.min_input, x.max_input, x.min_output, x.max_output) for x in schemas if x.min_output == 2 and x.max_output == 2]

    print(x)

# Examine 

[0].name

#11*11*15


#Max of 4

#len(schemas)

schema.name

#input0 = convSchema.inputs[0]

## FormalParameters
#input0.name
#input0.types
#input0.typeStr
#input0.description
#input0.option
#input0.isHomogeneous
#
## convSchema.outputs # This is the same...
#const = convSchema.type_constraints[0]
#const.type_param_str
#const.description
#const.allowed_type_strs
#getTypeConstraintParam(const)
#




#convSchema.SupportType.COMMON
#convSchema.SupportType.EXPERIMETNAL
#
#OpSchema.SupportType.EXPEREMENTAL
#
##convSchema.inputs[0].types
##convSchema.inputs[0].typeStr
#
#OpSchmea.SupportType.COMMON
#
#
#convSchema = next(x for x in schemas if x.name == "Conv")
#convSchema.file
#convSchema.line
#convSchema.support_level #OpSchema.SupportType.COMMON
#convSchema.doc
#convSchema.since_version
#convSchema.deprecated
#convSchema.domain
#convSchema.name
#convSchema.min_input
#convSchema.max_input
#convSchema.min_output
#convSchema.max_output
#
## Attribute
#attr = convSchema.attributes["auto_pad"]
#attr.name
#attr.description
#attr.type
#attr.default_value
#attr.required
#
#input0 = convSchema.inputs[0]
## FormalParameters
#input0.name
#input0.types
#input0.typeStr
#input0.typeStr
#input0.description
#input0.option
#input0.isHomogeneous
#
## convSchema.outputs # This is the same...
#const = convSchema.type_constraints[0]
#const.type_param_str
#const.description
#const.allowed_type_strs
#getTypeConstraintParam(const)
#



