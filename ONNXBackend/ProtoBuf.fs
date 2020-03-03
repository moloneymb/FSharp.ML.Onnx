module ProtoBuf

open System.Text
open System.IO
open Onnx
open Google.Protobuf.Collections
open Microsoft.ML.OnnxRuntime.Tensors
open Microsoft.ML.OnnxRuntime

type RepeatedField<'a> with
    static member FromArray(xs : 'a[]) =
        let v = RepeatedField<'a>()
        v.Add(xs)
        v

module RepeatedField =
    let Empty<'a>  = RepeatedField<'a>()

let makeShape(xs : int64[]) = 
    let shape = TensorShapeProto()
    for x in xs do
        shape.Dim.Add(TensorShapeProto.Types.Dimension(DimValue = x))
    shape

type DataType = 
    | FLOAT32 = 1
    | INT64   = 7

let stringAttribute(name:string, value: string) =
    AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.String, S = Google.Protobuf.ByteString.CopyFrom(value, Encoding.UTF8))

let intsAttribute(name:string, ints : int64[]) = 
    let ap = AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.Ints)
    ap.Ints.AddRange(ints)
    ap

let intAttribute(name: string, i: int64) = 
    AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.Int, I = i)

[<AutoOpen>]
module Node = 
    let simple op (name : string, inputs : string[], outputs : string[]) = 
        let np = NodeProto(OpType = op, Name = name)
        np.Input.AddRange(inputs)
        np.Output.AddRange(outputs)
        np

    let unaryOp op (name: string, input: string, output: string) =
        simple op (name, [|input|],[|output|])

    let binaryOp op (name: string, left: string, right: string, output: string) =
        simple op (name, [|left;right|],[|output|])

    let reshape = binaryOp "Reshape"
    let add = binaryOp "Add"

    let cnn(name: string, 
            input: string, 
            kernel: string, 
            output: string, 
            kernel_shape: int64[], 
            strides: int64[], 
            auto_pad: string , 
            group: int64,
            dilations: int64[]) = 
        let np = simple "Conv" (name, [|input;kernel|],[|output|])
        [|
            intsAttribute("kernel_shape", kernel_shape)// [|5L;5L|]
            intsAttribute("strides", strides) // [|1L;1L|]
            stringAttribute("auto_pad", auto_pad) //"SAME_UPPER"
            intAttribute("group",group) // 1L
            intsAttribute("dilations",dilations) //[|1L;1L|]
        |] |> np.Attribute.AddRange
        np

    let pool opType
               (name: string, 
                input: string, 
                output: string, 
                kernel_shape: int64[], 
                strides: int64[], 
                pads: int64[], 
                auto_pad : string) = 
        let np = simple opType (name, [|input|],[|output|])
        [|
            intsAttribute("kernel_shape",kernel_shape)
            intsAttribute("strides",strides)
            intsAttribute("pads",pads)
            stringAttribute("auto_pad",auto_pad)

        |] |> np.Attribute.AddRange
        np

    let maxPool = pool "MaxPool"

    let relu(name: string, input: string, output: string) = unaryOp "Relu" (name, input,output)

    let matmul = binaryOp "MatMul" 


let writeModelToStream(m: Onnx.ModelProto) = 
    let ms = new MemoryStream()
    let cos = new Google.Protobuf.CodedOutputStream(ms)
    m.WriteTo(cos)
    cos.Flush()
    ms.ToArray()

let floatsToBytes(xs : float32[]) = 
    let buffer = Array.zeroCreate<byte> (xs.Length * 4)
    System.Buffer.BlockCopy(xs, 0, buffer, 0, buffer.Length)
    buffer

let bytesToFloats(buffer : byte[]) = 
    let xs= Array.zeroCreate<float32> (buffer.Length / 4)
    System.Buffer.BlockCopy(buffer, 0, xs, 0, buffer.Length)
    xs

let intsToBytes(xs : int64[]) = 
    let buffer = Array.zeroCreate<byte> (xs.Length * 8)
    System.Buffer.BlockCopy(xs, 0, buffer, 0, buffer.Length)
    buffer

let bytesToInts(buffer : byte[]) = 
    let xs= Array.zeroCreate<int64> (buffer.Length / 8)
    System.Buffer.BlockCopy(buffer, 0, xs, 0, buffer.Length)
    xs

type Tensor with
    static member FromTensorProtoFloat32(tp : TensorProto) = 
        let dt = enum<DataType>(tp.DataType)
        if dt <> DataType.FLOAT32 then failwith "unsupported tensor datatype at this time"
        let dims = tp.Dims |> Seq.toArray |> Array.map int32
        let data = 
            if tp.RawData.Length > 0 then tp.RawData.ToByteArray() |> bytesToFloats
            else tp.FloatData |> Seq.toArray
        DenseTensor<float32>(System.Memory<float32>(data),System.ReadOnlySpan<int>(dims))

let getDataType<'a> = 
    let t = typeof<'a>
    match t.FullName with
    | "System.Single" -> DataType.FLOAT32
    | "System.Int64" -> DataType.INT64
    | "System.Int32" -> failwith "todo"
    | _ -> failwithf "Not yet supported %s" t.FullName

let graphToModel(graph: GraphProto) = 
    let mp = 
        ModelProto(DocString = "",
            Domain = "none",
            IrVersion = 3L,
            ModelVersion = 1L,
            ProducerName = "None",
            ProducerVersion = "0.0.0",
            Graph = graph)
    mp.OpsetImport.Add(OperatorSetIdProto(Version = 8L))
    mp

let makeValueInfoProto(name: string, dt: DataType) = 
    ValueInfoProto(Name = name, Type = 
        TypeProto(TensorType = TypeProto.Types.Tensor(ElemType = int32 dt)))

let runSingleOutputNode<'a> (node: NodeProto) (tensors: Tensor<'a>[]) : Tensor<'a> = 
    let dt = getDataType<'a>
    let mp = 
        let graph = GraphProto(Name = "MiniGraph")
        graph.Input.AddRange([|for i in 0..tensors.Length-1 -> makeValueInfoProto(sprintf "Input%i" (i+1), dt)|])
        graph.Output.Add(makeValueInfoProto("Output1", dt))
        graph.Node.AddRange([|node|])
        graph |> graphToModel
    use sess = new InferenceSession(writeModelToStream(mp))
    use res = sess.Run([|for (i,t) in tensors |> Array.indexed -> NamedOnnxValue.CreateFromTensor(sprintf "Input%i" (i+1),t)|])
    // NOTE: I'm expecting this not to leak memory
    res |> Seq.head |> fun x -> x.AsTensor<'a>().Clone()

let buildAndRunUnary<'a> (opName: string) (input1: Tensor<'a>)  =
    runSingleOutputNode (unaryOp opName ("Op", "Input1",  "Output1")) [|input1|]

let buildAndRunBinary<'a> (opName: string) (input1: Tensor<'a>) (input2: Tensor<'a>)  = 
    runSingleOutputNode (binaryOp opName ("Op", "Input1", "Input2", "Output1")) [|input1;input2|]

//type 'a = <uint8 | uint16 |uint32 |uint64 | int8 | int16 | int32 | int64 | float16 | float32 | float64 | string | bool | complex64 | complex128>
//type S = seq<tensor<'a>>
//type T = tensor<'a>
//type I = <int32 | int64>

//S ["seq(tensor(uint8))","seq(tensor(uint16))","seq(tensor(uint32))","seq(tensor(uint64))","seq(tensor(int8))","seq(tensor(int16))","seq(tensor(int32))","seq(tensor(int64))","seq(tensor(float16))","seq(tensor(float))","seq(tensor(double))","seq(tensor(string))","seq(tensor(bool))","seq(tensor(complex64))","seq(tensor(complex128))"]
//T ["tensor(uint8)","tensor(uint16)","tensor(uint32)","tensor(uint64)","tensor(int8)","tensor(int16)","tensor(int32)","tensor(int64)","tensor(float16)","tensor(float)","tensor(double)","tensor(string)","tensor(bool)","tensor(complex64)","tensor(complex128)"]
//I ["tensor(int32)","tensor(int64)"]
//
type ONNX() =
    static member Abs(x) = buildAndRunUnary "Abs" x
    
