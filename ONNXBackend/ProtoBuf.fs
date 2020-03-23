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

type Tensor<'a> with
    member this.ToArray() = 
        this.ToDenseTensor().Buffer.ToArray()

    member this.shape = this.Dimensions.ToArray()


let makeShape(xs : int64[]) = 
    let shape = TensorShapeProto()
    for x in xs do
        shape.Dim.Add(TensorShapeProto.Types.Dimension(DimValue = x))
    shape

[<RequireQualifiedAccess>]
type DataType = 
    | FLOAT32 = 1
    | UINT8 = 2
    | INT8 = 3
    | UINT16 = 4
    | INT16 = 5
    | INT32 = 6
    | INT64   = 7
    | STRING = 8
    | BOOL = 9
    // IEEE754 half-precision floating-point format (16 bits wide).
    // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
    | FLOAT16 = 10

    | DOUBLE = 11
    | UINT32 = 12
    | UINT64 = 13
    | COMPLEX64 = 14     // complex with float32 real and imaginary components
    | COMPLEX128 = 15    // complex with float64 real and imaginary components

    // Non-IEEE floating-point format based on IEEE754 single-precision
    // floating-point number truncated to 16 bits.
    // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    | BFLOAT16 = 16

type Attr() =
    static member float(name: string, value: float32) = 
        Some(AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.Float, F = value))
    static member float(name: string, value: float32 option, ?defaultValue: float32) = 
        value |> Option.orElse defaultValue |> Option.bind (fun value -> Attr.float(name,value))
        
    static member floats(name: string, values: float32[]) = 
        let x = AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.Floats)
        x.Floats.AddRange(values)
        Some(x)
    static member floats(name: string, values: float32[] option, ?defaultValue: float32[]) = 
        values |> Option.orElse defaultValue |> Option.bind(fun values -> Attr.floats(name,values))

    static member string(name: string, value: string) =
        Some(AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.String, S = Google.Protobuf.ByteString.CopyFrom(value, Encoding.UTF8)))

    static member string(name: string, value: string option, ?defaultValue: string) =
        value |> Option.orElse defaultValue |> Option.bind (fun value -> Attr.string(name,value))

    static member strings(name: string, values: string[]) =
        let x = AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.Strings)
        x.Strings.AddRange([|for v in values -> Google.Protobuf.ByteString.CopyFrom(v, Encoding.UTF8)|])
        Some(x)
    static member strings(name: string, values: string[] option, ?defaultValue: string[]) =
        values |> Option.orElse defaultValue |> Option.bind (fun values -> Attr.strings(name,values))

    static member int(name:string, value: int64) =
        Some(AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.Int, I = value))
    static member int(name:string, value: int64 option, ?defaultValue: int64) =
        value |> Option.orElse defaultValue |> Option.bind (fun value -> Attr.int(name,value))

    static member ints(name: string, ints: int64[]) =
        let ap = AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.Ints)
        ap.Ints.AddRange(ints)
        Some(ap)
    static member ints(name: string, value: int64[] option, ?defaultValue: int64[]) =
        value |> Option.orElse defaultValue |> Option.bind (fun value -> Attr.ints(name,value))

    static member tensor(x: Tensor<float32>) : AttributeProto option = 
        let t = TensorProto(DataType = int DataType.FLOAT32)
        t.Dims.AddRange(x.Dimensions.ToArray() |> Array.map int64)
        t.FloatData.AddRange(x.ToArray())
        Some(AttributeProto(Type = AttributeProto.Types.AttributeType.Tensor, T = t))

    static member tensor(x: Tensor<int32>) : AttributeProto option = 
        let t = TensorProto(DataType = int DataType.INT32)
        t.Dims.AddRange(x.Dimensions.ToArray() |> Array.map int64)
        t.Int32Data.AddRange(x.ToArray())
        Some(AttributeProto(Type = AttributeProto.Types.AttributeType.Tensor, T = t))

    static member tensor(x: Tensor<int64>) : AttributeProto option = 
        let t = TensorProto(DataType = int DataType.INT64)
        t.Dims.AddRange(x.Dimensions.ToArray() |> Array.map int64)
        t.Int64Data.AddRange(x.ToArray())
        Some(AttributeProto(Type = AttributeProto.Types.AttributeType.Tensor, T = t))

[<AutoOpen>]
module X =
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

type 'a``[]`` with
    member x.ToTensor() = ArrayTensorExtensions.ToTensor(x)

type Tensor<'a> with
    member x.Reshape(shape:int[]) = 
        x.Reshape(System.ReadOnlyMemory.op_Implicit(shape).Span)

[<AutoOpen>]
module Node = 
    let simple op (name: string, inputs: string[], outputs: string[], attributes: AttributeProto[]) = 
        let np = NodeProto(OpType = op, Name = name)
        np.Attribute.AddRange(attributes)
        np.Input.AddRange(inputs)
        np.Output.AddRange(outputs)
        np

    let unaryOp op (attrs: AttributeProto[]) (name: string, input: string, output: string)  =
        simple op (name, [|input|],[|output|],attrs)

    let binaryOp op (attrs: AttributeProto[]) (name: string, left: string, right: string, output: string)  =
        simple op (name, [|left;right|],[|output|],attrs)

    let reshape = binaryOp "Reshape" [||]
    let add = binaryOp "Add" [||]

    let cnn(name: string, 
            input: string, 
            kernel: string, 
            output: string, 
            kernel_shape: int64[], 
            strides: int64[], 
            auto_pad: string , 
            group: int64,
            dilations: int64[]) = 

        let attrs = 
            [|
                Attr.ints("kernel_shape", kernel_shape)// [|5L;5L|]
                Attr.ints("strides", strides) // [|1L;1L|]
                Attr.string("auto_pad", auto_pad) //"SAME_UPPER"
                Attr.int("group",group) // 1L
                Attr.ints("dilations",dilations) //[|1L;1L|]
            |] |> Array.choose id

        let np = simple "Conv" (name, [|input;kernel|],[|output|],attrs)
        np

    let pool opType
               (name: string, 
                input: string, 
                output: string, 
                kernel_shape: int64[], 
                strides: int64[], 
                pads: int64[], 
                auto_pad : string) = 

        let attrs = 
            [|
                Attr.ints("kernel_shape",kernel_shape)
                Attr.ints("strides",strides)
                Attr.ints("pads",pads)
                Attr.string("auto_pad",auto_pad)
            |] |> Array.choose id
        let np = simple opType (name, [|input|],[|output|],attrs)
        np

    let maxPool = pool "MaxPool"
    let relu(name: string, input: string, output: string) = unaryOp "Relu" [||] (name, input,output)
    let matmul = binaryOp "MatMul"  [||]


let getDataType(t: System.Type) =
    match t.FullName with
    | "System.Single" -> DataType.FLOAT32
    | "System.Double" -> DataType.DOUBLE
    | "System.SByte" ->  DataType.INT8
    | "System.Int16" -> DataType.INT16
    | "System.Int32" -> DataType.INT32
    | "System.Int64" -> DataType.INT64
    | "System.Byte" ->  DataType.UINT8
    | "System.UInt16" -> DataType.UINT16
    | "System.UInt32" -> DataType.UINT32
    | "System.UInt64" -> DataType.UINT64
    | "System.Boolean" -> DataType.BOOL
    | "System.String" -> DataType.STRING
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

let private makeValueInfoProto(name: string, dt: DataType) = 
    ValueInfoProto(Name = name, Type = 
        TypeProto(TensorType = TypeProto.Types.Tensor(ElemType = int32 dt)))

let writeModelToStream(m: Onnx.ModelProto) = 
    let ms = new MemoryStream()
    let cos = new Google.Protobuf.CodedOutputStream(ms)
    m.WriteTo(cos)
    cos.Flush()
    ms.ToArray()

let miniGraph(opName,  inputs : (NamedOnnxValue*ValueInfoProto)[], outputs , attrs) = 
    let node = (simple opName ("Op",(inputs |> Array.map (fun (x,_) -> x.Name)),(outputs |> Array.map fst),attrs)) 
    let graph = GraphProto(Name = "MiniGraph")
    graph.Input.AddRange(inputs |> Array.map snd)
    graph.Output.Add(outputs |> Array.map (fun (x,dt) -> makeValueInfoProto(x, dt)))
    graph.Node.AddRange([|node|])
    new InferenceSession(writeModelToStream(graph |> graphToModel))

type MV() = 
    let mutable i = 1
    member this.c(x: Tensor<'a>) = 
        i <- i + 1
        MV.mv(i,x)
    member this.c(x: Tensor<'a> option) = x |> Option.map this.c
    member this.c(xs: Tensor<'a>[]) = xs |> Array.map this.c
    static member mv(i:int , x: Tensor<'a>) =
        let dt = getDataType(typeof<'a>)
        let name = sprintf "I%i" i 
        (NamedOnnxValue.CreateFromTensor(name,x), makeValueInfoProto(name,dt))

let execNodeMany<'a> (opName: string) (inputs: (NamedOnnxValue*ValueInfoProto)[]) (outputCount: int)  (attrs: AttributeProto[]) : Tensor<'a>[] =
    let dt = getDataType(typeof<'a>)
    use sess = miniGraph(opName, inputs,(Array.init outputCount (fun i -> (sprintf "O%i" i, dt))),attrs)
    use res = sess.Run(inputs |> Array.map fst)
    [|for x in res -> x.AsTensor<'a>().Clone()|]

let execNode<'a> (opName: string) (inputs: (NamedOnnxValue*ValueInfoProto)[]) (attrs: AttributeProto[]) : Tensor<'a> =
    let dt = getDataType(typeof<'a>)
    use sess = miniGraph(opName, inputs,[|"O1", dt|],attrs)
    use res = sess.Run(inputs |> Array.map fst)
    res |> Seq.head |> fun x -> x.AsTensor<'a>().Clone()

let execNodeCheck<'a> (opName: string) (inputs: (NamedOnnxValue*ValueInfoProto)[]) (ids: int64[]) (attrs: AttributeProto[]) : Tensor<'a> =
    let dt = getDataType(typeof<'a>)
    if (ids |> Array.contains (int64 dt)) then
        failwithf "The data type %A is not permitted by this operation" dt
    use sess = miniGraph(opName, inputs,[|"O1", dt|],attrs)
    use res = sess.Run(inputs |> Array.map fst)
    res |> Seq.head |> fun x -> x.AsTensor<'a>().Clone()
    

let execNodeTuple2<'a,'b> (opName: string) (inputs: (NamedOnnxValue*ValueInfoProto)[]) (attrs: AttributeProto[]) : (Tensor<'a>*Tensor<'b>) =
    let outputNames = [|
        "O1A",getDataType(typeof<'a>);
        "O1B",getDataType(typeof<'b>)|]
    use sess = miniGraph(opName, inputs,outputNames,attrs)
    use res = sess.Run(inputs |> Array.map fst)
    res |> Seq.toArray |> fun ress -> (ress.[0].AsTensor<'a>().Clone(), 
                                       ress.[1].AsTensor<'b>().Clone())

let execNodeTuple3<'a,'b,'c> (opName: string) (inputs: (NamedOnnxValue*ValueInfoProto)[]) (attrs: AttributeProto[]) : (Tensor<'a>*Tensor<'b>*Tensor<'c>) =
    let outputNames = [|
        "O1A",getDataType(typeof<'a>);
        "O1B",getDataType(typeof<'b>);
        "O1C",getDataType(typeof<'c>)|]
    use sess = miniGraph(opName, inputs,outputNames,attrs)
    use res = sess.Run(inputs |> Array.map fst)
    res |> Seq.toArray |> fun ress -> (ress.[0].AsTensor<'a>().Clone(), 
                                       ress.[1].AsTensor<'b>().Clone(), 
                                       ress.[2].AsTensor<'c>().Clone())

let execNodeTuple4<'a,'b,'c,'d> (opName: string) (inputs: (NamedOnnxValue*ValueInfoProto)[]) (attrs: AttributeProto[]) : (Tensor<'a>*Tensor<'b>*Tensor<'c>*Tensor<'d>) =
    let outputNames = [|
        "O1A",getDataType(typeof<'a>);
        "O1B",getDataType(typeof<'b>);
        "O1C",getDataType(typeof<'c>);
        "O1D",getDataType(typeof<'d>); |]
    use sess = miniGraph(opName, inputs,outputNames,attrs)
    use res = sess.Run(inputs |> Array.map fst)
    res |> Seq.toArray |> fun ress -> (ress.[0].AsTensor<'a>().Clone(), 
                                       ress.[1].AsTensor<'b>().Clone(), 
                                       ress.[2].AsTensor<'c>().Clone(), 
                                       ress.[3].AsTensor<'d>().Clone())

let execNodeTuple5<'a,'b,'c,'d,'e> (opName: string) (inputs: (NamedOnnxValue*ValueInfoProto)[]) (attrs: AttributeProto[]) : (Tensor<'a>*Tensor<'b>*Tensor<'c>*Tensor<'c>*Tensor<'e>) =
    let outputNames = [|
        "O1A",getDataType(typeof<'a>);
        "O1B",getDataType(typeof<'b>);
        "O1C",getDataType(typeof<'c>);
        "O1D",getDataType(typeof<'d>);
        "O1E",getDataType(typeof<'e>); |]
    use sess = miniGraph(opName, inputs,outputNames,attrs)
    use res = sess.Run(inputs |> Array.map fst)
    res |> Seq.toArray |> fun ress -> (ress.[0].AsTensor<'a>().Clone(), 
                                       ress.[1].AsTensor<'b>().Clone(), 
                                       ress.[2].AsTensor<'c>().Clone(), 
                                       ress.[3].AsTensor<'c>().Clone(), 
                                       ress.[4].AsTensor<'e>().Clone())

