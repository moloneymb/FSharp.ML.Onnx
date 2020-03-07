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

//    static member int(name: string, value: int64 option) =
//        value |> Option.bind (fun value -> Attr.int(name,value))

    static member ints(name: string, ints: int64[]) =
        let ap = AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.Ints)
        ap.Ints.AddRange(ints)
        Some(ap)
    static member ints(name: string, value: int64[] option, ?defaultValue: int64[]) =
        value |> Option.orElse defaultValue |> Option.bind (fun value -> Attr.ints(name,value))

let floatAttr(name: string, value: float32) = 
    AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.Float, F = value)

let floatsAttr(name: string, value: float32[]) = 
    let x = AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.Floats)
    x.Floats.AddRange(value)
    x

let stringAttr(name: string, value: string) =
    AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.String, S = Google.Protobuf.ByteString.CopyFrom(value, Encoding.UTF8))

let stringsAttr(name: string, values: string[]) =
    let x = AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.Strings)
    x.Strings.AddRange([|for v in values -> Google.Protobuf.ByteString.CopyFrom(v, Encoding.UTF8)|])
    x

let intsAttr(name:string, ints: int64[]) = 
    let ap = AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.Ints)
    ap.Ints.AddRange(ints)
    ap

let intAttr(name: string, i: int64) = 
    AttributeProto(Name = name, Type = AttributeProto.Types.AttributeType.Int, I = i)

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

        let attrs = [|
            intsAttr("kernel_shape", kernel_shape)// [|5L;5L|]
            intsAttr("strides", strides) // [|1L;1L|]
            stringAttr("auto_pad", auto_pad) //"SAME_UPPER"
            intAttr("group",group) // 1L
            intsAttr("dilations",dilations) //[|1L;1L|]
        |] 

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

        let attrs = [|
            intsAttr("kernel_shape",kernel_shape)
            intsAttr("strides",strides)
            intsAttr("pads",pads)
            stringAttr("auto_pad",auto_pad)
        |] 
        let np = simple opType (name, [|input|],[|output|],attrs)
        np

    let maxPool = pool "MaxPool"

    let relu(name: string, input: string, output: string) = unaryOp "Relu" [||] (name, input,output)

    let matmul = binaryOp "MatMul"  [||]


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

let buildAndRunUnary<'a> (opName: string) (input1: Tensor<'a>) (attrs: AttributeProto[])  =
    runSingleOutputNode (unaryOp opName attrs ("Op", "Input1",  "Output1" )) [|input1|]


let buildAndRunBinary<'a> (opName: string) (input1: Tensor<'a>) (input2: Tensor<'a>) (attrs: AttributeProto[])  = 
    runSingleOutputNode (binaryOp opName attrs ("Op", "Input1", "Input2", "Output1")) [|input1;input2|]

let execNode<'a> (opName: string) (inputs: Tensor<'a>[]) (attrs: AttributeProto[]) =
    runSingleOutputNode (simple opName ("Op",(inputs |> Array.mapi (fun i _ -> sprintf "Input%i" i)),[|"Output1"|],attrs)) inputs

//type 'a = <uint8 | uint16 |uint32 |uint64 | int8 | int16 | int32 | int64 | float16 | float32 | float64 | string | bool | complex64 | complex128>
//type S = seq<tensor<'a>>
//type T = tensor<'a>
//type I = <int32 | int64>

//S ["seq(tensor(uint8))","seq(tensor(uint16))","seq(tensor(uint32))","seq(tensor(uint64))","seq(tensor(int8))","seq(tensor(int16))","seq(tensor(int32))","seq(tensor(int64))","seq(tensor(float16))","seq(tensor(float))","seq(tensor(double))","seq(tensor(string))","seq(tensor(bool))","seq(tensor(complex64))","seq(tensor(complex128))"]
//T ["tensor(uint8)","tensor(uint16)","tensor(uint32)","tensor(uint64)","tensor(int8)","tensor(int16)","tensor(int32)","tensor(int64)","tensor(float16)","tensor(float)","tensor(double)","tensor(string)","tensor(bool)","tensor(complex64)","tensor(complex128)"]
//I ["tensor(int32)","tensor(int64)"]
//


type X() =

    static member F (x: int, [<System.ParamArray>] xs : int[]) = 
        printfn "Args %A" xs
        printfn "X %A" x


        //failwith<int> "err"
    //static member F ([<System.ParamArray>] xs : Tensor<int>, ?x : Tensor<int>, ?y:string) = 
        //let attrs = [|Attr.string("x",y);Some(stringAttr("xx",""));y |> Option.map (fun x -> stringAttr("x",x))|] |> Array.choose id
        //buildAndRunUnary "Abs" x attrs
//        failwith "todo"

//let xs: int[] = 
//    failwith "todo"

//X.F([|1;2;3|])    
//X.F([|1;2|],2)    
//X.F(1,2)
//X.F(1)
