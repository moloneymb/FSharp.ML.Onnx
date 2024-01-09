module FSharp.ML.Onnx.Protobuf

// These are helper functions that are used by the generated code
open System
open System.Text
open System.IO
open Onnx
open Google.Protobuf.Collections
open Microsoft.ML.OnnxRuntime.Tensors
open Microsoft.ML.OnnxRuntime


module Array = 
    let mapSnd (f: 'b -> 'c) (xs:('a*'b)[]) = xs |> Array.map (fun (x,y : 'b) -> (x,f y))
    let mapFst (f: 'a -> 'c) (xs:('a*'b)[]) = xs |> Array.map (fun (x,y) -> (f x,y))
    let groupByFst (xs:('a*'b)[]) = xs |> Array.groupBy fst |> mapSnd (Array.map snd)
    let groupBySnd (xs:('a*'b)[]) = xs |> Array.groupBy snd |> mapSnd (Array.map fst)

type Tensor<'a> with
    member this.ToArray() = this.ToDenseTensor().Buffer.ToArray()
    member this.shape = this.Dimensions.ToArray()
    member x.Reshape(shape:int[]) = x.Reshape(System.ReadOnlyMemory.op_Implicit(shape).Span)

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

let tryDataTypeToType (x:DataType) = 
    match x with
    | DataType.FLOAT32 -> Some typeof<float32>
    | DataType.UINT8 -> Some typeof<uint8>
    | DataType.INT8 -> Some typeof<int8>
    | DataType.UINT16 -> Some typeof<uint16>
    | DataType.INT16 -> Some typeof<int16>
    | DataType.INT32 -> Some typeof<int32>
    | DataType.INT64  -> Some typeof<int64>
    | DataType.STRING -> Some typeof<string>
    | DataType.BOOL -> Some typeof<bool>
    | DataType.FLOAT16 -> None //typeof<float16>
    | DataType.DOUBLE -> Some typeof<double>
    | DataType.UINT32 -> Some typeof<uint32>
    | DataType.UINT64 -> Some typeof<uint64>
    | DataType.COMPLEX64 -> None //Some typeof<System.Numerics.Complex>
    | DataType.COMPLEX128 ->Some typeof<System.Numerics.Complex> 
    | DataType.BFLOAT16 -> None
    | _ -> None

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

    static member tensor(x: Tensor<double>) : AttributeProto option = 
        let t = TensorProto(DataType = int DataType.FLOAT32)
        t.Dims.AddRange(x.Dimensions.ToArray() |> Array.map int64)
        t.DoubleData.AddRange(x.ToArray())
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

let simple op (name: string, inputs: string[], outputs: string[], attributes: AttributeProto[],domain:string option) = 
    let np = NodeProto(OpType = op, Name = name)
    np.Attribute.AddRange(attributes)
    np.Input.AddRange(inputs)
    np.Output.AddRange(outputs)
    domain |> Option.iter (fun x -> np.Domain <- x)
    np

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
            IrVersion = 6L,
            ModelVersion = 1L,
            ProducerName = "None",
            ProducerVersion = "0.0.0",
            Graph = graph)
    mp.OpsetImport.Add(OperatorSetIdProto(Version = 14L))
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
    let node = (simple opName ("Op",(inputs |> Array.map (fun (x,_) -> x.Name)),(outputs |> Array.map fst),attrs,None)) 
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

type ValueInfo = {name : string; dt : DataType}

type Graph = 
    { mutable ops : NodeProto list; mutable usedNames : Map<string,int> } 
    static member Default() = {ops = []; usedNames = Map.empty}
    member this.GetName(name : string) : string = 
            let (x,y) = 
                match this.usedNames.TryFind(name) with
                | None -> name,this.usedNames.Add(name,0)
                | Some(v) -> 
                    let newName = name + string(v + 1)
                    newName,this.usedNames.Add(name,v+1).Add(newName,0)
            this.usedNames <- y
            x

    member this.AddNode(node: NodeProto) = this.ops <- node::this.ops
    member this.AddNode(name: string, inputs: ValueInfo [], outputs: DataType[], attrs: AttributeProto option[],?domain:string) =
            let outputValueInfos = outputs |> Array.map (fun dt -> { name = this.GetName(sprintf "%s_Output" name); dt = dt})
            this.AddNode(simple name (this.GetName(name), (inputs |> Array.map (fun x -> x.name)), (outputValueInfos |> Array.map (fun x -> x.name)), attrs |> Array.choose id,domain))
            outputValueInfos

// Helper functions for converting arrays to tuples
let toTuple1 (xs:'a[]) = xs.[0]
let toTuple2 (xs:'a[]) = (xs.[0],xs.[1])
let toTuple3 (xs:'a[]) = (xs.[0],xs.[1],xs.[2])
let toTuple4 (xs:'a[]) = (xs.[0],xs.[1],xs.[2],xs.[3])
let toTuple5 (xs:'a[]) = (xs.[0],xs.[1],xs.[2],xs.[3],xs.[4])

type Constants() =
    static member constant(graph: Graph, t: Tensor<float32>) : ValueInfo =
        graph.AddNode("Constant", [||], [|DataType.FLOAT32|],[|Attr.tensor(t)|]).[0]

    static member constant(graph: Graph, t: Tensor<double>) : ValueInfo =
        graph.AddNode("Constant", [||], [|DataType.DOUBLE|],[|Attr.tensor(t)|]).[0]

    static member constant(graph: Graph, t: Tensor<int32>) : ValueInfo =
        graph.AddNode("Constant", [||], [|DataType.INT32|],[|Attr.tensor(t)|]).[0]

    static member constant(graph: Graph, t: Tensor<int64>) : ValueInfo =
        graph.AddNode("Constant", [||], [|DataType.INT64|],[|Attr.tensor(t)|]).[0]


type Onnx.ModelProto with
  member this.ToArray() =
    let ms = new MemoryStream()
    let cos = new Google.Protobuf.CodedOutputStream(ms)
    this.WriteTo(cos)
    cos.Flush()
    ms.ToArray()

type Google.Protobuf.Collections.RepeatedField<'a> with
  member this.Replace(before:'a,after:'a) : bool =
    let index = this.IndexOf(before)
    if this.Remove(before) then
      this.Insert(index,after) // tested ok
      true
    else false

type GraphProto with

  member this.TransposeShape(target:ValueInfoProto,transpose:int[]) = 
      let shapeClone = target.Type.TensorType.Shape.Clone()
      target.Type.TensorType.Shape.Dim.Clear()
      for t in transpose do 
        target.Type.TensorType.Shape.Dim.Add(shapeClone.Dim.[t])

  member this.TransposeInputShape([<ParamArray>] shapeIndexs:(int*int[])[]) = for (indexNumber,ts) in shapeIndexs do this.TransposeShape(this.Input[indexNumber],ts)
  member this.TransposeInputShape(transpose:int[]) = this.TransposeInputShape([|1,transpose|])
  member this.TransposeOutputShape([<ParamArray>] shapeIndexs:(int*int[])[]) = for (indexNumber,ts) in shapeIndexs do this.TransposeShape(this.Output[indexNumber],ts)
  member this.TransposeOutputShape(transpose:int[]) = this.TransposeOutputShape([|1,transpose|])

  member this.GetUniqueNodeName(name:string) = 
    let names : Set<string> = this.NodeNames
    seq { 1 .. 10_000 } |> Seq.map (sprintf "%s_%i" name) |> Seq.filter (names.Contains >> not) |> Seq.tryHead
    |> function 
    | None -> failwith "unable to find unique name in 10K candidates"
    | Some(x) -> x

  member this.SimpleSummary() = 
    let inputs = this.Input |> Seq.map (fun x -> x.Name) |> String.concat ","  
    let outputs = this.Output |> Seq.map (fun x -> x.Name) |> String.concat ","  
    sprintf "%s --> %s" inputs outputs


  member this.ToDot(?useNodeNames:bool,?elideConstants) =
    // TODO:
      // * uniquly color each node (perhaps by type?) [shape=circle, style=filled, fillcolor=purple]
      // * each edge can have a tooltip name
      // * we can label the edges a4 -> a6 [label="  ordinary edge label"] 
      // * could use a triplet for edge color color="blue;0.5:red" where the colors transform from one to the other (we could use the destination edge color)
      //   color can be hexs e.g. #ffffff

    let useNodeNames = defaultArg useNodeNames false
    let elideConstants = defaultArg elideConstants false

    let getUniqueOp = 
      let mutable existingNames = Set.empty<string>
      fun (name:string) ->
        seq {yield name ; yield! seq { 1 .. 100_000 } |> Seq.map (sprintf "%s_%i" name)} |> Seq.filter (existingNames.Contains >> not) 
        |> Seq.tryHead 
        |> function
        | None -> failwith "unable to find new name"
        | Some(x) -> existingNames <- existingNames.Add(x); x

    let nodeNameMap = [|
                          for node in this.Node -> node.Name,getUniqueOp(node.OpType)
                          yield "GraphInput","GraphInput"
                          yield "GraphOutput","GraphOutput"
                      |] |> Map.ofArray

    let nodeMap = [| for node in this.Node do 
                      if node.OpType = "Constant" && elideConstants then ()
                      else yield node.Name,(node.Input |> Seq.toArray, node.Output |> Seq.toArray)
                     yield ("GraphInput", ([||], [|for x in this.Input -> x.Name|]))
                     yield ("GraphOutput", ([|for x in this.Output-> x.Name|],[||]))
                      |] |> Map.ofArray
    let insMap = [| for KeyValue(n,(ins,outs)) in nodeMap do for x in ins -> (x,n)|] |> Array.groupByFst |> Map.ofArray
    
    let f (name:string) = if useNodeNames then name.Replace('/','_') else nodeNameMap.[name]

    [|
      yield "digraph G {"
      for KeyValue(k,(_,outs)) in nodeMap do
        match (outs |> Array.choose insMap.TryFind |> Array.collect id) with
        | [||] -> ()
        | xs -> yield sprintf "%s -> { %s }" (f(k)) (xs |> Array.map f |> String.concat " ")
      if not useNodeNames then
        for KeyValue(k,(_,outs)) in nodeMap do
          yield sprintf "%s [tooltip=\"%s\"]" nodeNameMap.[k] k
      yield "}"
    |] |> String.concat "\n"

  member this.ReplaceInitializersWithConstants() = 
    let tensorProtos = this.Initializer |> Seq.toArray
    this.Initializer.Clear()
    for tp in tensorProtos do 
      // NOTE output is always the same, keeping the name should obviate the need to update the inputs...
      let constNode = 
        let att = [|AttributeProto(Type = AttributeProto.Types.AttributeType.Tensor, T = tp)|]
        simple "Constant" (this.GetUniqueNodeName("Constant"), [||], [|tp.Name|], att,None)
      this.Node.Add(constNode)

  member this.NodeNames = set [|for node in this.Node -> node.Name|]
  member this.LinkNames = set [|for node in this.Node do yield! node.Output; yield! node.Input|]

  member this.RenameLink(before:string,after:string,onInputs:bool, onOutputs:bool, ?preventCollision:bool) =
    let preventCollision = defaultArg preventCollision true
    if not(this.LinkNames.Contains(before)) then
      failwithf "link %s is not in graph" before
    elif preventCollision && this.LinkNames.Contains(after) then
      failwithf "link %s already exists in the graph" after
    for node in this.Node do
      if onInputs then node.Input.Replace(before,after) |> ignore<bool>
      if onOutputs then node.Output.Replace(before,after) |> ignore<bool>

  member this.RenameNodeName(before:string,after:string) =
    if (not (this.NodeNames.Contains(before))) then
      failwithf "node %s is not in the graph" before
    elif (this.NodeNames.Contains(after)) then
      failwithf "node %s already exists in the graph" after
    else
      let node = this.Node |> Seq.find (fun x -> x.Name = before)
      node.Name <- after

  /// NOTE: this does appear to work...
  /// NOTE: f and links transform should be well defined...
  member this.Rename(f:string->string,?links:string->string) =
    let links = defaultArg links f
    for node in this.Node do
      node.Name <- f(node.Name)
      let inputs = node.Input |> Seq.toArray
      node.Input.Clear()
      node.Input.AddRange(inputs |> Array.map links)
      let outputs = node.Output |> Seq.toArray
      node.Output.Clear()
      node.Output.AddRange(outputs |> Array.map links)
    for input in this.Input do
      input.Name <- links input.Name
    for output in this.Output do
      output.Name <- links output.Name

  member this.TryFindNodeByName([<ParamArray>]names:string[]) : NodeProto option [] = 
    [|for name in names do this.Node |> Seq.tryFind (fun x -> x.Name = name)|]

  member this.FindNodeByName(name:string) = this.Node |> Seq.find (fun x -> x.Name = name)

  member this.NamesByOpType(opType:string) = [| for node in this.Node do if node.OpType = opType then node.Name|]
  member this.NamesByInput(input:string) = [| for node in this.Node do if node.Input.Contains(input) then node.Name|]
  member this.TryNameByOutput(output:string) = [| for node in this.Node do if node.Output.Contains(output) then node.Name|] |> Seq.tryHead

  member this.TryRemoveNodeRecursive([<ParamArray>] names:string[]) =
    for node in  this.TryFindNodeByName(names) |> Array.choose id do this.Node.Remove(node) |> ignore<bool>
    let mutable flag = false
    while not flag do
      match [|yield! this.FindAllOrphanedNodes(true); yield! this.FindAllDeadEndNodes(true)|] |> Array.map fst |> Array.distinct with
      | [||] -> flag <- true
      | xs -> 
        for name in xs do
          this.Node.Remove(this.FindNodeByName(name)) |> ignore<bool>

  member this.FindAllOrphanedNodes(?fully:bool) : (string*string[])[] =
    let fully = defaultArg fully false
    // get all outputs
    let outputs = 
      Set [|
            for input in this.Input do yield input.Name; 
            for output in this.Node do yield! output.Output; 
    // get all outputs
            for x in this.Initializer do yield x.Name|]
    [| 
      for node in this.Node do
        let ins = node.Input |> Seq.toArray
        match ins |> Array.filter (outputs.Contains >> not)  with
        | [||] -> ()
        | missing_ins -> 
          if fully then if ins.Length = missing_ins.Length then yield node.Name,missing_ins
          else yield node.Name,missing_ins
    |]

  member this.FindAllDeadEndNodes(?fully:bool) : (string*string[])[] =
    let fully = defaultArg fully false
    let inputs = 
      Set [|
            for input in this.Output do yield input.Name; 
            for output in this.Node do yield! output.Input; |]
    [| 
      for node in this.Node do
        let outs = node.Output |> Seq.toArray
        match outs |> Array.filter (inputs.Contains >> not)  with
        | [||] -> ()
        | missing_outs -> 
          if fully then if outs.Length = missing_outs.Length then yield node.Name,missing_outs
          else yield node.Name,missing_outs
    |]

  member this.TryFindDesendent(initNode:NodeProto,f:NodeProto -> bool) : (NodeProto * string[])[] =
    let rec tryFindDesendent(node:NodeProto,path:string list) : (NodeProto * string list)[]  =
      [|
          if f node && node <> initNode then yield(node,path)
          else
            for output in node.Output do
              for node in this.NamesByInput(output) |> Array.map this.FindNodeByName do
                yield! tryFindDesendent(node,output :: path)
      |]
    tryFindDesendent(initNode,[]) |> Array.mapSnd (fun ys -> ys |> List.toArray |> Array.rev)

  member this.ShortCircuit(path:string[]) =
    if path.Length > 1 then
      let firstOutput = path[0]
      let lastInput = path.[path.Length-1]
      for nodes in this.NamesByInput(lastInput) |> Array.map this.FindNodeByName do
        nodes.Input.Replace(lastInput,firstOutput) |> ignore<bool>

  member this.Inject(startNode:NodeProto,endNode:NodeProto,singleNode:NodeProto) : bool =
    let ins,outs = singleNode.Input |> Seq.toArray, singleNode.Output |> Seq.toArray
    match ins,outs with
    | [|inName|],[|outName|] -> 
      if not(this.Node.Contains(singleNode)) then this.Node.Add(singleNode)
      this.Inject(startNode,endNode,inName,outName)
    | _ -> false

  member this.Inject(startNode:NodeProto,endNode:NodeProto,startInjectInput:string,endInjectOutput:string) : bool =
    let check = this.LinkNames |> fun x -> x.Contains(startInjectInput) && x.Contains(endInjectOutput)
    if not check then false
    else
      match this.TryFindDesendent(startNode, (=) endNode) |> Seq.tryHead with
      | Some(_,path) -> 
        let firstOutput = path[0] 
        let lastInput = path[path.Length-1]
        this.RenameLink(firstOutput,startInjectInput,onOutputs=true,onInputs=false,preventCollision=false)
        this.RenameLink(lastInput,endInjectOutput,onOutputs=false,onInputs=true,preventCollision=false)
        true
      | None -> false

  member this.Inject(output:string,inputP:string,outputP:string) : bool =
    match this.TryNameByOutput(output) with
    | None -> false
    | Some(name) ->
         this.FindNodeByName(name).Output.Replace(output,inputP) |> ignore<bool>
         for inNode in this.NamesByInput(output) |> Array.map this.FindNodeByName do
           inNode.Input.Replace(output,outputP) |> ignore<bool>
         true


