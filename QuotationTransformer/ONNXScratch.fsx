#I @"C:\Users\moloneymb\.nuget\packages\"
#r @"fsharp.compiler.service\25.0.1\lib\net45\FSharp.Compiler.Service.dll"
#r @"fantomas\2.9.2\lib\net452\Fantomas.dll"
#r @"falanx.machinery\0.5.2\lib\netstandard2.0\Falanx.Machinery.dll"
#r @"fsharp.quotations.evaluator\2.1.0\lib\netstandard2.0\FSharp.Quotations.Evaluator.dll"
#r @"system.runtime.compilerservices.unsafe/4.5.2/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"
#r @"system.memory/4.5.3/lib/netstandard2.0/System.Memory.dll"
#r "microsoft.ml.onnxruntime/1.1.2/lib/netstandard1.1/Microsoft.ML.OnnxRuntime.dll"
#r @"google.protobuf/3.11.2/lib/netstandard2.0/Google.Protobuf.dll"
#r @"fparsec/1.1.1/lib/net45/FParsecCS.dll"
#r @"fparsec/1.1.1/lib/net45/FParsec.dll"

#r "../protobuf/onnx/csharp/OnnxMLProto.dll"
#r "netstandard"

#load "../ONNXBackend/ProtoBuf.fs"
#load "../ONNXBackend/ONNXAPI.g.fs"

#load "Common.fs"

//open ProtoBuf
//open System.IO
//open Microsoft.ML.OnnxRuntime.Tensors
//open Microsoft.ML.OnnxRuntime
//open FSharp.Quotations.Evaluator
//open Microsoft.FSharp.Quotations
//open Microsoft.FSharp.Quotations.DerivedPatterns
//open Microsoft.FSharp.Quotations.ExprShape
//open Microsoft.FSharp.Quotations.Patterns
//open Microsoft.FSharp.Reflection
//open Onnx
//open System
//open System.Reflection
//
//type Graph = 
//    { 
//        mutable ops : NodeProto list
//        mutable usedNames : Map<string,int>
//    } 
//    static member Default() = {ops = []; usedNames = Map.empty}
//    member this.GetName(name : string) : string = 
//            let (x,y) = 
//                match this.usedNames.TryFind(name) with
//                | None -> name,this.usedNames.Add(name,0)
//                | Some(v) -> 
//                    let newName = name + string(v + 1)
//                    newName,this.usedNames.Add(name,v+1).Add(newName,0)
//            this.usedNames <- y
//            x
//
//    member this.AddNode(node: NodeProto) = this.ops <- node::this.ops
//
//type R() = 
//
//    static member Constant(graph: Graph, t: Tensor<float32>) : string =
//        let output = (graph.GetName("Constant_Output"))
//        graph.AddNode(Node.simple "Constant" (graph.GetName("Constant"),[||],[|output|],[|Attr.tensor(t).Value|]))
//        output
//
//    static member Relu(graph : Graph, A: string) : string =
//        let output = (graph.GetName("Relu_Output"))
//        graph.AddNode(Node.unaryOp "Relu" [||] (graph.GetName("Relu"),A,output))
//        output
//
//    static member Add(graph : Graph, A: string, B: string) : string =
//        let output = (graph.GetName("Add_Output"))
//        graph.AddNode(Node.binaryOp "Add" [||] (graph.GetName("Add"),A,B,output))
//        output
//
//type on = ONNXAPI.ONNX
//let g = Graph.Default()
//
//
//let t1 = ArrayTensorExtensions.ToTensor(Array.init 100 (fun i -> float32(i % 20)))
//let t2 = ArrayTensorExtensions.ToTensor(Array.init 100 (fun i -> float32((i % 20)-10)))
//let f1 = <@@ fun (x:Tensor<float32>,y:Tensor<float32>) -> on.Relu(on.Add(x,y)) @@>
//
//let f2 = f1.EvaluateUntyped() :?> (Tensor<float32>*Tensor<float32>->Tensor<float32>)
//
//
//let mnistDir = Path.Combine(__SOURCE_DIRECTORY__,"..","data","mnist")
//
//let test_data = 
//        let f(path: string) = 
//            TensorProto.Parser.ParseFrom(File.ReadAllBytes(path))
//        [| for i in [0;1;2] ->
//                Path.Combine(mnistDir,sprintf "test_data_set_0") 
//                |> fun dir -> (f(Path.Combine(dir,"input_0.pb")),f(Path.Combine(dir,"output_0.pb")))|]
//
//let bytesToFloats(buffer : byte[]) = 
//    let xs= Array.zeroCreate<float32> (buffer.Length / 4)
//    System.Buffer.BlockCopy(buffer, 0, xs, 0, buffer.Length)
//    xs
//
//let getTensorF(name,shape) =
//    let dts = File.ReadAllBytes(Path.Combine(mnistDir, name)) |> bytesToFloats
//    on.Reshape(ArrayTensorExtensions.ToTensor(dts) ,ArrayTensorExtensions.ToTensor(shape))
//
//type 'a``[]`` with
//    member x.ToTensor() = ArrayTensorExtensions.ToTensor(x)
//
//type Tensor<'a> with
//    member x.Reshape(shape:int[]) = 
//        x.Reshape(System.ReadOnlyMemory.op_Implicit(shape).Span)
//
//type ONNXAPI.ONNX with
//    static member Reshape(x: Tensor<float32>,shape: int32[]) = on.Reshape(x,(shape |> Array.map int64).ToTensor())
//
//type MNIST() = 
//    let p193 = getTensorF("Parameter193", [|16L; 4L; 4L; 10L|])
//    let p87  = getTensorF("Parameter87",  [|16L; 8L; 5L; 5L|])
//    let p5   = getTensorF("Parameter5",  [|8L; 1L; 5L; 5L|])
//    let p6   = getTensorF("Parameter6", [|8L; 1L; 1L|])
//    let p88  = getTensorF("Parameter88", [|16L; 1L; 1L|])
//    let p194 = getTensorF("Parameter194", [|1L; 10L|]) 
//
//    [<ReflectedDefinition>]
//    member this.Rec(x,p1,p2,k) = 
//        on.MaxPool(on.Relu(on.Add(on.Conv(x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst
//
//    [<ReflectedDefinition>]
//    member this.Forward(x: Tensor<float32>) = 
//        on.Add(on.MatMul(on.Reshape((this.Rec (this.Rec(x,p5,p6,2L),p87,p88,3L)),[|1;256|]),on.Reshape(p193,[|256;10|])),p194)
//
//let rd = Expr.TryGetReflectedDefinition(typeof<MNIST>.GetMember("Forward").[0] :?> MethodBase).Value


//#I @"C:\Users\moloneymb\source\repos\QuotationTesting\packages\"
//#I "C:/Users/moloneymb/.nuget/packages"
//#r @"system.runtime.compilerservices.unsafe/4.5.2/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"
//#r @"system.memory/4.5.3/lib/netstandard2.0/System.Memory.dll"
//#r "microsoft.ml.onnxruntime/1.1.2/lib/netstandard1.1/Microsoft.ML.OnnxRuntime.dll"
//#r @"google.protobuf/3.11.2/lib/netstandard2.0/Google.Protobuf.dll"
//#r "../protobuf/onnx/csharp/OnnxMLProto.dll"
//#r "netstandard"
//#r @"FSharp.Quotations.Evaluator.1.1.3\lib\net45\FSharp.Quotations.Evaluator.dll"
//#load "ProtoBuf.fs"
//#load "ONNXAPI.g.fs"
//#time
//
//// TODO transform all types of Tensor<'a>  to string or Container
//// TODO thread through a graph object into the functions
//
//open FSharp.Quotations.Evaluator
//open FSharp.Quotations.Evaluator.QuotationEvaluationExtensions
//open Microsoft.FSharp.Quotations
//open Microsoft.FSharp.Quotations.ExprShape
//open Microsoft.FSharp.Quotations.Patterns
//open Microsoft.FSharp.Quotations.DerivedPatterns
//open System
//open System.IO
//open Onnx
//open ProtoBuf
//open Microsoft.ML.OnnxRuntime
//open Microsoft.ML.OnnxRuntime.Tensors
//
//type on = ONNXAPI.ONNX
//
//type Graph = 
//    { mutable ops : NodeProto list; mutable usedNames : Map<string,int> } 
//    static member Default() = {ops = []; usedNames = Map.empty}
//    member this.GetName(name : string) : string = 
//            let (x,y) = 
//                match this.usedNames.TryFind(name) with
//                | None -> name,this.usedNames.Add(name,0)
//                | Some(v) -> 
//                    let newName = name + string(v + 1)
//                    newName,this.usedNames.Add(name,v+1).Add(newName,0)
//            this.usedNames <- y
//            x
//
//    member this.AddNode(node: NodeProto) = this.ops <- node::this.ops
//
//type ValueInfo = {name : string; dt : DataType}
//
//type R() = 
//    static member Abs(graph: Graph, x: ValueInfo) : ValueInfo = 
//        let output = (graph.GetName("Abs_Output"))
//        graph.AddNode(Node.simple "Abs" (graph.GetName("Abs"),[|x.name|],[|output|],[||]))
//        {name = output; dt = x.dt}
//
//    static member Constant(graph: Graph, t: Tensor<float32>) : ValueInfo =
//        let output = (graph.GetName("Constant_Output"))
//        graph.AddNode(Node.simple "Constant" (graph.GetName("Constant"),[||],[|output|],[|Attr.tensor(t).Value|]))
//        {name = output; dt = DataType.FLOAT32}
//
//    static member Constant(graph: Graph, t: Tensor<int32>) : ValueInfo =
//        let output = (graph.GetName("Constant_Output"))
//        graph.AddNode(Node.simple "Constant" (graph.GetName("Constant"),[||],[|output|],[|Attr.tensor(t).Value|]))
//        {name = output; dt = DataType.INT32}
//
//    static member Constant(graph: Graph, t: Tensor<int64>) : ValueInfo =
//        let output = (graph.GetName("Constant_Output"))
//        graph.AddNode(Node.simple "Constant" (graph.GetName("Constant"),[||],[|output|],[|Attr.tensor(t).Value|]))
//        {name = output; dt = DataType.INT64}
//
//    static member Relu(graph : Graph, A: ValueInfo) : ValueInfo =
//        let output = (graph.GetName("Relu_Output"))
//        graph.AddNode(Node.unaryOp "Relu" [||] (graph.GetName("Relu"),A.name,output))
//        {A with name = output}
//
//    static member Add(graph : Graph, A: ValueInfo, B: ValueInfo) : ValueInfo =
//        let output = (graph.GetName("Add_Output"))
//        graph.AddNode(Node.binaryOp "Add" [||] (graph.GetName("Add"),A.name,B.name,output))
//        {A with name = output}
//
//let onMethods = typeof<on>.GetMethods() |> Array.map (fun x -> x.Name) |> Set
//
//let rMethods = 
//    typeof<R>.GetMethods() 
//    |> Array.filter (fun x -> x.Name <> "Constant") 
//    |> Array.map (fun x -> (x.Name,x)) |> Map
//
//let cMethods = 
//    typeof<R>.GetMethods() 
//    |> Array.filter (fun x -> x.Name = "Constant") 
//    |> Array.map (fun x -> x.GetParameters().[1].ParameterType.GenericTypeArguments.[0].FullName,x) 
//    |> Map
//
//let fullName = typeof<on>.FullName
//let tensorGuid = typeof<Tensor<float32>>.GUID
//let denseTensorGuid = typeof<DenseTensor<float32>>.GUID
//let tensorGuids = set [ tensorGuid; denseTensorGuid]
//
//// TODO memoise reflected definition transformed expressions...
//
///// NOTE inputs are a bit hackey
//////let trans (inputs: Map<string,ValueInfo>) (quotation: Expr) : Expr = 
//
//let memoize fn =
//  let cache = new System.Collections.Generic.Dictionary<_,_>()
//  (fun x ->
//    match cache.TryGetValue x with
//    | true, v -> v
//    | false, _ -> let v = fn (x)
//                  cache.Add(x,v)
//                  v)
//
//open System.Reflection
//
//let tryGetRD = 
//    memoize (fun (x:MethodInfo) -> Expr.TryGetReflectedDefinition(x))
//
//let arr = [|0.f;-1.f;2.f|].ToTensor()
//
////let q1 = <@ on.Abs(arr) @>
////let q2 = <@ fun (arr:Tensor<float32>) -> on.Abs(arr) @>
////
////let makeValueInfoProto(x: ValueInfo) = 
////    ValueInfoProto(Name = x.name, Type = 
////        TypeProto(TensorType = TypeProto.Types.Tensor(ElemType = int32 x.dt)))
//
////let p = 
////    match <@ fun x -> x + 1 @> with
////    | Lambda(_,Call(_,_,[x;_])) -> x
////    | _ -> failwith "todo"
////
////
////match <@ arr @> with
////| PropertyGet(None,t,[]) -> t.PropertyType
////| _ -> failwith "todo"
////
////match p with 
////| Var(v) -> v
////| _ -> failwith "todo"
//
////let quotedProgramAsData = 
////    <@
////        let y = 1 
////        let z = 2 
////        20 * y * z
////    @>
//
//
////type S() = 
////    static member getExprGraph<'a> (f:Expr<Tensor<'a>>) : ((unit -> Tensor<'a>) * IDisposable) =
////        let g = Graph.Default()
////        let res = (trans g Map.empty f).EvaluateUntyped() :?> ValueInfo
////        let gp = GraphProto(Name = "G")
////        gp.Output.Add(makeValueInfoProto(res))
////        gp.Node.Add(g.ops)
////        let sess = new InferenceSession(writeModelToStream(gp |> graphToModel))
////        let f () = 
////            use res2 = sess.Run([])
////            (res2 |> Seq.head).AsTensor<'a>().Clone()
////        (f , {new IDisposable with member this.Dispose() = sess.Dispose()})
////
////    static member getExprGraph<'a> (f:Expr<Tensor<'a> -> Tensor<'a>>) : ((Tensor<'a> -> Tensor<'a>) * IDisposable) =
////        let g = Graph.Default()
////        let input = {name = g.GetName("Input"); dt = getDataType(typeof<'a>)}
////        let res = (trans g (Map([input.name,input])) f).EvaluateUntyped() :?> ValueInfo
////        let gp = GraphProto(Name = "G")
////        gp.Input.Add(makeValueInfoProto(input))
////        gp.Output.Add(makeValueInfoProto(res))
////        gp.Node.Add(g.ops)
////        let sess = new InferenceSession(writeModelToStream(gp |> graphToModel))
////        let f x = 
////            use res2 = sess.Run([])
////            (res2 |> Seq.head).AsTensor<'a>().Clone()
////        (f , {new IDisposable with member this.Dispose() = sess.Dispose()})
//
//
////let (g2,t) = S.getExprGraph q1
////let (g3,t2) = S.getExprGraph q2
////
////let g = Graph.Default()
////trans g Map.empty q1
//
////let o = 
////    match <@ fun (x,y,z) -> x + y + z @> with
////    //| Lambda(x,Let(_,Some(Let(_,Let(_,_))))) -> (x)
////    | Let(_,Let(_),_) -> failwith "ignore"
////    | _ -> failwith "ignore"
//
////type DF<'a>(f: 'a, d : unit -> unit) =
////    member this.F = f
////    interface System.IDisposable with
////        member this.Dispose() = d()

    //match <@ O.A @> with
    //| PropertyGet(_,PropertyGetterWithReflectedDefinition (Lambdas(yss,_) as rd),zs) -> yss 
    //| PropertySet(_,PropertySetterWithReflectedDefinition p,zs,_) -> failwith "todo"
    //| _ -> failwith "todo"


//
//let mn1 = MNIST()
//let mn2 = MNIST()
//
//(<@ mn1.Forward @> : Expr<Tensor<float32> -> Tensor<float32>>)
//
//let getF(x: MNIST)  = 
//    match <@ x.Forward @> : Expr<Tensor<float32> -> Tensor<float32>> with
//    | Lambda(_,Call(_,m,_)) -> m//Expr.TryGetReflectedDefinition(m).Value
//    | _ -> failwith "todo"
//
//// TODO, Closures are PropertyGet(None,
//
//type X() = 
////    static member y = 10
//    static member y(t:int) = t
//    static member y2 (t:int) = t
//
////<@ X.y 5 @>
//
//// PropertyGet w/o inputs
//
//match <@ x @> with | PropertyGet(None,t,[]) -> t.PropertyType | _ -> failwith "err"
//
//let t = typeof<Tensor<float32>>
//typeof<Tensor<float32>>.TypeHandle
//typeof<Tensor<int32>>.TypeHandle
//
//typeof<Tensor<int32>>.GUID
//typeof<Tensor<float32>>.GUID
//ArrayTensorExtensions.ToTensor([|System.Numerics.Complex(1.,2.)|])


