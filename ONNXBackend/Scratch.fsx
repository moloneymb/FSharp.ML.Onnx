#I @"C:\Users\moloneymb\source\repos\QuotationTesting\packages\"
#I "C:/Users/moloneymb/.nuget/packages"
#r @"system.runtime.compilerservices.unsafe/4.5.2/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"
#r @"system.memory/4.5.3/lib/netstandard2.0/System.Memory.dll"
#r "microsoft.ml.onnxruntime/1.1.2/lib/netstandard1.1/Microsoft.ML.OnnxRuntime.dll"
#r @"google.protobuf/3.11.2/lib/netstandard2.0/Google.Protobuf.dll"
#r "../protobuf/onnx/csharp/OnnxMLProto.dll"
#r "netstandard"
#r @"FSharp.Quotations.Evaluator.1.1.3\lib\net45\FSharp.Quotations.Evaluator.dll"
#load "ProtoBuf.fs"
#load "ONNXAPI.g.fs"
#time

// TODO transform all types of Tensor<'a>  to string or Container
// TODO thread through a graph object into the functions

open FSharp.Quotations.Evaluator
open FSharp.Quotations.Evaluator.QuotationEvaluationExtensions
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.ExprShape
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open System
open System.IO
open Onnx
open ProtoBuf
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors

type on = ONNXAPI.ONNX

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

type ValueInfo = {name : string; dt : DataType}

type R() = 
    static member Abs(graph: Graph, x: ValueInfo) : ValueInfo = 
        let output = (graph.GetName("Abs_Output"))
        graph.AddNode(Node.simple "Abs" (graph.GetName("Abs"),[|x.name|],[|output|],[||]))
        {name = output; dt = x.dt}

    static member Constant(graph: Graph, t: Tensor<float32>) : ValueInfo =
        let output = (graph.GetName("Constant_Output"))
        graph.AddNode(Node.simple "Constant" (graph.GetName("Constant"),[||],[|output|],[|Attr.tensor(t).Value|]))
        {name = output; dt = DataType.FLOAT32}

    static member Constant(graph: Graph, t: Tensor<int32>) : ValueInfo =
        let output = (graph.GetName("Constant_Output"))
        graph.AddNode(Node.simple "Constant" (graph.GetName("Constant"),[||],[|output|],[|Attr.tensor(t).Value|]))
        {name = output; dt = DataType.INT32}

    static member Constant(graph: Graph, t: Tensor<int64>) : ValueInfo =
        let output = (graph.GetName("Constant_Output"))
        graph.AddNode(Node.simple "Constant" (graph.GetName("Constant"),[||],[|output|],[|Attr.tensor(t).Value|]))
        {name = output; dt = DataType.INT64}

    static member Relu(graph : Graph, A: ValueInfo) : ValueInfo =
        let output = (graph.GetName("Relu_Output"))
        graph.AddNode(Node.unaryOp "Relu" [||] (graph.GetName("Relu"),A.name,output))
        {A with name = output}

    static member Add(graph : Graph, A: ValueInfo, B: ValueInfo) : ValueInfo =
        let output = (graph.GetName("Add_Output"))
        graph.AddNode(Node.binaryOp "Add" [||] (graph.GetName("Add"),A.name,B.name,output))
        {A with name = output}

let onMethods = typeof<on>.GetMethods() |> Array.map (fun x -> x.Name) |> Set

let rMethods = 
    typeof<R>.GetMethods() 
    |> Array.filter (fun x -> x.Name <> "Constant") 
    |> Array.map (fun x -> (x.Name,x)) |> Map

let cMethods = 
    typeof<R>.GetMethods() 
    |> Array.filter (fun x -> x.Name = "Constant") 
    |> Array.map (fun x -> x.GetParameters().[1].ParameterType.GenericTypeArguments.[0].FullName,x) 
    |> Map

let fullName = typeof<on>.FullName

/// NOTE inputs are a bit hackey
let rec trans (graph: Graph) (inputs: Map<string,ValueInfo>) (quotation: Expr)   : Expr = 
    match quotation with
    | Call(_,y,z) as t when y.DeclaringType.FullName = fullName &&  onMethods.Contains(y.Name) -> 
        let f (y:Expr) = 
            if y.Type.Name = "DenseTensor`1" || y.Type.Name = "Tensor`1" then
                Expr.Call(cMethods.[y.Type.GenericTypeArguments.[0].FullName],[ <@ graph @>;y]) 
            else y
//            match y with
//            | PropertyGet(None,y,[]) when y.PropertyType.FullName = "DenseTensor`1" || y.PropertyType.FullName = "Tensor`1" ->
//                // For handling closed over vaues
//                Expr.Call(cMethods.[y.PropertyType.GenericTypeArguments.[0].FullName],[ <@ graph @>;]) 
//            | Var(y) when inputs.ContainsKey(y.Name) -> 
//                // For handling input values. A better method that y.Name is needed...
//                // TODO check if shadowed...
//                //Var(y.Name,<@ inputs.[y.Name]@>,false)
//                Expr.Value(inputs.[y.Name],typeof<ValueInfo>)
//            | _ -> y
        let z' = z |> List.map (trans graph inputs) |> List.map f
        match rMethods.TryFind(y.Name) with
        | None -> failwithf "Method name %s did not have a target method available" y.Name 
        | Some(targetMethod) ->
            Expr.Call(targetMethod,[yield <@ graph @>; yield! z'])
    | ShapeVar v -> Expr.Var v
    | ShapeLambda (v, expr) -> Expr.Lambda (v, trans graph inputs expr)
    | ShapeCombination (o, exprs) -> RebuildShapeCombination (o, List.map (trans graph inputs) exprs)

let arr = [|0.f;-1.f;2.f|].ToTensor()
let q1 = <@ on.Abs(arr) @>
let q2 = <@ fun (arr:Tensor<float32>) -> on.Abs(arr) @>

let makeValueInfoProto(x: ValueInfo) = 
    ValueInfoProto(Name = x.name, Type = 
        TypeProto(TensorType = TypeProto.Types.Tensor(ElemType = int32 x.dt)))
//
//let p = 
//    match <@ fun x -> x + 1 @> with
//    | Lambda(_,Call(_,_,[x;_])) -> x
//    | _ -> failwith "todo"
//
//
//match <@ arr @> with
//| PropertyGet(None,t,[]) -> t.PropertyType
//| _ -> failwith "todo"
//
//match p with 
//| Var(v) -> v
//| _ -> failwith "todo"

type S() = 
    static member getExprGraph<'a> (f:Expr<Tensor<'a>>) : ((unit -> Tensor<'a>) * IDisposable) =
        let g = Graph.Default()
        let res = (trans g Map.empty f).EvaluateUntyped() :?> ValueInfo
        let gp = GraphProto(Name = "G")
        gp.Output.Add(makeValueInfoProto(res))
        gp.Node.Add(g.ops)
        let sess = new InferenceSession(writeModelToStream(gp |> graphToModel))
        let f () = 
            use res2 = sess.Run([])
            (res2 |> Seq.head).AsTensor<'a>().Clone()
        (f , {new IDisposable with member this.Dispose() = sess.Dispose()})

    static member getExprGraph<'a> (f:Expr<Tensor<'a> -> Tensor<'a>>) : ((Tensor<'a> -> Tensor<'a>) * IDisposable) =
        let g = Graph.Default()
        let input = {name = g.GetName("Input"); dt = getDataType(typeof<'a>)}
        let res = (trans g (Map([input.name,input])) f).EvaluateUntyped() :?> ValueInfo
        let gp = GraphProto(Name = "G")
        gp.Input.Add(makeValueInfoProto(input))
        gp.Output.Add(makeValueInfoProto(res))
        gp.Node.Add(g.ops)
        let sess = new InferenceSession(writeModelToStream(gp |> graphToModel))
        let f x = 
            use res2 = sess.Run([])
            (res2 |> Seq.head).AsTensor<'a>().Clone()
        (f , {new IDisposable with member this.Dispose() = sess.Dispose()})


let (g2,t) = S.getExprGraph q1

//g2

let (g3,t2) = S.getExprGraph q2

let g = Graph.Default()

trans g Map.empty q1

//q1
//let g = Graph.Default()

//<@ fun x y z -> x + y + z @>

//let o = 
//    match <@ fun (x,y,z) -> x + y + z @> with
//    //| Lambda(x,Let(_,Some(Let(_,Let(_,_))))) -> (x)
//    | Let(_,Let(_),_) -> failwith "ignore"
//    | _ -> failwith "ignore"

type DF<'a>(f: 'a, d : unit -> unit) =
    member this.F = f
    interface System.IDisposable with
        member this.Dispose() = d()

//o.Type.GenericTypeArguments.Length

//<@ fun (x:int) -> 1 @>

//let rec lambda q = 
//    match q with
//    | Lambda(_,x) ->
//
//let y = 
//    match <@ fun (x,y,z) -> x + y + z @> with
//    | Lambda(_,Let(_,_,Let(_,_,Let(y,_,_)))) -> y
//    | _ -> failwith "todo"



