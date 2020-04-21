#load @"Base.fsx"

open Common
open Microsoft.ML.OnnxRuntime.Tensors
open ProtoBuf
open System.IO
open Common
open FSharp.Quotations.Evaluator
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors
open Onnx
open ProtoBuf
open System
open System.IO
open Microsoft.FSharp.Reflection
open Base
open ExprRun

/// NOTE: Not sufficiently tested
let rec getLeafNodes (m:Map<Var,Expr>) (expr: Expr) : Expr[] =
    [|
        match expr with
        | Patterns.Var v -> 
            match m.TryFind(v) with
            | Some(x) -> yield! getLeafNodes m x
            | None ->  yield expr
        | Patterns.Value (_,_) 
        | Patterns.Call (_, _, _) 
        | ShapeVar _ -> yield expr
        | Patterns.Sequential (_, e1) 
        | ShapeLambda(_,e1) -> 
            yield! getLeafNodes m e1
        | Patterns.Let (v, e0, e1) -> yield! getLeafNodes (m.Add(v,e0)) e1
        | Patterns.Application (e0, _) -> 
            yield! getLeafNodes m e0
        | Patterns.WhileLoop (_, _) 
        | Patterns.ForIntegerRangeLoop (_, _, _, _) -> ()
        | Patterns.IfThenElse (_, e1, e2) -> yield! getLeafNodes m e1; yield! getLeafNodes m e2
        | _ -> failwith "not supported expr type"
    |]

<@ let f y = y + 2 in f 10 @> |> getLeafNodes Map.empty

let addFunction = <@ fun (x:Tensor<float32>,y:Tensor<float32>) -> on.add(x,y) @> 

//let addFunction = <@ on.add : (Tensor<float32>*Tensor<float32>) -> Tensor<float32> @> |> eagerToGraph 
// TODO TupleGet need to convert receivedType

module Expr = 
    let merge<'a>(xs: Expr list) = Expr.Cast<'a[]>(Expr.NewArray(typeof<'a>, xs))

    let applyTransforms (fs:(Expr -> Expr option)[])  = 
        let cmb (fs:(Expr -> Expr option)[]) (expr: Expr) = 
            seq { for f in fs do match f expr with | Some(x) -> yield x | _ -> ()} |> Seq.tryHead
        Expr.applyTransform (cmb fs)

let simplify (expr: Expr) = 
    expr
    |> Expr.applyTransform ExprTransforms.expandWithReflectedDefinition
    |> Expr.applyTransform ExprTransforms.Simple.builtIns
    |> Expr.applyTransforms 
        [|
            ExprTransforms.reduceApplicationsAndLambdas true
            ExprTransforms.Simple.tuples
            ExprTransforms.Simple.newTuple
        |]
    |> Expr.applyTransform ExprTransforms.Simple.bindings

let mnist = Base.MNIST()

let expr2 = 
    <@ mnist.Forward @> 
    |> simplify


//let flattenStructualReturnType<'b> (expr:Expr) (mm:MM) (f:Expr -> Expr) = 
//    let rec trans (expr: Expr) (mm:MM) : Expr list =
//        [
//            match mm,expr with
//            | MM.Single(_),_ -> yield f expr 
//            | MM.Tuple(xs),NewTuple(ys) -> yield! (xs |> List.ofArray,ys) ||> List.zip |> List.collect (fun (x,y) -> trans y x)
//            | MM.Record(_,xs), NewRecord(_,ys) -> yield! (xs |> List.ofArray,ys) ||> List.zip |> List.collect (fun ((_,x),y) -> trans y x)
//            | _,_ -> failwith "err"
//        ] 
//    trans expr mm
//    |> Expr.merge<'b>

//let p1 = [|1.f|].ToTensor()
//let v1 = {name = ""; dt = DataType.FLOAT32}

// Expression leaf nodes


//<@ let f() = (p1,p1) in fst (f()),snd (f()) @>  |> simplify
//<@ fst <| f() @> |> simplify


//(flattenStructualReturnType<ValueInfo> expr1 (getMM(typeof<Tensor<float32>*Tensor<float32>>)) (fun _ -> <@@ v1 @@>))
//(flattenStructualReturnType<ValueInfo> expr2 (getMM(typeof<Tensor<float32>>)) (fun _ -> <@@ v1 @@>))

type InputType = Tensor<float32>
type OutputType = Tensor<float32>
let inputMM = ExprRun.getMM typeof<InputType>
let outputMM = ExprRun.getMM typeof<OutputType>

let inputs = ExprRun.getValueInfo(0,inputMM) |> snd |> Array.item 0

//Map strucutal type of 'a to a flat array
//Expr<'a -> 'b> to Expr<'a[] -> 'b>

// Replace TupleGet and FieldGet for type with Array Get
// What happens if other functions are in the way..., generally can't support them anyway....


let rec getValueInfo(index:int, mm:MM) : (int*ValueInfo[]) = 
    let f (index,xs) = 
        ((index,[]),xs ) 
        ||> Array.fold (fun (index,acc) x -> getValueInfo(index,x) |> fun (i,x) -> (i,x ::acc))
        |> fun (i,xs) -> (i,xs |> List.toArray |> Array.collect id)
    match mm with
    | Single(bt,_) -> (index+1,[|{name = sprintf "Input%i" index; dt = bt.ToDataType()}|])
    | MM.Tuple(xs) -> f (index,xs) 
    | MM.Record(_,xs) -> f (index,xs |> Array.map snd)

// NOTE: Deeply nested structures here would be marginally more performant
// NOTE: wrap in a Lambda expression and cast and compile into a function
// Alternatively do common expression elimination
let rec getNamedValues(index:int, value: Expr, mm:MM) : (int*Expr<NamedOnnxValue>[]) = 
        match mm with
        | MM.Single(bt,_) ->
            let m = createFromTensor.MakeGenericMethod(bt.ToDataType() |> tryDataTypeToType |> Option.get)
            // NOTE: May have to add a cast here... if DenseTensor
            (index+1,[|Expr.Call(m,[Expr.Value(sprintf "Input%i" index); value ]) |> Expr.Cast<NamedOnnxValue>|])
        | MM.Tuple(xs) -> 
            ((index,[]),xs |> Array.indexed) 
            ||> Array.fold (fun (index,acc) (i,x) -> getNamedValues(index, Expr.TupleGet(value,i), x) |> fun (i,x) -> (i,x ::acc))
            |> fun (i,xs) -> (i,xs |> List.toArray |> Array.collect id)
        | MM.Record(_,xs) -> 
            ((index,[]),xs) 
            ||> Array.fold (fun (index,acc) (pi,x) -> getNamedValues(index, Expr.PropertyGet(value,pi,[]), x) |> fun (i,x) -> (i,x ::acc))
            |> fun (i,xs) -> (i,xs |> List.toArray |> Array.collect id)

let rec combineResult(index:int, value: Expr, mm:MM) : (int*Expr) =
        match mm with
        | MM.Single(bt,tt) ->
            if bt = BT.Unknown then failwith "unsupported"
            match tt with
            | TT.SparseTensor -> failwith "unsupported"
            | TT.Unknown -> failwith "unsupported"
            | TT.Tensor -> 
                //let m = getArray.MakeGenericMethod(typedefof<Tensor<_>>.MakeGenericType()
                let mt = astensor.MakeGenericMethod(bt.ToDataType() |> tryDataTypeToType |> Option.get)
                (index+1,Expr.Call(mt,[Expr.Call(getArray.MakeGenericMethod(typeof<NamedOnnxValue>),[value; Expr.Value(index)])]))
            | TT.DenseTensor -> 
                let t = bt.ToDataType() |> tryDataTypeToType |> Option.get
                let mt = astensor.MakeGenericMethod(t)
                (index+1,Expr.Call(unboxGeneric.MakeGenericMethod(typedefof<DenseTensor<_>>.MakeGenericType(t)), [Expr.Call(Expr.Call(getArray.MakeGenericMethod(typeof<NamedOnnxValue>),[value; Expr.Value(index)]),mt,[])]))
        | MM.Tuple(xs) -> 
            ((index,[]),xs) 
            ||> Array.fold (fun (index,acc) x -> combineResult(index, value, x) |> fun (i,x) -> (i,x ::acc))
            |> fun (i,xs) -> (i,Expr.NewTuple(xs |> List.rev))
        | MM.Record(t,xs) -> 
            ((index,[]),xs) 
            ||> Array.fold (fun (index,acc) (_,x) -> combineResult(index, value, x) |> fun (i,x) -> (i,x ::acc))
            |> fun (i,xs) -> (i, Expr.NewRecord(t, xs |> List.rev))


////This creates a structual object
//let rec getValueInfoObj(index:int, mm:MM) : (int*obj) = 
//    let f (index,xs) = 
//        ((index,[]),xs ) 
//        ||> Array.fold (fun (index,acc) x -> getValueInfo(index,x) |> fun (i,x) -> (i,x ::acc))
//        |> fun (i,xs) -> (i,xs |> List.toArray |> Array.collect id)
//    match mm with
//    | Single(bt,_) -> (index+1,[|{name = sprintf "Input%i" index; dt = bt.ToDataType()}|])
//    | MM.Tuple(xs) -> f (index,xs) 
//    | MM.Record(_,xs) -> f (index,xs |> Array.map snd)

let runExpr (f: Expr<'a -> 'b>) : DV<'a -> DV<'b>> = 
    //let eagerToGraph (f: Expr<'a -> 'b>) =
    let makeValueInfoProto(valueInfo: ValueInfo) = ValueInfoProto(Name = valueInfo.name, Type = TypeProto(TensorType = TypeProto.Types.Tensor(ElemType = int32 valueInfo.dt)))
    let graph = Graph.Default()
    let mnist = Base.MNIST()
    let funExpr : Expr<Tensor<float32> -> Tensor<float32>> = <@ mnist.Forward @>
    let fExpr (f:Expr<'a -> 'b>) = 
        f
        |> Expr.applyTransform ExprTransforms.expandWithReflectedDefinition 
        |> Expr.applyTransform (ExprTransforms.reduceApplicationsAndLambdas true) 
        |> ExprGraph.processExpr <@ graph @>

    // TODO find alternative for ValueInfo cast
    let outputs : ValueInfo = Expr.Cast<ValueInfo -> ValueInfo>(funExpr |> Common.ExprGraph.processExpr  <@ graph @> ).Evaluate()(inputs)
    let gp = GraphProto(Name = "G")
    gp.Input.Add(makeValueInfoProto(inputs))
    gp.Output.Add(makeValueInfoProto(outputs))
    gp.Node.Add(graph.ops)
    let sess = new InferenceSession(gp |> graphToModel |> writeModelToStream)
    let flattenResult : Tensor<float32> -> NamedOnnxValue[] = 
        failwith "todo"
    let combineResult : NamedOnnxValue[] -> Tensor<float32> = 
        fun xs -> xs.[0].AsTensor<float32>()
    let partialRun (x: 'a ) = 
        let rc = sess.Run(flattenResult x)
        let yy = combineResult [|for x in rc -> x :> NamedOnnxValue|]
        new DV<'b>(yy, fun () -> rc.Dispose())
    new DV<'a -> DV<'b>> (partialRun, fun () -> sess.Dispose())

let testModel(f: Expr<Tensor<float32> -> Tensor<float32>>) = 
    let test_data = 
        let f(path: string) = 
            TensorProto.Parser.ParseFrom(File.ReadAllBytes(path))
        [| for i in [0;1;2] ->
                Path.Combine(mnistDir,sprintf "test_data_set_%i" i) 
                |> fun dir -> (f(Path.Combine(dir,"input_0.pb")),f(Path.Combine(dir,"output_0.pb")))|]
    use mm = runExpr f
    for (index,(input,output)) in test_data |> Array.indexed do
        use values2 = mm.F(upcast Tensor.FromTensorProtoFloat32(input)) 
        let ys = values2.F |> Seq.toArray
        let diff = 
            (ys, Tensor.FromTensorProtoFloat32(output) |> Seq.toArray)
            ||> Array.zip
            |> Array.sumBy (fun (x,y) -> System.Math.Abs(x-y))
        if diff > 0.1f then failwithf "Unexpected result in example %i with a difference of %f" index diff
        printfn "%f %A" diff ys

//
//let testModel(model : byte[]) = 
//    use sess = new InferenceSession(model)
//    for (index,(input,output)) in test_data |> Array.indexed do
//        use values2 = sess.Run([|NamedOnnxValue.CreateFromTensor("Input3",Tensor.FromTensorProtoFloat32(input))|])
//        let ys = values2 |> Seq.toArray |> Array.head |> fun v -> v.AsTensor<float32>() |> Seq.toArray
//        let diff = 
//            (ys, Tensor.FromTensorProtoFloat32(output) |> Seq.toArray)
//            ||> Array.zip
//            |> Array.sumBy (fun (x,y) -> System.Math.Abs(x-y))
//        if diff > 0.1f then failwithf "Unexpected result in example %i with a difference of %f" index diff
//        printfn "%f %A" diff ys
//
//testModel(writeModelToStream(gp |> graphToModel))

//
//
//
//let v = (ArrayTensorExtensions.ToTensor([|9|]) , (ArrayTensorExtensions.ToTensor([|9.f|]) , ArrayTensorExtensions.ToTensor([|9.|])))
//
//let wrapGraphExec(f:Expr<'a -> 'b>) : DF<'a -> 'b> =
//    failwith "err"
//
//let mm = getMM (v.GetType())
//let (_,ex) = getNamedValues (0, <@ v @>, getMM (v.GetType()))
//let tt = Expr.NewArray(typeof<NamedOnnxValue>, ex |> Array.rev |> List.ofArray |> List.map (fun x -> x :> Expr))
//let outputs = tt.EvaluateUntyped() :?> NamedOnnxValue[]
//let _,cr = combineResult(0,<@@ outputs @@>, mm)
//
////Expr.Cast<DenseTensor<int>*(DenseTensor<float32>*DenseTensor<double>)>(cr).Evaluate()
//
////let fromNames (outputs : ValueInfo[]) (mm:MM) : obj -> NamedOnnxValue[] = 
////    failwith "err"
//
////let runFast (expr:Expr<'a -> 'b>) : DF<'a -> 'b> = 
////let mmA = getMM typeof<'a>
////let mmB = getMM typeof<'b>
//
////type Foo  = {x:Tensor<float>;y:Tensor<int>;z:(Tensor<uint8>*Tensor<int>)}
//
////getMM (typeof<Tensor<int>*Tensor*Foo>)
////getMM (typeof<Tensor<int>*Tensor>)
////getMM (typeof<Tensor<int>*Tensor*Foo>)
//
//// given the object and the meta-model make a function 
//
////let rec unwrapTensors (x:obj) : UTensor[] = 
////    let runtimeType = x.GetType()
////    let genericTypeDefinition = runtimeType.GetGenericTypeDefinition()
////    if  genericTypeDefinition = typedefof<Tensor<_>> || genericTypeDefinition = typedefof<Tensor> then
////         [|box x :?> Tensor|]
////    else
////        if FSharpType.IsTuple runtimeType then
////            FSharpValue.GetTupleFields x |> Array.collect unwrapTensors
////        elif FSharpType.IsRecord runtimeType then
////            FSharpValue.GetRecordFields x |> Array.collect unwrapTensors
////        else
////            failwithf "Type %s is unsupported" runtimeType.FullName
//
////FSharpType.IsTuple typeof<int*int>
////FSharpType.IsRecord
//
