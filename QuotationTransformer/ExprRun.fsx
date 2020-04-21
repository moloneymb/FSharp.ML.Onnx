#load "Base.fsx"
//
//(*
//    TODO 
//        Creating the model
//        MM -> ValueInfo[]
//        MM -> ('a -> NamedOnnxValue[])
//        MM -> (NamedOnnxValue[] -> 'b)
//        (ValueInfo[],Expr<'a -> 'b>) -> ValueInfo[] // outputs names
//    From types generate the meta model
//        'a -> NammedOnnxValue[] -> run -> NammedOnnxValue[] -> 'b
//    NOTE: We cast after transform from Tensor<'a> to ValueInfo 
//    NOTE: In creating the graph we 
//    NOTE: In running graph creation we get the names returned
//*)
//
//open Common
//open FSharp.Quotations.Evaluator
//open Microsoft.FSharp.Quotations
//open Microsoft.FSharp.Quotations.Patterns
//open Microsoft.FSharp.Quotations.DerivedPatterns
//open Microsoft.FSharp.Quotations.ExprShape
//open Microsoft.ML.OnnxRuntime
//open Microsoft.ML.OnnxRuntime.Tensors
//open Onnx
//open ProtoBuf
//open System
//open System.IO
//open Microsoft.FSharp.Reflection
//
//
//[<RequireQualifiedAccess>]
//type BT = 
//    | Unknown // Tensor without a generic type argument
//    | Float64
//    | Float32
//    | Int64
//    | Int32
//    | Int16
//    | Int8
//    | UInt64
//    | UInt32
//    | UInt16
//    | UInt8
//    static member tryOfType(t: Type) : BT option = 
//        if t = typeof<uint8> then Some BT.UInt8 
//        elif t = typeof<uint16> then Some BT.UInt16 
//        elif t = typeof<uint32> then Some BT.UInt32 
//        elif t = typeof<uint64> then Some BT.UInt64 
//        elif t = typeof<int8> then Some BT.Int8  
//        elif t = typeof<int16> then Some BT.Int16
//        elif t = typeof<int32> then Some BT.Int32 
//        elif t = typeof<int64> then Some BT.Int64 
//        elif t = typeof<float32> then Some BT.Float32 
//        elif t = typeof<double> then Some BT.Float64
//        else None
//
//    member this.ToDataType() = 
//        match this with
//        | BT.Unknown -> failwith "err" // Will probably need to thread through a datatype
//        | BT.Float32 -> DataType.FLOAT32
//        | BT.Float64 -> DataType.DOUBLE
//        | BT.Int64-> DataType.INT64
//        | BT.Int32-> DataType.INT32
//        | BT.Int16 -> DataType.INT16
//        | BT.Int8 -> DataType.INT8
//        | BT.UInt64-> DataType.UINT64
//        | BT.UInt32-> DataType.UINT32
//        | BT.UInt16 -> DataType.UINT16
//        | BT.UInt8 -> DataType.UINT8
//
//[<RequireQualifiedAccess>]
//type TT = 
//    | DenseTensor
//    | SparseTensor
//    | Tensor
//    | Unknown
//
//type MM = 
//    | Single of BT * TT
//    | Tuple of MM[]
//    | Record of Type*(Reflection.PropertyInfo*MM)[]
//
//
//let createFromTensor = Expr.tryGetGenericMethod <@@ NamedOnnxValue.CreateFromTensor @@> |> Option.get
//let getArray = <@ [|0|].[0] @> |> Expr.tryGetGenericMethod |> Option.get
//let unboxGeneric = <@@ unbox<_> @@> |> Expr.tryGetGenericMethod |> Option.get
//let astensor = <@ (obj() :?> NamedOnnxValue).AsTensor<int>() @> |> Expr.tryGetGenericMethod |> Option.get
//
//let rec getMM (t:Type) : MM = 
//    if  t = typedefof<Tensor> then MM.Single(BT.Unknown,TT.Unknown)
//    elif FSharpType.IsTuple t then
//        MM.Tuple(FSharpType.GetTupleElements(t) |> Array.map getMM)
//    elif FSharpType.IsRecord t then
//        MM.Record(t,FSharpType.GetRecordFields(t) |> Array.map (fun pi -> pi, pi.PropertyType  |> getMM))
//    elif t.IsGenericType then
//        match BT.tryOfType(t.GetGenericArguments().[0]) with
//        | Some(x) -> 
//            let gtd = t.GetGenericTypeDefinition()
//            if gtd = typedefof<Tensor<_>>  then Single(x,TT.Tensor)
//            elif gtd = typedefof<DenseTensor<_>> then Single(x,TT.DenseTensor)
//            else failwithf "type %s is unsupported" t.FullName
//        | None -> failwithf "generic type argument %s is unsupported" (t.GetGenericArguments().[0].FullName)
//    else
//        failwithf "Type %s is unsupported" t.FullName
//
//let rec getValueInfo(index:int, mm:MM) : (int*ValueInfo[]) = 
//    let f (index,xs) = 
//        ((index,[]),xs ) 
//        ||> Array.fold (fun (index,acc) x -> getValueInfo(index,x) |> fun (i,x) -> (i,x ::acc))
//        |> fun (i,xs) -> (i,xs |> List.toArray |> Array.collect id)
//    match mm with
//    | Single(bt,_) -> (index+1,[|{name = sprintf "Input%i" index; dt = bt.ToDataType()}|])
//    | MM.Tuple(xs) -> f (index,xs) 
//    | MM.Record(_,xs) -> f (index,xs |> Array.map snd)
//
//// NOTE: Deeply nested structures here would be marginally more performant
//// NOTE: wrap in a Lambda expression and cast and compile into a function
//// Alternatively do common expression elimination
//let rec getNamedValues(index:int, value: Expr, mm:MM) : (int*Expr<NamedOnnxValue>[]) = 
//        match mm with
//        | MM.Single(bt,_) ->
//            let m = createFromTensor.MakeGenericMethod(bt.ToDataType() |> tryDataTypeToType |> Option.get)
//            // NOTE: May have to add a cast here... if DenseTensor
//            (index+1,[|Expr.Call(m,[Expr.Value(sprintf "Input%i" index); value ]) |> Expr.Cast<NamedOnnxValue>|])
//        | MM.Tuple(xs) -> 
//            ((index,[]),xs |> Array.indexed) 
//            ||> Array.fold (fun (index,acc) (i,x) -> getNamedValues(index, Expr.TupleGet(value,i), x) |> fun (i,x) -> (i,x ::acc))
//            |> fun (i,xs) -> (i,xs |> List.toArray |> Array.collect id)
//        | MM.Record(_,xs) -> 
//            ((index,[]),xs) 
//            ||> Array.fold (fun (index,acc) (pi,x) -> getNamedValues(index, Expr.PropertyGet(value,pi,[]), x) |> fun (i,x) -> (i,x ::acc))
//            |> fun (i,xs) -> (i,xs |> List.toArray |> Array.collect id)
//
//
//
//let rec combineResult(index:int, value: Expr, mm:MM) : (int*Expr) =
//        match mm with
//        | MM.Single(bt,tt) ->
//            if bt = BT.Unknown then failwith "unsupported"
//            match tt with
//            | TT.SparseTensor -> failwith "unsupported"
//            | TT.Unknown -> failwith "unsupported"
//            | TT.Tensor -> 
//                //let m = getArray.MakeGenericMethod(typedefof<Tensor<_>>.MakeGenericType()
//                let mt = astensor.MakeGenericMethod(bt.ToDataType() |> tryDataTypeToType |> Option.get)
//                (index+1,Expr.Call(mt,[Expr.Call(getArray.MakeGenericMethod(typeof<NamedOnnxValue>),[value; Expr.Value(index)])]))
//            | TT.DenseTensor -> 
//                let t = bt.ToDataType() |> tryDataTypeToType |> Option.get
//                let mt = astensor.MakeGenericMethod(t)
//                (index+1,Expr.Call(unboxGeneric.MakeGenericMethod(typedefof<DenseTensor<_>>.MakeGenericType(t)), [Expr.Call(Expr.Call(getArray.MakeGenericMethod(typeof<NamedOnnxValue>),[value; Expr.Value(index)]),mt,[])]))
//        | MM.Tuple(xs) -> 
//            ((index,[]),xs) 
//            ||> Array.fold (fun (index,acc) x -> combineResult(index, value, x) |> fun (i,x) -> (i,x ::acc))
//            |> fun (i,xs) -> (i,Expr.NewTuple(xs |> List.rev))
//        | MM.Record(t,xs) -> 
//            ((index,[]),xs) 
//            ||> Array.fold (fun (index,acc) (_,x) -> combineResult(index, value, x) |> fun (i,x) -> (i,x ::acc))
//            |> fun (i,xs) -> (i, Expr.NewRecord(t, xs |> List.rev))
//
