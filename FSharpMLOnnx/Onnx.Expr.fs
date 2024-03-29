﻿module FSharp.ML.Onnx.Expr

open System
open Utils.Expr
open FSharp.ML.Onnx
open FSharp.ML.Onnx.Utils
open FSharp.ML.Onnx.API.Graph
open FSharp.Quotations.Evaluator
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.ExprShape
open Microsoft.ML.OnnxRuntime.Tensors
open Microsoft.ML.OnnxRuntime
open Microsoft.FSharp.Reflection
open FSharp.ML.Onnx.Protobuf
open Onnx

type MethodSubstitution( name: string ) =
    inherit System.Attribute()
    member this.Name = name

let (|TryGetMethodSubstitution|_|) (methodInfo:Reflection.MethodInfo) : Option<string> =
  match methodInfo.GetCustomAttributes(typeof<MethodSubstitution>,false) with 
  | [| :? MethodSubstitution as ms|] -> Some(ms.Name)
  | _ -> None

let rec containsRecord (t:Type) : bool =
    if FSharpType.IsRecord t then true
    elif FSharpType.IsTuple t then
        FSharpType.GetTupleElements t |> Array.exists containsRecord
    else false

let rec mapType2 (t:Type) = 
    if FSharpType.IsRecord t then
        let fields = FSharpType.GetRecordFields t
        FSharpType.MakeTupleType(fields |> Array.map (fun x -> x.PropertyType |> mapType2))
    elif FSharpType.IsTuple t then
        if not <| containsRecord t then t
        else 
            FSharpType.MakeTupleType(t |> FSharpType.GetTupleElements |> Array.map mapType2)
    else t

let mapRecordsToTuples(expr:Expr) = 
    let rec getRecordFields(t:Type) : (string*System.Reflection.PropertyInfo[])[] = 
        [| 
            if FSharpType.IsRecord t then 
                let fields = FSharpType.GetRecordFields t
                yield t.FullName,fields
                yield! fields |> Array.collect (fun f -> getRecordFields f.PropertyType)
            elif FSharpType.IsTuple t then
                yield! FSharpType.GetTupleElements t |> Array.collect (fun x -> getRecordFields x)
        |]

    let mapRecordsToTuples   = 
        let rec mapRecordsToTuples (vars:Map<Var,Var>) (expr:Expr) = 
            match expr with
            | Var(Found vars (v)) -> Some(Expr.Var(v))
            | Var(v) when containsRecord v.Type -> 
                // I'm not sure when this would happen outside of an identiy Expr
                Some(Expr.Var(Var(v.Name,mapType2 v.Type,v.IsMutable)))
            | Let(v,e1,e2) when v.Type |> containsRecord -> 
                let v2 = Var(v.Name,mapType2 v.Type)
                printfn "Var changed %s %s %s" v.Name v.Type.Name v2.Type.Name
                let f e = e |> Expr.applyTransform (mapRecordsToTuples (vars.Add(v,v2)))
                Some(Expr.Let(v2, f e1,f e2))
            | Lambda(v,e) when v.Type |> containsRecord ->
                let v2 = Var(v.Name,mapType2 v.Type)
                printfn "Var changed %s %s %s" v.Name v.Type.Name v2.Type.Name
                Some(Expr.Lambda(v2, e |> Expr.applyTransform (mapRecordsToTuples (vars.Add(v,v2)))))
            | TupleGet(x,i) -> 
                // NOTE: This is needed as the type change needs to be threaded through
                x |> Expr.unfoldWhileChanged (mapRecordsToTuples vars) |> Seq.tryLast
                |> Option.map (fun x -> Expr.TupleGet(x,i))
            | NewRecord(_,xs) ->
               Some(Expr.NewTuple(xs |> List.map (Expr.applyTransform (mapRecordsToTuples vars)))) 
            | NewTuple(xs) -> 
                // check if any changes were made
                let ys = xs |> List.map (fun x -> x |> Expr.unfoldWhileChanged (mapRecordsToTuples vars) |> Seq.tryLast)
                if ys |> List.exists Option.isSome then
                    Some(Expr.NewTuple((xs,ys) ||> List.zip |> List.map (fun (x,y) -> defaultArg y x)))
                else
                    None
            | PropertyGet(Some(e),propertyInfo,[]) ->
                if FSharpType.IsRecord propertyInfo.DeclaringType then
                    FSharpType.GetRecordFields propertyInfo.DeclaringType 
                    |> Array.indexed 
                    |> Array.tryFind (fun (_,x) -> propertyInfo.Name = x.Name )
                    |> Option.map (fun (i,_) -> Expr.TupleGet(e |> Expr.applyTransform (mapRecordsToTuples vars) ,i))
                else
                    None
            | _ -> None
        mapRecordsToTuples Map.empty
    mapRecordsToTuples expr


// exclude 
let canEval(q:Expr) =
  let blackListTypeNames = set ["FSharp.ML.Onnx.API.PascalCase+Onnx"]

  q |> Expr.flatten 
    |> Seq.scan (fun (vars,_) y -> (match y with | Let(v,_,_) -> Set.add v vars | _ -> vars),  Some(y)) (Set.empty<Var>,None)
    |> Seq.exists (fun (vars,x) ->
      match x with  
      //| Some(NewTuple _) -> true && incTuple
      | Some(Var(v)) -> 
        if v.Name = "arg0_0" then false
        else not(vars.Contains(v) 
        )
      | Some(Call(_,f,_) as e) -> 
        blackListTypeNames.Contains(f.DeclaringType.FullName) || e.Type.FullName.StartsWith("Microsoft.ML.OnnxRuntime.Tensors")
      | Some(PropertyGet(None,_,_)) -> true
      | Some(PropertyGet(_,y,_)) ->  
        let isStructualType(t:Type) =
          t.IsArray ||
          FSharpType.IsTuple t ||
          FSharpType.IsUnion t
        // Not sure about record support
        if isStructualType y.DeclaringType then false
        else isNull y.DeclaringType.ReflectedType // this is incorrect.. when working with None
      | _ -> false)
    |> not

let singleUseBindings (body:Expr) =
  let letBindings = 
    body 
    |> Expr.flatten
    |> Seq.choose (function | Let(v,x,_) -> Some(v,x) | _ -> None)
    |> Map.ofSeq

  // vars...
  let varCounts = 
    body 
    |> Expr.flatten 
    |> Seq.choose (function | Var(v) -> Some(v) | _ -> None)
    |> Seq.toArray |> Array.groupBy id |> Array.map (fun (x,ys) -> (x,ys|>Array.length))
    |> Map.ofArray

  let replaceVars = Set [|for KeyValue(k,v) in varCounts do if v < 2 then yield k|]

  let canEvals = 
      [|for KeyValue(k,v) in letBindings do 
          match v with
          | NewUnionCase _ -> yield (k,v)
          | _ ->
            if v |> canEval then 
              yield (k,v)
            |] |> Map.ofArray

  let mutable flag = false

  let body' =
    body 
    |> Expr.map (function
      | ShapeVar v when canEvals.ContainsKey(v) -> 
        flag <- true
        canEvals.[v]
      | ShapeVar v when replaceVars.Contains(v) && letBindings.ContainsKey(v) -> 
        flag <- true
        letBindings.[v]
      | Let(v,_,body) as letExpr -> 
        match letBindings.TryFind(v) with
        | None -> failwith "shouldn't happen"
        | Some(expr) ->
          if replaceVars.Contains(v) || (not (varCounts.ContainsKey(v))) then
            flag <- true
            body
          else
            letExpr
      | x -> x)

  if flag then Some(body')
  else None

let doEvals(t:Expr) =
  match t with
  | Value(_) -> None
  | _ ->
    if t |> canEval then Some(Expr.Value(FSharp.Quotations.Evaluator.QuotationEvaluator.EvaluateUntyped t,t.Type))
    else None


type DV<'a> = DisposableValue<'a>

[<RequireQualifiedAccess>]
type BT = 
    | Unknown | Float64 | Float32 | Int64 | Int32 | Int16 | Int8 | UInt64 | UInt32 | UInt16 | UInt8
    static member tryOfType(t: Type) : BT option = 
        if t = typeof<uint8> then Some BT.UInt8 
        elif t = typeof<uint16> then Some BT.UInt16 
        elif t = typeof<uint32> then Some BT.UInt32 
        elif t = typeof<uint64> then Some BT.UInt64 
        elif t = typeof<int8> then Some BT.Int8  
        elif t = typeof<int16> then Some BT.Int16
        elif t = typeof<int32> then Some BT.Int32 
        elif t = typeof<int64> then Some BT.Int64 
        elif t = typeof<float32> then Some BT.Float32 
        elif t = typeof<double> then Some BT.Float64
        else None

    member this.ToDataType() = 
        match this with
        | BT.Unknown -> failwith "err" // Will probably need to thread through a datatype
        | BT.Float32 -> DataType.FLOAT32
        | BT.Float64 -> DataType.DOUBLE
        | BT.Int64-> DataType.INT64
        | BT.Int32-> DataType.INT32
        | BT.Int16 -> DataType.INT16
        | BT.Int8 -> DataType.INT8
        | BT.UInt64-> DataType.UINT64
        | BT.UInt32-> DataType.UINT32
        | BT.UInt16 -> DataType.UINT16
        | BT.UInt8 -> DataType.UINT8

[<RequireQualifiedAccess>]
type TT = | DenseTensor | SparseTensor | Tensor | Unknown

type MM = 
    | Single of BT * TT
    | Tuple of MM[]
    | Record of Type*(Reflection.PropertyInfo*MM)[]

let createFromTensor = Expr.tryGetGenericMethod <@@ NamedOnnxValue.CreateFromTensor @@> |> Option.get
let getArray = <@ [|0|].[0] @> |> Expr.tryGetGenericMethod |> Option.get
let unboxGeneric = <@@ unbox<_> @@> |> Expr.tryGetGenericMethod |> Option.get
let astensor = <@ (obj() :?> NamedOnnxValue).AsTensor<int>() @> |> Expr.tryGetGenericMethod |> Option.get

let rec getMM (t:Type) : MM = 
    if  t = typedefof<Tensor> then MM.Single(BT.Unknown,TT.Unknown)
    elif FSharpType.IsTuple t then
        MM.Tuple(FSharpType.GetTupleElements(t) |> Array.map getMM)
    elif FSharpType.IsRecord t then
        MM.Record(t,FSharpType.GetRecordFields(t) |> Array.map (fun pi -> pi, pi.PropertyType  |> getMM))
    elif t.IsGenericType then
        match BT.tryOfType(t.GetGenericArguments().[0]) with
        | Some(x) -> 
            let gtd = t.GetGenericTypeDefinition()
            if gtd = typedefof<Tensor<_>>  then Single(x,TT.Tensor)
            elif gtd = typedefof<DenseTensor<_>> then Single(x,TT.DenseTensor)
            else failwithf "type %s is unsupported" t.FullName
        | None -> failwithf "generic type argument %s is unsupported" (t.GetGenericArguments().[0].FullName)
    else
        failwithf "Type %s is unsupported" t.FullName

let getValueInfo(mm:MM) =
    let rec getValueInfo(index:int, mm:MM) : (int*ValueInfo[]) = 
        let f (index,xs) = 
            ((index,[]),xs ) 
            ||> Array.fold (fun (index,acc) x -> getValueInfo(index,x) |> fun (i,x) -> (i,x ::acc))
            |> fun (i,xs) -> (i,xs |> List.toArray |> Array.collect id)
        match mm with
        | Single(bt,_) -> (index+1,[|{name = sprintf "Input%i" index; dt = bt.ToDataType()}|])
        | MM.Tuple(xs) -> f (index,xs) 
        | MM.Record(_,xs) -> f (index,xs |> Array.map snd)
    getValueInfo(0,mm) |> snd

let filterMethods (m: Reflection.MethodInfo) =
    match m.Name with
    | "Equals"
    | "GetHashCode"
    | "GetType"
    | "ToString" -> false
    | _ -> true

// Thid does not appear to be used anywhere..
//let mutable onMethodsPascal = 
//    typeof<FSharp.ML.Onnx.API.PascalCase.Onnx>.GetMethods() 
//    |> Array.filter filterMethods

let mutable targetMethods = 
    typeof<OnnxGraph>.GetMethods() 
    |> Array.filter filterMethods
    |> Array.map (fun mi -> mi.Name,mi)
    |> Map.ofArray

let mutable constantFunction = 
    typeof<Constants>.GetMethods() 
    |> Array.filter filterMethods
    |> Array.map (fun mi -> (mi.GetParameters().[1].ParameterType.FullName,mi)) 
    |> Map.ofArray

let ONNXAPIFullName =
    typeof<FSharp.ML.Onnx.API.PascalCase.Onnx>.FullName

let ONNXAPIGraphFullName =
    typeof<FSharp.ML.Onnx.API.Graph.OnnxGraph>.FullName



let mutable whiteListNamespaces =
    [|
        ONNXAPIFullName
        //"Microsoft.FSharp.Core.Operators"
        "Microsoft.FSharp.Core"
        "Microsoft.FSharp.Collections.ArrayModule"
    |]

// NOTE: Only support certain types for now
let suportedBaseTypes = 
    [|
        typeof<Tensor<int>>
        typeof<Tensor<int64>>
        typeof<Tensor<float32>>
        typeof<Tensor<double>>
    |]

let paramTypes = 
    [|
        typeof<uint8>
        typeof<uint16>
        typeof<uint32>
        typeof<uint64>
        typeof<int8>
        typeof<int16>
        typeof<int32>
        typeof<int64>
        typeof<float32>
        typeof<double>
        typeof<string>
        typeof<bool>
        typeof<System.Numerics.Complex>
    |] |> Array.map (fun x -> x.FullName) |> set

let rec mapType  (t:Type) : Type option = 
    let f (xs: Type[]) = xs |> Array.map mapType |> Option.all
    if t.IsArray then
        t.GetElementType() |> mapType |> Option.map (fun t2 -> t2.MakeArrayType())
    elif t |> FSharpType.IsTuple then
        t |> FSharpType.GetTupleElements |> f |> Option.map  FSharpType.MakeTupleType 
    elif FSharpType.IsUnion t || FSharpType.IsRecord t then
        t.GetGenericArguments() |> f |> Option.map (fun xs -> t.GetGenericTypeDefinition().MakeGenericType(xs))
    elif suportedBaseTypes |> Array.exists (fun x -> x.IsAssignableFrom(t)) then
        Some(typeof<ValueInfo>)
    elif FSharpType.IsFunction t then
        let (x,y) = FSharpType.GetFunctionElements t
        match mapType x, mapType y with
        | Some(x),Some(y) -> Some(FSharpType.MakeFunctionType(x,y))
        | _,_ -> None
    elif paramTypes.Contains(t.FullName)  then  
        Some(t)
    else
        None

let isWhitelist (t:Type) = t.FullName |> fun fn -> whiteListNamespaces |> Array.exists (fn.StartsWith)

let tryMapUnionCaseInfo (uci:UnionCaseInfo) = 
    uci.DeclaringType.GenericTypeArguments 
    |> Array.map mapType 
    |> Option.all 
    |> Option.map (fun ts -> 
        uci.DeclaringType.GetGenericTypeDefinition().MakeGenericType(ts) |> FSharpType.GetUnionCases 
        |> Seq.find (fun x -> x.Tag = uci.Tag))

// TODO, change tryAssignable to support structual typing
let tryAssignable (t:Type) = suportedBaseTypes |> Array.tryFind (fun x -> x.IsAssignableFrom(t))

let mapVar (v1:Var) = 
        match mapType(v1.Type) with 
        | Some(t) -> Var(v1.Name,t)
        | None ->  
            printfn "Var %s has type %s which is not mappable" v1.Name v1.Type.FullName
            v1

module Map =
    let addRange (xs:#seq<'a*'b>) (map: Map<'a,'b>)  =
        xs |> Seq.fold (fun (map: Map<'a,'b>) (v1,v2) -> map.Add(v1,v2)) map

type Expr with
    static member Call(instanceO : Expr option, mi : Reflection.MethodInfo, args: Expr list) = 
        match instanceO with | Some(x) -> Expr.Call(x,mi,args) | None -> Expr.Call(mi,args)

// wrapping functions in a class for optional parameters
// will move instance globals into instance at a later date
type OnnxProcessor() =
  static member ProcessExpr(expr: Expr,graphExpr:Expr<Graph>, ?mappedFunctions: Map<string,Var list * Expr>) : Expr = 
    let mappedFunctions = defaultArg mappedFunctions Map.empty
    let rec processExpr (varMap: Map<Var,Var>) (expr: Expr) : Expr = 
        //printfn "%s" (sprintf "%A"expr |> fun x -> x.Substring(0,min 60 (x.Length - 1)))
        match expr with
        | NewUnionCase (uci,args) ->
            match tryMapUnionCaseInfo uci with
            | Some(uci) -> 
              // NOTE: we need to conver the type here...
              Expr.NewUnionCase(uci, args |> List.map (processExpr varMap))
            | None -> failwithf "Unable to process union type %A" uci
        | TupleGet(x,i) -> Expr.TupleGet(processExpr varMap x,i)
        | NewTuple(xs) -> Expr.NewTuple(xs |> List.map (processExpr varMap))
        | Var v -> match varMap.TryFind(v) with | Some(v) -> Expr.Var(v) | _ -> failwithf "Var %s not found %s" v.Name v.Type.FullName
        | VarSet(_,_) -> failwith "Ensure VarSet is working as expected" ///v1 |> mapVar |> fun v2 -> Expr.VarSet(v2, processExpr (varMap.Add(v1,v2)) body)
        | Lambda(v1,body) -> v1 |> mapVar |> fun v2 -> Expr.Lambda(v2, processExpr (varMap.Add(v1,v2)) body)
        | Let(v1,exp1,exp2) -> v1 |> mapVar |> fun v2 -> Expr.Let(v2, processExpr varMap exp1, processExpr (varMap.Add(v1,v2)) exp2)
        | LetRecursive(xs,body) ->  
            let (es, vs1,vs2) = xs |> List.map fst |> fun x -> (xs |> List.map (snd >> processExpr varMap),x,x |> List.map mapVar)
            Expr.LetRecursive((vs2,es) ||> List.zip, body |> processExpr (varMap |> Map.addRange ((vs1,vs2) ||> List.zip)))
        | PropertyGet(None,v,[]) as expr -> // Not sure if None here mean it's external
          if v.PropertyType.FullName = "FSharp.ML.Onnx.Protobuf+Graph" then
            expr // special case graph
          else
            match tryAssignable v.PropertyType with
            | Some(u) -> Expr.Call(constantFunction.[u.FullName],[graphExpr; expr])
            | None -> 
              //let obj = expr.EvaluateUntyped()
              //Expr.Value(obj,obj.GetType()) // yeeet
              failwithf "Unsupported PropertyGet %A as type is not assignable to a supported type" expr
        | FieldGet (_,v) as expr ->
            match tryAssignable v.FieldType with
            | Some(u) -> Expr.Call(constantFunction.[u.FullName],[graphExpr; expr])
            | None -> failwithf "Unsupported FieldGet %A as type is not assignable to a supported type" expr
        | Call(_,mi,_) when mi.Name = "constant"-> failwith "we should have aborted progress before here, TODO remove this later"
        | Coerce (_,t) -> 
            match tryAssignable t with
            | Some(u) -> Expr.Call(constantFunction.[u.FullName],[graphExpr; expr]) 
            | None -> failwithf "Unsupported Coercions %A as type is not assignable to a supported type" expr

        | Call(None,TryGetMethodSubstitution(name),callArgs) ->
          match mappedFunctions.TryFind(name) with
          | None -> 
            failwithf "Expression for substitution %s was not found" name
          | Some(args,innerExpr) -> 
            // TODO - make this a bit more elegant... maybe..
            if callArgs.Length = args.Length - 1 then
              (innerExpr,(args,[yield graphExpr.Raw;yield! (callArgs |> List.map (processExpr varMap))]) ||> List.zip) ||> List.fold (fun x (y,z) -> Expr.Let(y,z,x))
            else
              failwithf "MethodSubstitution arg length missmatch %A %A" callArgs args

        | Call(instanceO,mi,args) as expr ->
            if isWhitelist mi.DeclaringType then
                if mi.DeclaringType.FullName = ONNXAPIGraphFullName then
                    failwithf "Graph member %s should not have been found at this stage" mi.Name
                elif mi.DeclaringType.FullName = ONNXAPIFullName then
                    match targetMethods.TryFind(mi.Name) with
                    | Some(targetMethod) -> 
                        //let ys = 
                        //    targetMethod.GetParameters().[1..] 
                        //    |> Array.map (fun x -> x.ParameterType = typeof<ValueInfo> || x.ParameterType = typeof<ValueInfo option>)
                        //    |> Array.toList
                        //let args = graphExpr.Raw :: ((ys,args) ||> List.zip |> List.map (fun (y,x) -> if y then processExpr varMap x else x))
                        let args = graphExpr.Raw :: args |> List.map (processExpr varMap)
                        Expr.Call(instanceO,targetMethod, args)
                    | None -> failwithf "Unsupported %s method %s" ONNXAPIFullName mi.Name
                else 
                    match mi.GetGenericArguments() |> Array.map  mapType |> Option.all with
//                    | Some([||]) ->
//                        // for when we're unable to de generic?
//                        Expr.Call(instanceO,mi,args |> List.map (processExpr varMap))
                    | Some(ts) -> 
                        let mi' = mi.GetGenericMethodDefinition().MakeGenericMethod(ts)
                        Expr.Call(instanceO,mi',args |> List.map (processExpr varMap))
                    | None -> failwithf "Method with unsupported type %A" expr
            else
                match tryAssignable mi.DeclaringType with
                | None -> failwithf "Unsupported type %s; it is neither whitelist or assignable to a supported tensor" mi.DeclaringType.FullName
                | Some (u) -> Expr.Call(constantFunction.[u.FullName],[graphExpr; expr])
        | Application(expr1, expr2) -> Expr.Application(expr1 |> processExpr varMap,expr2 |> processExpr varMap)
        | PropertyGet(x,y,z) -> 
            match x with 
            | Some(x) -> Expr.PropertyGet(x,y,z) 
            | None -> Expr.PropertyGet(y,z)
        | Value(o,t) -> Expr.Value(o,t)
        | ShapeVar _ -> failwithf "ShapeVar %A" expr
        | ShapeLambda (v, expr) -> failwithf "ShapeLamda %A" expr
        | ShapeCombination (o, xs) -> 
            printfn "ShapeCombination %O %O" o xs
            RebuildShapeCombination (o, xs |> List.map (processExpr varMap))
    processExpr Map.empty expr 

  static BuildGraph(func: Expr<'a->'b>,mmIn:MM,mmOut:MM, ?mappedFunctions:Map<string,Var list * Expr>) : ValueInfo[]*ValueInfo[]*ModelProto =
    let inputs = getValueInfo(mmIn)

    let assembleInput(value: Expr) (mm:MM) : (Expr) =
        let getValueInfoFromArray = 
            let x = getArray.MakeGenericMethod(typeof<ValueInfo>)
            fun index -> Expr.Call(x,[value; Expr.Value(index)])
        let rec combineValueInfoResult(index:int,  mm:MM) : (int*Expr) =
            let f(index,xs) =
                ((index,[]),xs) 
                ||> Array.fold (fun (index,acc) x -> combineValueInfoResult(index,  x) |> fun (i,x) -> (i,x ::acc))
                |> fun (i,xs) -> (i,Expr.NewTuple(xs |> List.rev))
            match mm with
            | MM.Single(_,_) ->
                (index+1,getValueInfoFromArray index)
            | MM.Tuple(xs) -> f(index,xs)
            | MM.Record(_,xs) -> f(index,xs |> Array.map snd)
        combineValueInfoResult(0,mm) |> snd

    let flattenOutput(value: Expr, mm:MM) : (Expr<ValueInfo[]>) =
        let rec trans (expr: Expr) (mm:MM) : Expr[] =
            [|
                let f(xs:MM[]) = 
                    xs 
                    |> Array.mapi (fun i x -> trans (Expr.TupleGet(expr,i)) x)
                    |> Array.collect id
                match mm with
                | MM.Single(_) -> yield expr 
                | MM.Tuple(xs) -> yield! f(xs)
                | MM.Record(_,xs) -> yield! f(xs |> Array.map snd)
            |] 
        trans (value) mm
        |> List.ofArray
        |> Expr.concat<ValueInfo>

    let graph = Graph.Default()

    let transformedFunc = 
        func 
        |> simplify 
        |> Expr.applyTransform  mapRecordsToTuples
        |> Expr.applyTransforms [| 
          ExprTransforms.lambdasToLets
          ExprTransforms.boolTransform
          singleUseBindings ; 
          doEvals |]
        |> fun x -> OnnxProcessor.ProcessExpr(x,<@ graph @>,?mappedFunctions=mappedFunctions)
        |> fun f -> 
            // argument type does not match 
            let v = Expr.Application(f, assembleInput <@ inputs @> mmIn).EvaluateUntyped()
            flattenOutput(Expr.Value(v,v.GetType()),mmOut)

    let outputs = transformedFunc.Evaluate()

    let makeValueInfoProto(valueInfo: ValueInfo) = 
        ValueInfoProto(Name = valueInfo.name, Type = TypeProto(TensorType = TypeProto.Types.Tensor(ElemType = int32 valueInfo.dt)))

    let gp = GraphProto(Name = "G")
    gp.Input.Add(inputs |> Array.map makeValueInfoProto)
    gp.Output.Add(outputs |> Array.map makeValueInfoProto)
    gp.Node.Add(graph.ops)
    inputs, outputs, gp |> graphToModel

  static ToOnnxGraph<'a,'b>(expr: Expr<'a -> 'b>,?extraFunctions:FunctionProto[],?mappedFunctions)  : DV<'a -> DV<'b>>= 
    let mmIn = getMM (typeof<'a>)
    let mmOut = getMM (typeof<'b>)

    let inputs,outputs,model = OnnxProcessor.BuildGraph(expr,mmIn,mmOut,?mappedFunctions=mappedFunctions)
    extraFunctions |> Option.iter model.Functions.AddRange

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
                let mt = astensor.MakeGenericMethod(bt.ToDataType() |> tryDataTypeToType |> Option.get)
                let x1 = Expr.Call(getArray.MakeGenericMethod(typeof<NamedOnnxValue>),[value; Expr.Value(index)])
                let x2 = Expr.Call(x1,mt,[])
                (index+1,x2)
            | TT.DenseTensor -> 
                let t = bt.ToDataType() |> tryDataTypeToType |> Option.get
                let mt = astensor.MakeGenericMethod(t)
                let x1 = Expr.Call(getArray.MakeGenericMethod(typeof<NamedOnnxValue>),[value; Expr.Value(index)])
                let x2 = Expr.Call(x1,mt,[])
                let x3 = Expr.Call(unboxGeneric.MakeGenericMethod(typedefof<DenseTensor<_>>.MakeGenericType(t)), [x2])
                (index+1,x3)
        | MM.Tuple(xs) -> 
            ((index,[]),xs) 
            ||> Array.fold (fun (index,acc) x -> combineResult(index, value, x) |> fun (i,x) -> (i,x ::acc))
            |> fun (i,xs) -> (i,Expr.NewTuple(xs |> List.rev))
        | MM.Record(t,xs) -> 
            ((index,[]),xs) 
            ||> Array.fold (fun (index,acc) (_,x) -> combineResult(index, value, x) |> fun (i,x) -> (i,x ::acc))
            |> fun (i,xs) -> (i, Expr.NewRecord(t, xs |> List.rev))

    let sess = new InferenceSession(model |> writeModelToStream)

    let flatten = 
        let v = Var("x",typeof<'a>)
        let flatInputs = Expr.Lambda(v,(getNamedValues(0,Expr.Var(v),mmIn) |> snd |> Array.map (fun x -> x.Raw) |> List.ofArray  |> Expr.concat<NamedOnnxValue>)).EvaluateUntyped() :?> 'a -> NamedOnnxValue[]
        flatInputs

    let cmb = 
        let v2 = Var("r",typeof<NamedOnnxValue[]>)
        let r2 = combineResult(0,Expr.Var(v2),mmOut) |> snd
        Expr.Lambda(v2, r2).EvaluateUntyped() :?> NamedOnnxValue[] -> 'b

    let partialRun (x: 'a ) = 
        let flatInputs = flatten x
        let results = sess.Run(flatten x)
        let results3 = 
            let mOut = [| for x in results -> (x.Name,x :> NamedOnnxValue)|] |> Map.ofArray 
            // It appears that inputs that make it to outputs untouched are not initialized
            // We fix this by short-circuiting
            let mIn = [| for x in flatInputs -> (x.Name,x)|] |> Map.ofArray 
            [| for x in outputs -> (mIn.TryFind(x.name) |> Option.defaultValue mOut.[x.name]) |]
            |> cmb
        new DV<'b>(results3, fun () -> results.Dispose())
    new DV<'a -> DV<'b>> (partialRun, fun () -> sess.Dispose())


