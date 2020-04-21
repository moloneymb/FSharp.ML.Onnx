#load "Base.fsx"
//
//// TODO Evaulting sub expressions may not work if there are missing variable declarations
////      For example 'shape' var is missing in the tested sub expression
////      Reducing the sub expression will often fix this as it will push the variable assignment into 
////      the sub expression.
////      In general it's probably OK not to worry about it because the expression will still contain the Var
//
//open Common
//open FSharp.Quotations.Evaluator
//open Microsoft.FSharp.Quotations
//open Microsoft.FSharp.Quotations.Patterns
//open Microsoft.FSharp.Quotations.DerivedPatterns
//open Microsoft.FSharp.Quotations.ExprShape
//open Microsoft.ML.OnnxRuntime.Tensors
//open Onnx
//open ProtoBuf
//open System
//open System.IO
//open Microsoft.FSharp.Reflection
//open Base
//
//
//let filterMethods (m: Reflection.MethodInfo) =
//    match m.Name with
//    | "Equals"
//    | "GetHashCode"
//    | "GetType"
//    | "ToString" -> false
//    | _ -> true
//
//let onMethods = 
//    typeof<on>.GetMethods() 
//    |> Array.filter filterMethods
//
//let targetMethods = 
//    typeof<ONNXAPIGraph.ONNXGraph>.GetMethods() 
//    |> Array.filter filterMethods
//    |> Array.map (fun mi -> mi.Name,mi)
//    |> Map.ofArray
//
//let constantFunction = 
//    typeof<Constants>.GetMethods() 
//    |> Array.filter filterMethods
//    |> Array.map (fun mi -> (mi.GetParameters().[1].ParameterType.FullName,mi)) 
//    |> Map.ofArray
//
//let ONNXAPIFullName = typeof<ONNXAPI.ONNX>.FullName
//let ONNXAPIGraphFullName = typeof<ONNXAPIGraph.ONNXGraph>.FullName
//
//let whiteListNamespaces =
//    [|
//        ONNXAPIFullName
//        //"Microsoft.FSharp.Core.Operators"
//        "Microsoft.FSharp.Core"
//        "Microsoft.FSharp.Collections.ArrayModule"
//    |]
//
//// NOTE: Only support certain types for now
//let suportedBaseTypes = 
//    [|
//        typeof<Tensor<int>>
//        typeof<Tensor<int64>>
//        typeof<Tensor<float32>>
//        typeof<Tensor<double>>
//    |]
//
//
//module Option =
//    let all (xs: #seq<'a option>) = 
//        let xs = xs |> Seq.toArray
//        let ys = xs |> Array.choose id
//        if xs.Length = ys.Length then Some(ys) else None
//
//
//let rec mapType  (t:Type) : Type option = 
//    let f (xs: Type[]) = xs |> Array.map mapType |> Option.all
//    if t.IsArray then
//        t.GetElementType() |> mapType |> Option.map (fun t2 -> t2.MakeArrayType())
//    elif t |> FSharpType.IsTuple then
//        t |> FSharpType.GetTupleElements |> f |> Option.map (fun xs -> FSharpType.MakeTupleType xs)
//    elif FSharpType.IsUnion t || FSharpType.IsRecord t then
//        t.GetGenericArguments() |> f |> Option.map (fun xs -> t.GetGenericTypeDefinition().MakeGenericType(xs))
//    elif suportedBaseTypes |> Array.exists (fun x -> x.IsAssignableFrom(t)) then
//        Some(typeof<ValueInfo>)
//    else
//        None
//
//let isWhitelist (t:Type) = t.FullName |> fun fn -> whiteListNamespaces |> Array.exists (fn.StartsWith)
//
//let tryMapUnionCaseInfo (uci:UnionCaseInfo) = 
//    uci.DeclaringType.GenericTypeArguments 
//    |> Array.map mapType 
//    |> Option.all 
//    |> Option.map (fun ts -> 
//        uci.DeclaringType.GetGenericTypeDefinition().MakeGenericType(ts) |> FSharpType.GetUnionCases 
//        |> Seq.find (fun x -> x.Tag = uci.Tag))
//
//// TODO, change tryAssignable to support structual typing
//let tryAssignable (t:Type) = suportedBaseTypes |> Array.tryFind (fun x -> x.IsAssignableFrom(t))
//
//let mapVar (v1:Var) = 
//        match mapType(v1.Type) with 
//        | Some(t) -> Var(v1.Name,t)
//        | None -> v1 //failwithf "Var %s has type %s which is not mappable" v1.Name v1.Type.FullName
//
//module Map =
//    let addRange (xs:#seq<'a*'b>) (map: Map<'a,'b>)  =
//        xs |> Seq.fold (fun (map: Map<'a,'b>) (v1,v2) -> map.Add(v1,v2)) map
//
//type Expr with
//    static member Call(instanceO : Expr option, mi : Reflection.MethodInfo, args: Expr list) = 
//        match instanceO with | Some(x) -> Expr.Call(x,mi,args) | None -> Expr.Call(mi,args)
//
//let processExpr (graphExpr:Expr<Graph>) (expr: Expr) : Expr = 
//    let rec processExpr (varMap: Map<Var,Var>) (expr: Expr) : Expr = 
//        match expr with
//        | NewUnionCase (uci,args) ->
//            match tryMapUnionCaseInfo uci with
//            | Some(uci) -> Expr.NewUnionCase(uci, args |> List.map (processExpr varMap))
//            | None -> failwithf "Unable to process union type %A" uci
//        | Var v -> match varMap.TryFind(v) with | Some(v) -> Expr.Var(v) | _ -> failwithf "Var %s not found %s" v.Name v.Type.FullName
//        | VarSet(v1,body) -> v1 |> mapVar |> fun v2 -> Expr.VarSet(v2, processExpr (varMap.Add(v1,v2)) body)
//        | Lambda(v1,body) -> v1 |> mapVar |> fun v2 -> Expr.Lambda(v2, processExpr (varMap.Add(v1,v2)) body)
//        | Let(v1,exp1,exp2) -> v1 |> mapVar |> fun v2 -> Expr.Let(v2, processExpr varMap exp1, processExpr (varMap.Add(v1,v2)) exp2)
//        | LetRecursive(xs,body) ->  
//            let (es, vs1,vs2) = xs |> List.map fst |> fun x -> (xs |> List.map (snd >> processExpr varMap),x,x |> List.map mapVar)
//            Expr.LetRecursive((vs2,es) ||> List.zip, body |> processExpr (varMap |> Map.addRange ((vs1,vs2) ||> List.zip)))
//        | FieldGet (_,v) as expr ->
//            match tryAssignable v.FieldType with
//            | Some(u) -> Expr.Call(constantFunction.[u.FullName],[graphExpr; expr])
//            | None -> failwithf "Unsupported FieldGet %A as type is not assignable to a supported type" expr
//        | Call(_,mi,_) when mi.Name = "constant"-> failwith "we should have aborted progress before here, TODO remove this later"
//        | Coerce (_,t) -> 
//            match tryAssignable t with
//            | Some(u) -> Expr.Call(constantFunction.[u.FullName],[graphExpr; expr]) 
//            | None -> failwithf "Unsupported Coercions %A as type is not assignable to a supported type" expr
//        | Call(instanceO,mi,args) as expr ->
//            if isWhitelist mi.DeclaringType then
//                if mi.DeclaringType.FullName = ONNXAPIGraphFullName then
//                    failwithf "Graph member %s should not have been found at this stage" mi.Name
//                elif mi.DeclaringType.FullName = ONNXAPIFullName then
//                    match targetMethods.TryFind(mi.Name) with
//                    | Some(targetMethod) -> 
//                        let ys = 
//                            targetMethod.GetParameters().[1..] 
//                            |> Array.map (fun x -> x.ParameterType = typeof<ValueInfo> || x.ParameterType = typeof<ValueInfo option>)
//                            |> Array.toList
//                        let args = graphExpr.Raw :: ((ys,args) ||> List.zip |> List.map (fun (y,x) -> if y then processExpr varMap x else x))
//                        Expr.Call(instanceO,targetMethod, args)
//                    | None -> failwithf "Unsupported %s method %s" ONNXAPIFullName mi.Name
//                else 
//                    match mi.GetGenericArguments() |> Array.map  mapType |> Option.all with
//                    | Some(ts) -> 
//                        let mi' = mi.GetGenericMethodDefinition().MakeGenericMethod(ts)
//                        Expr.Call(instanceO,mi',args |> List.map (processExpr varMap))
//                    | None -> failwithf "Method with unsupported type %A" expr
//            else
//                match tryAssignable mi.DeclaringType with
//                | None -> failwithf "Unsupported type %s; it is neither whitelist or assignable to a supported tensor" mi.DeclaringType.FullName
//                | Some (u) -> Expr.Call(constantFunction.[u.FullName],[graphExpr; expr])
//        | ShapeVar _ -> failwithf "ShapeVar %A" expr
//        | ShapeLambda (v, expr) -> failwithf "ShapeLamda %A" expr
//        | ShapeCombination (o, xs) -> RebuildShapeCombination (o, xs |> List.map (processExpr varMap))
//    processExpr Map.empty expr 
//
//let eagerToGraph (f: Expr<'a -> 'b>) =
//    f
//    |> Expr.applyTransform ExprTransforms.expandWithReflectedDefinition 
//    |> Expr.applyTransform (ExprTransforms.reduceApplicationsAndLambdas true) 
//    |> processExpr <@ Graph.Default() @>
