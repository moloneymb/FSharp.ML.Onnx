module Common

open System
open System.Reflection
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors
open ProtoBuf
open Microsoft.FSharp.Reflection
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape
open Microsoft.FSharp.Quotations.Patterns
open System.Text.RegularExpressions

let (|TryFunc|_|) (f: 'a -> 'b option ) (x:'a) = f x
let (|Found|_|) map key = map |> Map.tryFind key

module Option =
    let all (xs: #seq<'a option>) = 
        let xs = xs |> Seq.toArray
        let ys = xs |> Array.choose id
        if xs.Length = ys.Length then Some(ys) else None

//open Fantomas
//let private regexs = [| Regex(@"Operators\.string\<[_A-Za-z0-9]*\>"), "string"; Regex(@"\(unitVar[0-9]* : Unit\)"), "()" |]
//
//
//let sprintExpr (expr:Expr) = 
//    // TODO figure out ommit Enclosing Types
//    Falanx.Machinery.Quotations.ToAst(expr,knownNamespaces = set ["Operators"])
//    |> snd
//    |> fun x -> CodeFormatter.FormatAST(x, "helloWorld.fs",  None, FormatConfig.FormatConfig.Default) 
//    |> Falanx.Machinery.Expr.cleanUpTypeName
//    |> fun x -> 
//        let xs = x.Substring(31).Replace("\r\n","\n").Split([|'\n';'\r'|]) 
//        if xs.[0].Contains(";") then xs.[0].Substring(33) 
//        else 
//            (xs.[2..] |> Array.choose (fun x -> if x.Length > 3 then Some(x.Substring(4)) else None) |> String.concat "\n")
//        |> fun x -> x.Trim(' ','\n','\r')
//        |> fun x -> (x,regexs) ||> Array.fold (fun (x:string) ((r,s):Regex*string) -> r.Replace(x,s))
//

module List =
    let chop n xs =
        if n < 0 then invalidArg "n" "n must not be negative"
        let rec split l =
            match l with
            | 0, xs -> [], xs
            | n, x :: xs ->
                let front, back = split (n-1, xs)
                x :: front, back
            | _, [] -> failwith "List.chop: not enough elts list"
        split (n, xs)

    /// TODO, do a better name and more efficent implementation
    let splitWhen (f : 'a -> bool) (xs: 'a list) = (xs |> List.takeWhile f, xs |> List.skipWhile f)

    let reshape  (shape: int list) (xs: 'a list) : 'a list list =
        (([],xs),shape) ||> List.fold (fun (acc,xs) count -> chop (max count xs.Length) xs |> fun (head,xs) -> (head::acc,xs)) |> fst |> List.rev


module Expr = 


    let rec flatten (expr: Expr) : Expr seq = 
        match expr with
        | ShapeVar _ -> Seq.singleton expr
        | Let(_,exp1,exp2) -> seq {yield expr; yield! flatten exp1; yield! flatten exp2}
        | LetRecursive(xs,body) -> seq {yield expr; yield! xs |> Seq.collect (snd >> flatten); yield! flatten body}
        | ShapeLambda (_, exp) -> seq { yield expr; yield! flatten exp}
        | ShapeCombination (_, exprs) ->  seq { yield expr; yield! exprs |> List.toSeq |> Seq.collect flatten}

    let rec visitor (f: 'a -> Expr -> ('a*Expr)) (state: 'a) (expr: Expr) : ('a*Expr) =
        let swap (x,y) = (y,x)
        let fold (acc:'a) (xs:Expr list) = xs |> List.mapFold (fun acc e -> visitor f acc e |> swap) acc |> swap
        let (acc,expr) = f state expr
        match expr with
        | ShapeVar _ -> (acc,expr)
        | Let(v,exp1,exp2) -> [exp1; exp2] |> fold acc |> fun (acc,xs) -> (acc, Expr.Let(v,xs.[0],xs.[1]))
        | LetRecursive(xs,body) ->  
            let (acc, ys) = xs |> List.map snd |> fold acc 
            let (acc, body) = visitor f acc body
            (acc,Expr.LetRecursive((xs |> List.map fst, ys) ||> List.zip, body))
        | ShapeLambda (v, expr) -> visitor f acc expr |> fun (x,y) -> (x, Expr.Lambda (v, y))
        | ShapeCombination (o, xs) -> 
            let (ys,acc) = xs |> List.mapFold (fun (acc:'a) (e:Expr) -> visitor f acc e |> swap) acc
            (acc, RebuildShapeCombination (o, ys))

    // TODO enable early stopping of visitor with a flag?
    let rec earyStopVisitor (f: 'a -> Expr -> ('a*Expr*bool)) (state: 'a) (expr: Expr) : ('a*Expr) =
        let swap (x,y) = (y,x)
        let fold (acc:'a) (xs:Expr list) = xs |> List.mapFold (fun acc e -> earyStopVisitor f acc e |> swap) acc |> swap
        let (acc,expr,earyStop) = f state expr
        if earyStop then (acc,expr) 
        else
            match expr with
            | ShapeVar _ -> (acc,expr)
            | Let(v,exp1,exp2) -> [exp1; exp2] |> fold acc |> fun (acc,xs) -> (acc, Expr.Let(v,xs.[0],xs.[1]))
            | LetRecursive(xs,body) ->  
                let (acc, ys) = xs |> List.map snd |> fold acc 
                let (acc, body) = earyStopVisitor f acc body
                (acc,Expr.LetRecursive((xs |> List.map fst, ys) ||> List.zip, body))
            | ShapeLambda (v, expr) -> earyStopVisitor f acc expr |> fun (x,y) -> (x, Expr.Lambda (v, y))
            | ShapeCombination (o, xs) -> 
                let (ys,acc) = xs |> List.mapFold (fun (acc:'a) (e:Expr) -> earyStopVisitor f acc e |> swap) acc
                (acc, RebuildShapeCombination (o, ys))


    let map (f: Expr -> Expr) (expr: Expr) = visitor (fun _ e -> ((),f(e))) () expr |> snd

    let unfoldWhileChangedWithEarlyStop (f: Expr -> bool*Expr option) (expr: Expr) =
        Seq.unfold (fun expr -> 
            match earyStopVisitor (fun state expr -> match f(expr) with | r,Some(e) -> (true,e,r) | r,_ -> (state,expr,r)) false expr with
            | false, _ -> None
            | true, expr -> Some(expr,expr)) expr

    let unfoldWhileChanged (f: Expr -> Expr option) (expr: Expr) =
        Seq.unfold (fun expr -> 
            match visitor (fun state expr -> match f(expr) with | Some(e) -> (true,e) | _ -> (state,expr)) false expr with
            | false, _ -> None
            | true, expr -> Some(expr,expr)) expr

    let rec count (f: Expr -> bool) (expr: Expr) : int = 
        expr |> flatten |> Seq.sumBy (fun x -> if f x then 1 else 0)

    let replace (x: Expr) (y:Expr) (expr:Expr) = map (fun t -> if t = x then y else t) expr

    let isApplication (x:Expr) = match x with | Application(_,_) -> true | _ -> false
    let isLambda (x:Expr) = match x with | Lambda(_,_) -> true | _ -> false

    let getVarsUsage (expr: Expr) = expr |> flatten |> Seq.choose (function | Var(v) -> Some(v) | _ -> None) |> Seq.toArray

    /// To find expressions that are assigned but are not used
    //let getVarsAssignment (expr: Expr) = expr |> flatten |> Seq.choose (function | Lambda(v,_) | Let(v,_,_) -> Some(v) | _ -> None) |> Seq.toArray

    let getCallNames (expr: Expr) = expr |> flatten |> Seq.choose (function | Call(_,m,_) -> Some(m.Name) | _ -> None) |> Set
    let tryFirstCall (expr: Expr) = expr |> flatten |> Seq.tryFind (function | Call(_,_,_) -> true | _ -> false)
    let firstCall (expr: Expr) = expr |> flatten |> Seq.tryFind (function | Call(_,_,_) -> true | _ -> false) |> Option.get
    let countVar (v: Var) expr = count (function | ShapeVar v2 -> v = v2 | _ -> false) expr

    /// This gets the method and makes it generic 
    let tryGetGenericMethod (expr: Expr) = 
        expr 
        |> tryFirstCall
        |> Option.bind (function | Call(_,mi,_) -> Some(mi) | _ -> None)
        |> Option.map (fun mi -> if mi.IsGenericMethod then mi.GetGenericMethodDefinition() else mi)

    let applyTransform (f: Expr -> Expr option) (expr: Expr) =
        expr 
        |> unfoldWhileChanged f
        |> Seq.tryLast 
        |> Option.defaultValue expr


    // TODO better name
    let merge<'a>(xs: Expr list) = Expr.Cast<'a[]>(Expr.NewArray(typeof<'a>, xs))

    let applyTransforms (fs:(Expr -> Expr option)[])  = 
        let cmb (fs:(Expr -> Expr option)[]) (expr: Expr) = 
            seq { for f in fs do match f expr with | Some(x) -> yield x | _ -> ()} |> Seq.tryHead
        applyTransform (cmb fs)

type Expr with
    member this.TryGetLocation() =
        this.CustomAttributes 
        |> List.choose (function 
            | NewTuple [String "DebugRange"; NewTuple [String file; Int32 a; Int32 b; Int32 c; Int32 d]] -> Some(file, a, b, c, d) 
            | _ -> None)
        |> List.tryHead

    static member Lambdas(vars: Var list list, body: Expr) : Expr = 
        let makeTupledLambda (vars : Var list) (body: Expr) : Expr =
            match vars with
            | [] ->  Expr.Lambda(Var("unitVar", typeof<unit>), body)
            | [x] -> Expr.Lambda(x,body)
            | _ -> 
                let tuple = Var("tupledArg",Microsoft.FSharp.Reflection.FSharpType.MakeTupleType([|for v in vars -> v.Type|]))
                Expr.Lambda(tuple,(vars |> List.indexed, body) ||> List.foldBack (fun (i,v) state -> Expr.Let(v,Expr.TupleGet(Expr.Var(tuple),i),state)))
        (vars,  body) ||> List.foldBack makeTupledLambda

module ExprTransforms = 

    /// A collection of simple transforms
    /// These are probably superseded by more advanced transforms
    module Simple = 
        /// Simplifies a common binding patterns
        let bindings = 
            function 
            | Patterns.Application(ExprShape.ShapeLambda(v, body), assign) 
            | Patterns.Let(v, assign, body) ->
                match Expr.countVar v body with
                | 0 -> Some(body)
                | 1 -> Some(Expr.map (function | ShapeVar v2 when v = v2 -> assign | x -> x) body)
                | _ -> None
            | _ -> None

        /// Removes construcing and deconstructing a tuple which is only used once
        /// This can happen when a Expr.Var is substitued by an Expr.NewTuple and only one item from the tuple is used
        let tuples = 
            function 
            | TupleGet(NewTuple(exp),index) when index < exp.Length -> Some(exp.[index]) 
            | _ -> None

        /// TODO generalize to beyond immediate Tuple decomposition
        let newTuple = 
            function
            | Application(NewTuple(args),Lambda(patternInput,body))
            | Let(patternInput,NewTuple(args),body) ->
                let rec f =
                    function
                    | Let(v2,TupleGet(Var v3 ,index ),body) when  v3 = patternInput && index < args.Length -> 
                        (v2,args.[index],body) :: f body 
                    | _ -> []
                match f body with
                | [] -> None
                | ys ->
                    let (_,_,body) = ys |> List.last 
                    if Expr.countVar patternInput body > 0 then None
                    else Some((ys, body) ||> List.foldBack (fun (v,e,_) body -> Expr.Let(v,e,body)))
            | _ -> None

        /// NOTE find out if this style of SpecificCall is slow
        let builtIns = function
            | SpecificCall <@ (|>) @> (None,_,[arg;func]) 
            | SpecificCall <@ (<|) @> (None,_,[func;arg]) -> Some(Expr.Application(func,arg))
            | SpecificCall <@ (||>) @> (None,_,[arg1;arg2;func]) 
            | SpecificCall <@ (<||) @> (None,_,[func;arg1;arg2]) -> Some(Expr.Applications(func,[[arg1];[arg2]]))
            | SpecificCall <@ (|||>) @> (None,_,[arg1;arg2;arg3;func]) 
            | SpecificCall <@ (<|||) @> (None,_,[func;arg1;arg2;arg3]) -> Some(<@@ (%%func) %%arg1 %%arg2 %%arg3 @@>) 
            | SpecificCall <@ id @> (None,_,[arg]) -> Some(arg)
            | SpecificCall <@ fst @> (None,_,[arg]) -> Some(Expr.TupleGet(arg,0))
            | SpecificCall <@ snd @> (None,_,[arg]) -> Some(Expr.TupleGet(arg,1))
            //| SpecificCall <@ ignore @> (None,_,[arg]) -> Some(Expr.Lambda( Value.n)) // Lambda(_arg1, Value (<null>))
            | _ -> None

        let selfMatch (expr: Expr) = 
            match expr with
            //| Lambda(v1,Var(v2)) when v1 = v2 -> Some(Expr.Var(v1))
            | NewTuple([]) -> None
            | NewTuple(xs) ->
                xs 
                |> List.map (function | TupleGet(x,i) -> Some(x,i) | _ -> None) 
                |> Option.all
                |> Option.bind (fun xs -> 
                    let isSelf, _ = ((true,0),xs) ||> Array.fold (fun (v,i1) (x,i2) -> (v && i1 = i2 && x = fst xs.[0],i1+1))
                    if isSelf then Some(fst xs.[0]) else None)
            | _ -> None
    /// Combines an array of transform functions and performs the transforms recursively at the same time in the order given
    //let groupUnfold (fs: (Expr -> Expr option)[]) (expr: Expr) =
    //    expr |> Expr.unfoldWhileChanged (fun x -> fs |> Seq.choose (fun f -> f(x)) |> Seq.tryHead)

    /// This replaces function calls with the reflected definition
    /// TODO extend this to include PropertyGet and PropertySet (?)
//    let expandCalls  = 
//        function 
//        | Patterns.Call(instanceO, MethodWithReflectedDefinition (Lambdas(yss,_) as rd), zs) ->
//            // WARN instanceO is not handled and 
//            Some(Expr.Applications(rd,zs |> List.reshape [for ys in yss -> ys.Length] |> List.map (List.filter (fun z -> z.Type <> typeof<unit>))))
//        | _ -> None

//match <@ (|>) @> |> Expr.tryFirstCall |> Option.get with
////| SpecificCall <@ (|>) @> (None,_,[arg;func]) -> Some(Expr.Application(func,arg))
//| SpecificCall <@ (|>) @> (None,_,[arg;func]) -> Some(<@@ (%%func) %%arg @@>)
//| _ -> failwith "todo"
//|> Option.get


//let pipeCalls = function
//    | SpecificCall <@ (|>) @> (None,_,[arg;func]) 
//    | SpecificCall <@ (<|) @> (None,_,[func;arg]) -> Some(Expr.Application(func,arg))
//    | SpecificCall <@ (||>) @> (None,_,[arg1;arg2;func]) 
//    | SpecificCall <@ (<||) @> (None,_,[func;arg1;arg2]) -> Some(Expr.Applications(func,[[arg1];[arg2]]))
//    | SpecificCall <@ (|||>) @> (None,_,[arg1;arg2;arg3;func]) 
//    //| SpecificCall <@ (<|||) @> (None,_,[func;arg1;arg2;arg3]) -> Some(<@@ (%%func) %%arg1 %%arg2 %%arg3 @@>) //Some(Expr.Applications(func,[[arg1];[arg2];[arg3]]))
//    | SpecificCall <@ (<|||) @> (None,_,[func;arg1;arg2;arg3]) -> Some(<@@ (%%func) %%arg1 %%arg2 %%arg3 @@>) //Some(Expr.Applications(func,[[arg1];[arg2];[arg3]]))
//    | _ -> None

//    let expand  (f: MethodInfo -> Expr option) (g: PropertyInfo -> Expr option) (expr: Expr)=
//        match expr with
//        | PropertyGet(instanceO,TryFunc g (Lambdas(yss,_) as rd),zs) 
//        //| Call(None, TryFunc f (Lambdas (yss,_) as rd), zs) 
//        | Patterns.Call(instanceO, TryFunc f (Lambdas(yss,_) as rd), zs) ->
//            // Reshape will stop short of filling new 'shape' when there are not enough elements
//            // This happens when Lambdas pattern matches with more Lambdas than there are Call parameters 
//            // This is because there is no early stopping on the Lambdas pattern
//            let reshape  (shape: int list) (xs: 'a list) : 'a list list =
//                (([],xs),shape) ||> List.fold (fun (acc,xs) count -> 
//                    if xs.Length = 0 then (acc,[])
//                    elif xs.Length < count then (xs::acc,[])
//                    else List.chop count  xs |> fun (head,xs) -> (head::acc,xs))
//                |> fst |> List.rev
//            match instanceO,zs,yss with
//            // special case single unit parameters which are empty
//            | None,[], [y]::_ when y.Type = typeof<unit> -> Expr.Applications(rd,[[]])
//            | Some(instance),[], [x]::[y]::_ when instance.Type = x.Type && y.Type = typeof<unit> -> Expr.Applications(rd,[[instance];[]])
//            | None,[], _ -> failwithf "found an unexpected value %A" rd
//            | Some(instance),_,(_::yss) -> Expr.Applications(rd,[instance] :: (zs |> reshape [for ys in yss -> ys.Length]))
//            | None,_,_ -> Expr.Applications(rd,zs |> reshape [for ys in yss -> ys.Length])
//            | _,_,_ -> failwithf "found an unexpected value %A " rd
//            |> Some
//        | _ -> None

//    let expandWithReflectedDefinition (expr: Expr) = expr |> expand (``|MethodWithReflectedDefinition|_|``) (``|PropertyGetterWithReflectedDefinition|_|``)
    let expandWithReflectedDefinition  = //|MethodWithReflectedDefinition|_|PropertyGetterWithReflectedDefinition|_|``)
        function
        | PropertyGet(instanceO,PropertyGetterWithReflectedDefinition (Lambdas(yss,_) as rd),zs) 
        //| Call(None, TryFunc f (Lambdas (yss,_) as rd), zs) 
        | Patterns.Call(instanceO, MethodWithReflectedDefinition (Lambdas(yss,_) as rd), zs) ->
            // Reshape will stop short of filling new 'shape' when there are not enough elements
            // This happens when Lambdas pattern matches with more Lambdas than there are Call parameters 
            // This is because there is no early stopping on the Lambdas pattern
            let reshape  (shape: int list) (xs: 'a list) : 'a list list =
                (([],xs),shape) ||> List.fold (fun (acc,xs) count -> 
                    if xs.Length = 0 then (acc,[])
                    elif xs.Length < count then (xs::acc,[])
                    else List.chop count  xs |> fun (head,xs) -> (head::acc,xs))
                |> fst |> List.rev
            match instanceO,zs,yss with
            // special case single unit parameters which are empty
            | None,[], [y]::_ when y.Type = typeof<unit> -> Expr.Applications(rd,[[]])
            | Some(instance),[], [x]::[y]::_ when instance.Type = x.Type && y.Type = typeof<unit> -> Expr.Applications(rd,[[instance];[]])
            | None,[], _ -> failwithf "found an unexpected value %A" rd
            | Some(instance),_,(_::yss) -> Expr.Applications(rd,[instance] :: (zs |> reshape [for ys in yss -> ys.Length]))
            | None,_,_ -> Expr.Applications(rd,zs |> reshape [for ys in yss -> ys.Length])
            | _,_,_ -> failwithf "found an unexpected value %A " rd
            |> Some
        | _ -> None

//        | PropertyGet(instanceO,PropertyGetterWithReflectedDefinition (Lambdas(yss,_) as rd),zs) 
//        //| Call(None, TryFunc f (Lambdas (yss,_) as rd), zs) 
//        | Patterns.Call(instanceO, MethodWithReflectedDefinition (Lambdas(yss,_) as rd), zs) ->

    /// This simplifies applying exprs into lambdas vars where vars are either not used or only used once
    /// TODO / WARN we're removing unit applications which will cause issues when there is side-effectful code
    /// It could be possible to check if side-effectful code is in the body and use this to determine if to keep
    let reduceApplicationsAndLambdas removeUnits  (expr: Expr) =
        let f (xss : Expr list list,yss,body) =
            // TODO check if any changes can be made, if not then short circuit back
            let (yss, remaining) = yss |> List.chop xss.Length
            let body = Expr.Lambdas(remaining,body)
            let pairs = (xss,yss) ||> List.zip 
            let checkMatch = 
                pairs |> List.exists (function 
                    | ([],[_]) -> false 
                    | (xs,ys) -> 
                        (xs.Length = ys.Length) && ((xs,ys) 
                        ||> List.zip |> List.exists (fun (x,y) -> x.Type <> y.Type))) 
            if checkMatch then failwith "Non matching Applications and Lambdas - Not sure if this will ever happen"
            let varCounts = body |> Expr.getVarsUsage |> Array.countBy id |> Map.ofArray
            let varMapping = pairs |> List.collect (function | ([],[_]) -> [] | (xs,ys) -> (xs,ys) ||> List.zip |> List.map (fun (x,y) -> (y,x))) |> Map.ofList
            let singleVars = varMapping |> Map.filter (fun k _ -> varCounts.TryFind(k) = Some(1))
            let body = body |> Expr.map (function | ShapeVar (Found singleVars x) ->  x | x -> x)
            let filterVars = set [ for KeyValue(k,_) in varMapping do match varCounts.TryFind(k) with | None | Some(1) -> yield k | _ -> () ]
            let (xssCount,yssCount) = (xss |> List.sumBy List.length,yss |> List.sumBy List.length)
            let (xss,yss) = 
                pairs 
                |> List.map (function 
                    | [],[_] as x -> (if removeUnits then [],[] else x) 
                    | zs -> zs ||> List.zip |> List.filter (fun (_,v) -> filterVars.Contains v |> not) |> List.unzip)
                |> List.filter (function | [],[] -> false | _ -> true)
                |> List.unzip
            let (xssCount1,yssCount1) = (xss |> List.sumBy List.length,yss |> List.sumBy List.length)
            if (xssCount,yssCount) = (xssCount1,yssCount1) then None else Some (xss,yss,body)

        match expr with
        //| Applications(Lambdas(yss,body),xss) ->  
        | Applications(Let(objectArg, expr,Lambdas(yss,body)),xss) ->  
            f (xss, yss, body) |> Option.map (fun (xss,yss,body) -> Expr.Applications(Expr.Let(objectArg,expr,Expr.Lambdas(yss,  body)),xss))
        | Applications(Lambdas(yss,body),xss) ->  
            f (xss, yss, body) |> Option.map (fun (xss,yss,body) -> Expr.Applications(Expr.Lambdas(yss,  body) ,xss))
        | _ -> None

    /// This takes a tupled function application and turns it into a curried function
    /// This was used expanding calls without tuples. After reduceApplicationAndLambdas was implemented this function was no longer needed
    let toCurried =
        function
        | Lambdas(xss : Var list list,body) when xss.Length > 1 -> Some(Expr.Lambdas([for xs in xss do for x in xs -> [x]],body))
        | _ -> None

    /// Old toCuried function as a reference in case it's needed again
    //let toCurried paramCount (exp: Expr) : Expr =
    //    if paramCount = 0 then exp
    //    else
    //        let rec getApp paramCount (exp : Expr) : (Var*Expr)[] =
    //            if paramCount = 0 then [||]
    //            else
    //                match exp with 
    //                | Lambda(v,exp) -> 
    //                    if FSharpType.IsTuple v.Type then
    //                        let lIndex = (FSharpType.GetTupleElements v.Type |> Array.length) - 1
    //                        (exp,[|0..lIndex|]) 
    //                        ||> Array.mapFold (fun state _ -> match state with | Let(x,_,z) -> ((x,z),z) | x -> failwithf "expected Let Expr %A" x) |> fst
    //                        |> fun xs -> [| yield! xs; yield! getApp (paramCount - 1 - lIndex) (snd xs.[lIndex]) |]
    //                    else
    //                        [|yield (v,exp); yield! getApp (paramCount - 1) exp|]
    //                | _ -> failwithf "err %A" exp
    //        (getApp paramCount exp) |> fun xs -> ((xs |> Array.map fst), xs |> Seq.last |> snd) ||> Array.foldBack (fun v e -> Expr.Lambda(v,e))

    /// This pre-computes Math when used on literals
    /// NOTE Will consider expanding this if it could be useful
//    let evaulateMath =
//        function 
//        | SpecificCall <@ (*) @> (None,_,[exp1;exp2]) as exp -> 
//            match exp1,exp2 with 
//            | Int32(a),Int32(b) -> Some(Expr.Value(a * b)) 
//            | Int64(a),Int64(b) -> Some(Expr.Value(a * b)) 
//            | Single(a),Single(b) -> Some(Expr.Value(a * b)) 
//            | Double(a),Double(b) -> Some(Expr.Value(a * b)) 
//            | Value(_),Value(_) -> Expr.Value(exp.EvaluateUntyped())
//            | _ -> None
//        | _ -> None


       /// Alternative method to simplifying values would be to have a white list of functions where every expression for which the sub expression is white list is evaluated
//       let evaluateMath2 = 
//            function
//            | Value v -> None
//            | _ when not (expr |> Expr.flatten |> Seq.map (function | SpecificCall <@ (+) @> _ -> true | SpecificCall <@ (*) @> _ -> false | _ -> false ) |> Seq.exists id) -> 
//                Some(Expr.Value(expr.EvaluateUntyped(),expr.Type))
//            | _ -> None)  
//            |> Seq.map sprintExpr


/// This was to make some error messages more readable
/// The root cause of these errors have been fixed so this is not needed for now
//
//module ErrorSimplifier = 
//    open FParsec
//
//    let identifier =
//        let isIdentifierFirstChar c = isLetter c || c = '_'
//        let isIdentifierChar c = isLetter c || isDigit c || c = '_'
//        many1Satisfy2L isIdentifierFirstChar isIdentifierChar "identifier" .>> spaces
//
//    let identifierWithNamespaces : Parser<string list, unit> = 
//        sepBy1 identifier (pchar '.' <|> pchar '+')
//
//    type TypeAST = 
//        | Base of string list
//        | Generic of string list * TypeAST list
//
//    let pvalue, pvalueRef = createParserForwardedToRef<TypeAST, unit>()
//
//    let pbase = identifierWithNamespaces |>> TypeAST.Base
//
//    let pgeneric : Parser<TypeAST,unit> = 
//        identifierWithNamespaces 
//        .>>. (pchar '`'
//        >>. pint32
//        >>. pchar '['
//        >>. (sepBy1 pvalue (pchar ','))
//        .>> pchar ']')
//        |>> fun (x,y) -> TypeAST.Generic(x,y)
//
//    do pvalueRef :=  (attempt pgeneric) <|> pbase 
//
//    let rec printTypeAst (ast: TypeAST) =
//        match ast with
//        | TypeAST.Base(names) ->
//            match names with 
//            | ["System";"String"] -> "string"
//            | ["System";"Int32"] -> "int"
//            | ["System";"Boolean"] -> "bool"
//            | ["Microsoft";"FSharp"; "Core"; "Unit"] -> "()"
//            | _ -> names |> String.concat "\."
//        | TypeAST.Generic(names,xs) -> 
//            match names with
//            | ["FSharp"; "Quotations"; "Evaluator"; "Tools"; "FuncFSharp"] 
//            | ["Microsoft"; "FSharp"; "Core"; "FSharpFunc"] -> 
//                xs |> List.map printTypeAst |> String.concat " -> "
//            | ["System"; "Tuple"] -> 
//                "(" + (xs |> List.map printTypeAst |> String.concat "*") + ")"
//            | _ -> (names |> String.concat ".") + "`" + (string xs.Length) + "[" + ((xs |> List.map printTypeAst) |> String.concat ",") + "]"
//
//    let processMsg (msg:string) = 
//        let printErrorMsgAST (xs: Choice<string,TypeAST>[]) = xs |> Array.map (function | Choice1Of2 x -> x | Choice2Of2 x -> sprintf "(%s)" <| printTypeAst x) |> String.concat ""
//        let tpe = between (pchar '\'') (pchar '\'') pvalue 
//        let txt = Choice1Of2
//        let typ = Choice2Of2
//        let errMsg1 = " cannot be used for parameter of type " |> fun t1 -> ((tpe .>> pstring t1 .>>. tpe ) |>> fun (a,b) -> [|typ a; txt t1; typ b|])
//        match run (choice [errMsg1]) msg with
//        | Success(xs,_,_)   -> printErrorMsgAST(xs)
//        | Failure(errorMsg, _, _) -> failwithf "Failure: %s" errorMsg
//



type on = ONNXAPI.ONNX

type ONNXAPI.ONNX with
    [<ReflectedDefinition>]
    static member reshape(x: Tensor<float32>,shape: int32[]) = on.reshape(x,(shape |> Array.map int64).ToTensor())

module ExprRun = 
    [<RequireQualifiedAccess>]
    type BT = 
        | Unknown // Tensor without a generic type argument
        | Float64
        | Float32
        | Int64
        | Int32
        | Int16
        | Int8
        | UInt64
        | UInt32
        | UInt16
        | UInt8
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
    type TT = 
        | DenseTensor
        | SparseTensor
        | Tensor
        | Unknown

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



module ExprGraph = 
    open FSharp.Quotations.Evaluator
    open Microsoft.FSharp.Quotations
    open Microsoft.FSharp.Quotations.Patterns
    open Microsoft.FSharp.Quotations.DerivedPatterns
    open Microsoft.FSharp.Quotations.ExprShape
    open Microsoft.ML.OnnxRuntime.Tensors
    open Onnx
    open ProtoBuf
    open System
    open System.IO
    open Microsoft.FSharp.Reflection

    let filterMethods (m: Reflection.MethodInfo) =
        match m.Name with
        | "Equals"
        | "GetHashCode"
        | "GetType"
        | "ToString" -> false
        | _ -> true

    let onMethods = 
        typeof<on>.GetMethods() 
        |> Array.filter filterMethods

    let targetMethods = 
        typeof<ONNXAPIGraph.ONNXGraph>.GetMethods() 
        |> Array.filter filterMethods
        |> Array.map (fun mi -> mi.Name,mi)
        |> Map.ofArray

    let constantFunction = 
        typeof<Constants>.GetMethods() 
        |> Array.filter filterMethods
        |> Array.map (fun mi -> (mi.GetParameters().[1].ParameterType.FullName,mi)) 
        |> Map.ofArray

    let ONNXAPIFullName = typeof<ONNXAPI.ONNX>.FullName
    let ONNXAPIGraphFullName = typeof<ONNXAPIGraph.ONNXGraph>.FullName

    let whiteListNamespaces =
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

    let rec mapType  (t:Type) : Type option = 
        let f (xs: Type[]) = xs |> Array.map mapType |> Option.all
        if t.IsArray then
            t.GetElementType() |> mapType |> Option.map (fun t2 -> t2.MakeArrayType())
        elif t |> FSharpType.IsTuple then
            t |> FSharpType.GetTupleElements |> f |> Option.map (fun xs -> FSharpType.MakeTupleType xs)
        elif FSharpType.IsUnion t || FSharpType.IsRecord t then
            t.GetGenericArguments() |> f |> Option.map (fun xs -> t.GetGenericTypeDefinition().MakeGenericType(xs))
        elif suportedBaseTypes |> Array.exists (fun x -> x.IsAssignableFrom(t)) then
            Some(typeof<ValueInfo>)
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
            | None -> v1 //failwithf "Var %s has type %s which is not mappable" v1.Name v1.Type.FullName

    module Map =
        let addRange (xs:#seq<'a*'b>) (map: Map<'a,'b>)  =
            xs |> Seq.fold (fun (map: Map<'a,'b>) (v1,v2) -> map.Add(v1,v2)) map

    type Expr with
        static member Call(instanceO : Expr option, mi : Reflection.MethodInfo, args: Expr list) = 
            match instanceO with | Some(x) -> Expr.Call(x,mi,args) | None -> Expr.Call(mi,args)

    let processExpr (graphExpr:Expr<Graph>) (expr: Expr) : Expr = 
        let rec processExpr (varMap: Map<Var,Var>) (expr: Expr) : Expr = 
            match expr with
            | NewUnionCase (uci,args) ->
                match tryMapUnionCaseInfo uci with
                | Some(uci) -> Expr.NewUnionCase(uci, args |> List.map (processExpr varMap))
                | None -> failwithf "Unable to process union type %A" uci
            | TupleGet(x,i) -> Expr.TupleGet(processExpr varMap x,i)
            | NewTuple(xs) -> Expr.NewTuple(xs |> List.map (processExpr varMap))
            | Var v -> match varMap.TryFind(v) with | Some(v) -> Expr.Var(v) | _ -> failwithf "Var %s not found %s" v.Name v.Type.FullName
            | VarSet(v1,body) -> v1 |> mapVar |> fun v2 -> Expr.VarSet(v2, processExpr (varMap.Add(v1,v2)) body)
            | Lambda(v1,body) -> v1 |> mapVar |> fun v2 -> Expr.Lambda(v2, processExpr (varMap.Add(v1,v2)) body)
            | Let(v1,exp1,exp2) -> v1 |> mapVar |> fun v2 -> Expr.Let(v2, processExpr varMap exp1, processExpr (varMap.Add(v1,v2)) exp2)
            | LetRecursive(xs,body) ->  
                let (es, vs1,vs2) = xs |> List.map fst |> fun x -> (xs |> List.map (snd >> processExpr varMap),x,x |> List.map mapVar)
                Expr.LetRecursive((vs2,es) ||> List.zip, body |> processExpr (varMap |> Map.addRange ((vs1,vs2) ||> List.zip)))
            | FieldGet (_,v) as expr ->
                match tryAssignable v.FieldType with
                | Some(u) -> Expr.Call(constantFunction.[u.FullName],[graphExpr; expr])
                | None -> failwithf "Unsupported FieldGet %A as type is not assignable to a supported type" expr
            | Call(_,mi,_) when mi.Name = "constant"-> failwith "we should have aborted progress before here, TODO remove this later"
            | Coerce (_,t) -> 
                match tryAssignable t with
                | Some(u) -> Expr.Call(constantFunction.[u.FullName],[graphExpr; expr]) 
                | None -> failwithf "Unsupported Coercions %A as type is not assignable to a supported type" expr
            | Call(instanceO,mi,args) as expr ->
                if isWhitelist mi.DeclaringType then
                    if mi.DeclaringType.FullName = ONNXAPIGraphFullName then
                        failwithf "Graph member %s should not have been found at this stage" mi.Name
                    elif mi.DeclaringType.FullName = ONNXAPIFullName then
                        match targetMethods.TryFind(mi.Name) with
                        | Some(targetMethod) -> 
                            let ys = 
                                targetMethod.GetParameters().[1..] 
                                |> Array.map (fun x -> x.ParameterType = typeof<ValueInfo> || x.ParameterType = typeof<ValueInfo option>)
                                |> Array.toList
                            let args = graphExpr.Raw :: ((ys,args) ||> List.zip |> List.map (fun (y,x) -> if y then processExpr varMap x else x))
                            Expr.Call(instanceO,targetMethod, args)
                        | None -> failwithf "Unsupported %s method %s" ONNXAPIFullName mi.Name
                    else 
                        match mi.GetGenericArguments() |> Array.map  mapType |> Option.all with
                        | Some(ts) -> 
                            let mi' = mi.GetGenericMethodDefinition().MakeGenericMethod(ts)
                            Expr.Call(instanceO,mi',args |> List.map (processExpr varMap))
                        | None -> failwithf "Method with unsupported type %A" expr
                else
                    match tryAssignable mi.DeclaringType with
                    | None -> failwithf "Unsupported type %s; it is neither whitelist or assignable to a supported tensor" mi.DeclaringType.FullName
                    | Some (u) -> Expr.Call(constantFunction.[u.FullName],[graphExpr; expr])
            | ShapeVar _ -> failwithf "ShapeVar %A" expr
            | ShapeLambda (v, expr) -> failwithf "ShapeLamda %A" expr
            | ShapeCombination (o, xs) -> RebuildShapeCombination (o, xs |> List.map (processExpr varMap))
        processExpr Map.empty expr 

    // TODO move and rename
    // TODO also add some logging to make debugging easier
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
        |> Expr.applyTransform ExprTransforms.Simple.selfMatch


    // This only supports simple records
    // TODO: Whitelist records or... Blacklist records?
    // figure out when to stop applying the transform so that we can continue to use records outside of ONNX Graph API
    (* (records:Map<string,Reflection.PropertyInfo[]>) (fieldOffsets:Map<string*string,int>) *)
    // Must also map tuple types

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

open FSharp.Quotations.Evaluator
open Onnx
open ExprRun
module Foo = 
    let wrapGraph<'a,'b>(expr: Expr<'a -> 'b>)  : DV<'a -> DV<'b>>= 
        let mmIn = getMM (typeof<'a>)
        let mmOut = getMM (typeof<'b>)
        let buildGraph(func: Expr<'a->'b>) (mmIn:MM) (mmOut:MM) : ValueInfo[]*ValueInfo[]*ModelProto =
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
                |> Expr.merge<ValueInfo>

            let graph = Graph.Default()

            let transformedFunc = 
                func 
                |> ExprGraph.simplify 
                |> Expr.applyTransform  ExprGraph.mapRecordsToTuples 
                |> ExprGraph.processExpr <@ graph @>
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

        let inputs,outputs,model = buildGraph(expr) mmIn mmOut
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
            let flatInputs = Expr.Lambda(v,(getNamedValues(0,Expr.Var(v),mmIn) |> snd |> Array.map (fun x -> x.Raw) |> List.ofArray  |> Expr.merge<NamedOnnxValue>)).EvaluateUntyped() :?> 'a -> NamedOnnxValue[]
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
            new DV<'b>(results3, fun () ->results.Dispose())
        new DV<'a -> DV<'b>> (partialRun, fun () -> sess.Dispose())

