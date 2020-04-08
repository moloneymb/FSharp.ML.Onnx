module Common

open System
open System.Reflection
open Fantomas
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape
open Microsoft.FSharp.Quotations.Patterns
open System.Text.RegularExpressions

let (|TryFunc|_|) (f: 'a -> 'b option ) (x:'a) = f x

let private regexs = [| Regex(@"Operators\.string\<[_A-Za-z0-9]*\>"), "string"; Regex(@"\(unitVar[0-9]* : Unit\)"), "()" |]


let sprintExpr (expr:Expr) = 
    // TODO figure out ommit Enclosing Types
    Falanx.Machinery.Quotations.ToAst(expr,knownNamespaces = set ["Operators"])
    |> snd
    |> fun x -> CodeFormatter.FormatAST(x, "helloWorld.fs",  None, FormatConfig.FormatConfig.Default) 
    |> Falanx.Machinery.Expr.cleanUpTypeName
    |> fun x -> 
        let xs = x.Substring(31).Replace("\r\n","\n").Split([|'\n';'\r'|]) 
        if xs.[0].Contains(";") then xs.[0].Substring(33) 
        else 
            (xs.[2..] |> Array.choose (fun x -> if x.Length > 3 then Some(x.Substring(4)) else None) |> String.concat "\n")
        |> fun x -> x.Trim(' ','\n','\r')
        |> fun x -> (x,regexs) ||> Array.fold (fun (x:string) ((r,s):Regex*string) -> r.Replace(x,s))


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

let (|Found|_|) map key = map |> Map.tryFind key

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

    let map (f: Expr -> Expr) (expr: Expr) = visitor (fun _ e -> ((),f(e))) () expr |> snd

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
                match Expr.count (function | ShapeVar v2 -> v = v2 | _ -> false) body with
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
module ErrorSimplifier = 
    open FParsec

    let identifier =
        let isIdentifierFirstChar c = isLetter c || c = '_'
        let isIdentifierChar c = isLetter c || isDigit c || c = '_'
        many1Satisfy2L isIdentifierFirstChar isIdentifierChar "identifier" .>> spaces

    let identifierWithNamespaces : Parser<string list, unit> = 
        sepBy1 identifier (pchar '.' <|> pchar '+')

    type TypeAST = 
        | Base of string list
        | Generic of string list * TypeAST list

    let pvalue, pvalueRef = createParserForwardedToRef<TypeAST, unit>()

    let pbase = identifierWithNamespaces |>> TypeAST.Base

    let pgeneric : Parser<TypeAST,unit> = 
        identifierWithNamespaces 
        .>>. (pchar '`'
        >>. pint32
        >>. pchar '['
        >>. (sepBy1 pvalue (pchar ','))
        .>> pchar ']')
        |>> fun (x,y) -> TypeAST.Generic(x,y)

    do pvalueRef :=  (attempt pgeneric) <|> pbase 

    let rec printTypeAst (ast: TypeAST) =
        match ast with
        | TypeAST.Base(names) ->
            match names with 
            | ["System";"String"] -> "string"
            | ["System";"Int32"] -> "int"
            | ["System";"Boolean"] -> "bool"
            | ["Microsoft";"FSharp"; "Core"; "Unit"] -> "()"
            | _ -> names |> String.concat "\."
        | TypeAST.Generic(names,xs) -> 
            match names with
            | ["FSharp"; "Quotations"; "Evaluator"; "Tools"; "FuncFSharp"] 
            | ["Microsoft"; "FSharp"; "Core"; "FSharpFunc"] -> 
                xs |> List.map printTypeAst |> String.concat " -> "
            | ["System"; "Tuple"] -> 
                "(" + (xs |> List.map printTypeAst |> String.concat "*") + ")"
            | _ -> (names |> String.concat ".") + "`" + (string xs.Length) + "[" + ((xs |> List.map printTypeAst) |> String.concat ",") + "]"

    let processMsg (msg:string) = 
        let printErrorMsgAST (xs: Choice<string,TypeAST>[]) = xs |> Array.map (function | Choice1Of2 x -> x | Choice2Of2 x -> sprintf "(%s)" <| printTypeAst x) |> String.concat ""
        let tpe = between (pchar '\'') (pchar '\'') pvalue 
        let txt = Choice1Of2
        let typ = Choice2Of2
        let errMsg1 = " cannot be used for parameter of type " |> fun t1 -> ((tpe .>> pstring t1 .>>. tpe ) |>> fun (a,b) -> [|typ a; txt t1; typ b|])
        match run (choice [errMsg1]) msg with
        | Success(xs,_,_)   -> printErrorMsgAST(xs)
        | Failure(errorMsg, _, _) -> failwithf "Failure: %s" errorMsg
