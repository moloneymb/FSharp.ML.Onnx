#I @"C:\Users\moloneymb\.nuget\packages\"
#r @"fsharp.compiler.service\25.0.1\lib\net45\FSharp.Compiler.Service.dll"
#r @"fantomas\2.9.2\lib\net452\Fantomas.dll"
#r @"falanx.machinery\0.5.2\lib\netstandard2.0\Falanx.Machinery.dll"
#r @"fsharp.quotations.evaluator\2.1.0\lib\netstandard2.0\FSharp.Quotations.Evaluator.dll"
#r @"system.runtime.compilerservices.unsafe/4.5.2/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"
#r @"system.memory/4.5.3/lib/netstandard2.0/System.Memory.dll"
#r @"fparsec/1.1.1/lib/net45/FParsecCS.dll"
#r @"fparsec/1.1.1/lib/net45/FParsec.dll"

// https://github.com/fable-compiler/Fable/blob/75a5f78bdc162d6785bef3e1d7012679d7a54f84/src/Fable.Transforms/Replacements.fs
// https://github.com/ZachBray/FunScript/blob/941d28b40752cdc81a22f7e42ae21e9844c86fdb/src/main/FunScript/ExpressionReplacer.fs

#load "Common.fs"

open Common
open FSharp.Quotations.Evaluator
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.Patterns
open System
open System.Reflection
open System.IO

// TODO replace built in operators with 
// NOTE: This is not needed as we can execute built-ins during graph construction

// NOTES On generic type substitution on Expressions
// Functions that use generics have a type of the name of the generic, these are knowable ahead of time
// In theory it should be possible to substitute the variable assignments with variables of the replaced types
// It would have to thread through structual typeing, and perhaps even NewObject
// It could go a long way
[<ReflectedDefinition>]
module BuiltIns = 
    let inline fst (a, _) = a
    let inline snd (_, b) = b
    let inline ignore _ = ()
    let ref value = { contents = value }
    let (:=) cell value = cell.contents <- value
    let (!) cell = cell.contents
    let (|>) arg func = func arg
    let (||>) (arg1, arg2) func = func arg1 arg2
    let (|||>) (arg1, arg2, arg3) func = func arg1 arg2 arg3
    let (<|) func arg1 = func arg1
    let (<||) func (arg1, arg2) = func arg1 arg2
    let (<|||) func (arg1, arg2, arg3) = func arg1 arg2 arg3
    let (>>) func1 func2 x = func2 (func1 x)
    let (<<) func2 func1 x = func2 (func1 x)
    let id x = x
    type internal Marker = interface end
    let t = typeof<Marker>.DeclaringType

let builtInMap = 
    BuiltIns.t.GetMethods() 
    |> Array.filter (fun x -> match x.Name with | "GetType" | "get_t" | "ToString" | "GetHashCode" | "Equals" -> false | _ -> true)
    |> Array.choose (fun x -> Expr.TryGetReflectedDefinition (x) |> Option.map (fun y -> x.Name, y)) |> Map.ofArray



//let (|BuiltIn|_|) (table: Map<string,Expr>) (mi: MethodInfo)  = table.TryFind(mi.Name)

//builtInMap |> Map.toArray |> Array.map fst

//let t = 
//    match builtInMap.TryFind("op_PipeRight").Value with
//    | Lambda(a,Lambda(b,Application (c, d))) -> a.Type //(a.Type.Name,b.Type.Name, c.Type.Name, d.Type.Name)
//    | _ -> failwith "err"

//Expr.Cast(builtInMap.TryFind("op_PipeRight")

//(builtInMap.TryFind("op_PipeRight").Value |> Expr.getVarsUsage |> Array.item 1).Type
//(builtInMap.TryFind("op_PipeRight").Value |> Expr.getVarsUsage |> Array.item 0).Type

//match <@@ (|>) @@> with
//| Lambda(a, Lambda(b, _)) -> (a.Type.Name, b.Type.Name)
//| _ -> failwith "err"
//
//match <@@ (|>) @@> with
//| Lambda(a,Lambda(b,Application (c, d))) -> a.Type.Name
//| _ -> failwith "err"
//let f (mi:MethodInfo) =  builtInMap.TryFind(mi.Name)


module Option =
    let toNone _ = None

//<@ "a" |> fun b -> b + "c" @> |> Expr.unfoldWhileChanged (ExprTransforms.expand (fun mi -> builtInMap.TryFind(mi.Name)) Option.toNone)

//open BuiltIns

let expand (g: MethodInfo -> Expr option) (expr: Expr) = 
    let f ((instanceO : Expr option),(yss:Var list list),rd, zs) =
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
    match expr with
    | Call(None, TryFunc g (Lambdas (yss,_) as rd), zs) -> 
        f (None, yss, rd, zs)
    | PropertyGet(instanceO,PropertyGetterWithReflectedDefinition (Lambdas(yss,_) as rd),zs) 
    | Patterns.Call(instanceO, MethodWithReflectedDefinition (Lambdas(yss,_) as rd), zs) ->
        f (instanceO, yss, rd, zs)
    //| Patterns.Call(instanceO, (TryFunc f (Lambdas(yss,_) as rd)), zs) ->
    | _ -> None

//<@ "a" |> fun b -> b + "c" @> |> Expr.unfoldWhileChanged (expand f) |> Seq.last


//match <@@ (|>) @@> |> Expr.tryFirstCall |> Option.get with
//| Call(None, TryFunc f (Lambdas (yss,body)), args) -> yss
//| _ -> failwith "todo"


