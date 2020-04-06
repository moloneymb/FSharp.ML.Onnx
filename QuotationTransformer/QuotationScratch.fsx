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

#load "Common.fs"

open Common
open FSharp.Quotations.Evaluator
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.Patterns
open System.Reflection


type O(a: string) = 
    [<ReflectedDefinition>] static member A = "A"
    [<ReflectedDefinition>] static member AA with get() = "A" and set(x) = printf "%s" x
    [<ReflectedDefinition>] static member B () = "B"
    [<ReflectedDefinition>] static member BA () () = "BA"
    [<ReflectedDefinition>] static member C (c1:string) = c1 + "C"
    [<ReflectedDefinition>] static member D (d1:string, d2:int) = d1 + string d2 + "D"
    [<ReflectedDefinition>] static member E (e1:string) (e2:int) = e1 + string e2 + "E"
    [<ReflectedDefinition>] static member F( f1:string) (f2:int,f3:bool) = f1 + string f2 + string f3 + "F"
    [<ReflectedDefinition>] static member G (g1:string) = fun (g2:int) -> g1 + string g2 + "G"
    [<ReflectedDefinition>] static member H () (h1:string) (h2:string,h3:int) (h4:string,h5:int,h6:bool) = h1 + h2 + string h3 + h4 + string h5 + string h6 + "H"
    [<ReflectedDefinition>] member x.A0 = "A" + a
    [<ReflectedDefinition>] member x.B0 () = "B" + a
    [<ReflectedDefinition>] member x.C0 (c1:string) = c1 + "C" + a
    [<ReflectedDefinition>] member x.D0 (d1:string, d2:int) = d1 + string d2 + "D" + a
    [<ReflectedDefinition>] member x.E0 (e1:string) (e2:int) = e1 + string e2 + "E" + a
    [<ReflectedDefinition>] member x.F0 (f1:string) (f2:int,f3:bool) = f1 + string f2 + string f3 + "F" + a
    [<ReflectedDefinition>] member x.G0 (g1:string) = fun (g2:int) -> g1 + string g2 + "G" + a
    [<ReflectedDefinition>] member x.H0 () (h1:string) (h2:string,h3:int) (h4:string,h5:int,h6:bool) = h1 + h2 + string h3 + h4 + string h5 + string h6 + "H" + a
    [<ReflectedDefinition>] static member Combo() =
                                        O.A + O.B() + O.BA () () +
                                        O.C "c1" + O.D("d1",2) + O.E "e1" 2 + O.F "f1" (2,true) +
                                        O.G "g1" 2 + O.H () "h1" ("h2",3) ("h4",5,true)

    [<ReflectedDefinition>] member x.Combo0() = 
                                        x.A0 + x.B0 () + 
                                        x.C0 "c1" + x.D0("d1",2) + x.E0 "e1" 2 + 
                                        x.F0 "f1" (2,true) + x.G0 "g1" 2 + x.H0 () "h1" ("h2",3) ("h4",5,true) + 
                                        O.Combo()


let qt = <@ O.Combo() @>
let minQuote = 
    qt 
    |> Expr.unfoldWhileChanged ExprTransforms.expandWithReflectedDefinition |> Seq.last
    |> Expr.unfoldWhileChanged (ExprTransforms.reduceApplicationsAndLambdas true) |> Seq.last
    |> Expr.Cast<string>



//let rdH = Expr.TryGetReflectedDefinition(typeof<O>.GetMember("H").[0] :?> MethodBase) |> Option.get
//
//match <@ O.A @> with
//| PropertyGet(_,PropertyGetterWithReflectedDefinition (Lambdas(yss,_) as rd),zs) -> yss 
//| PropertySet(_,PropertySetterWithReflectedDefinition p,zs,_) -> failwith "todo"
//| _ -> failwith "todo"
//
//((<@ O("A").H0 () "foo" ("bar",1) ("fiz",2,true) @> |> Expr.unfoldWhileChanged expand |> Seq.toArray |> Seq.last) //|> Expr.getCallNames
//|> Expr.unfoldWhileChanged (ExprTransforms.reduceApplicationsAndLambdas true) |> Seq.last ).EvaluateUntyped()
//
//(<@ O("A").Combo0() @> |> Expr.unfoldWhileChanged expand |> Seq.toArray |> Seq.last)
//|> Expr.unfoldWhileChanged (ExprTransforms.reduceApplicationsAndLambdas true) |> Seq.last
//|> sprintExpr
//
//|> Seq.last
//|> ``|Applications|_|``
//|> Seq.last
//
//(O("A").H0 () "foo" ("bar",1) ("fiz",2,true))


// O("A").Combo0() 

//(<@ O("A").Combo0() @> |> Expr.unfoldWhileChanged expand |> Seq.toArray |> Seq.last).EvaluateUntyped()
//<@ O("A").Combo0() @>.EvaluateUntyped()
//<@ O("A").H0 () "foo" ("bar",1) ("fiz",2,true) @>.Evaluate()
//(<@ O("A").H0 @> |> Expr.unfoldWhileChanged expand |> Seq.toArray |> Seq.last).EvaluateUntyped()
//(<@ O("A").G0 "bar" 1 @> |> Expr.unfoldWhileChanged expand |> Seq.toArray |> Seq.last).EvaluateUntyped()

<@ O("A").G0 "bar" 1 @>.EvaluateUntyped()

(<@ O("A").F0 "bar" (1,false)  @> |> Expr.unfoldWhileChanged expand |> Seq.toArray |> Seq.last).EvaluateUntyped()
(<@ O("A").C0 "foo" @> |> Expr.unfoldWhileChanged expand |> Seq.toArray |> Seq.last).EvaluateUntyped()
(<@ O("A").D0("foo",1) @> |> Expr.unfoldWhileChanged expand |> Seq.toArray |> Seq.last).EvaluateUntyped()
(<@ O("A").E0 "bar" 1  @> |> Expr.unfoldWhileChanged expand |> Seq.toArray |> Seq.last).EvaluateUntyped()


<@ O.B @> |> Expr.tryFirstCall |> Option.get |> expand //|> Option.get |> sprintExpr
<@ O.E @> |> Expr.tryFirstCall |> Option.get |> expand //|> Option.get |> sprintExpr
<@ O.D @> |> Expr.tryFirstCall |> Option.get |> expand //|> Option.get |> sprintExpr
<@ O.H @> |> Expr.tryFirstCall |> Option.get |> expand //|> Option.get |> sprintExpr
<@ O.G @> |> Expr.tryFirstCall |> Option.get |> expand //|> Option.get |> sprintExpr
<@ O.Combo @> |> Expr.tryFirstCall |> Option.get |> expand //|> Option.get |> sprintExpr

<@ O.Combo @> |> Expr.unfoldWhileChanged expand
<@ O.B @> |> Expr.unfoldWhileChanged expand
<@ O.C @> |> Expr.unfoldWhileChanged expand
<@ O.D @> |> Expr.unfoldWhileChanged expand
<@ O.E @> |> Expr.unfoldWhileChanged expand
<@ O.F @> |> Expr.unfoldWhileChanged expand
<@ O.G @> |> Expr.unfoldWhileChanged expand

//<@ O.A @> |> Expr.unfoldWhileChanged expand |> Seq.last |> ExprTransforms.reduceApplicationsAndLambdas true


//(fun () -> "A") ()

//PropertyGet(instanceO,PropertyGetterWithReflectedDefinition (Lambdas(yss,_) as rd),zs) 



//((<@ O.Combo() @> |> Expr.unfoldWhileChanged expand |> Seq.last) |> Expr.unfoldWhileChanged (ExprTransforms.reduceApplicationsAndLambdas true) |> Seq.last).EvaluateUntyped()


//O.Combo()


(<@ O("A").B0() @> |> Expr.unfoldWhileChanged expand |> Seq.toArray |> Seq.last).EvaluateUntyped()
(<@ O("A").C0 "foo" @> |> Expr.unfoldWhileChanged expand |> Seq.toArray |> Seq.last).EvaluateUntyped()

//<@ O("A").B0() @>.EvaluateUntyped()

//<@ O("A").B0() @>.To

//<@ O.B @> |> Expr.tryFirstCall |> Option.get |> expand |> Option.get |> sprintExpr

([[1];[2]],([],1)) 

(([],1), [[1];[2]]) 
||> List.fold (fun (acc:int list list,c :int) (xs:int list)-> if xs.Length <= c then (xs::acc,c-xs.Length) else ((xs |> List.take c) :: acc,0))
|> fst |> List.skip 1 |> List.rev

match <@ O.Combo @> |> Expr.tryFirstCall |> Option.get with
| Call(_,MethodWithReflectedDefinition (Lambdas(yss,_) as rd),xs) -> (yss |> List.take 1,xs)
| _ -> failwith "err"


match <@ O.G @> |> Expr.tryFirstCall |> Option.get with
| Call(_,MethodWithReflectedDefinition (Lambdas(yss,_) as rd),xs) -> (yss,xs)
        let zs = [yield! instanceO |> Option.toList; yield! zs] 
        //let yss = (zs.Length, yss) ||> List.fold (fun c xs -> 
            /// Truncate based on zs as we can pick up too many lambdas by accident
        let reshape  (shape: int list) (xs: 'a list) : 'a list list =
            (([],xs),shape) ||> List.fold (fun (acc,xs) count -> List.chop (min count xs.Length) xs |> fun (head,xs) -> (head::acc,xs)) |> fst |> List.rev
        let zss = zs |> reshape [for ys in yss -> ys.Length] |> List.map (List.filter (fun z -> z.Type <> typeof<unit>))
        Expr.Applications(rd,zss)
| _ -> failwith "err"

//<@ O.Combo () @> |> Expr.tryFirstCall |> Option.get 
//<@ O.Combo () @> |> Expr.tryFirstCall //|> Option.get |> ExprTransforms.toCurried
//|> Expr.unfoldWhileChanged 
//|> Expr.unfoldWhileChanged (ExprTransforms.reduceApplicationsAndLambdas true) |> Seq.last |> sprintExpr

//((<@ O.H "h1" ("h2",1) ("h3",2,false) @> |> expandCalls |> Seq.last)).EvaluateUntyped()
//(<@ O.H "h1" ("h2",1) ("h3",2,false) @> |> expandCalls |> Seq.last).EvaluateUntyped()
// (<@ O.H () "h1" ("h2",1) ("h3",2,false) @> |> expandCalls |> Seq.last).EvaluateUntyped()
//(<@ O.H @> |> expandCalls |> Seq.last |> reduceApplications |> Seq.last).CompileUntyped()

// TODO turn Applications 

//(trans <@ O.A @>).EvaluateUntyped()
//(trans <@ O.B() @>).EvaluateUntyped()
//(trans <@ O.C("c1") @>).EvaluateUntyped()
//(trans <@ O.D("d1",2) @>).EvaluateUntyped()
//(trans <@ O.E "e1" 2 @>).EvaluateUntyped()
//(trans <@ O.F "e1" (1,true) @>).EvaluateUntyped()
//(trans <@ O.F "e1" (1,true) @>)
//(fixUp (trans <@ O.G "g1" 1 @>)).ToString()
//fixUp((fixUp (trans <@ O.G "g1" 1 @>)))
//(trans <@ O.G "g1" 1 @>).ToString()
//fixUp (trans <@ O.G "g1" 1 @>)

//<@ O.B("a","b") @>
//<@ O.C "a" "b" @>
//<@ O.D "a" ("b","c") @>

//let t = {x = 10; y = {x = 10; y = 10}}
//Expr.Call(typeof<O>.GetMember("B").[0] :?> MethodInfo,[Expr.Value("b1");Expr.Value("b2")]).EvaluateUntyped()
//((rdC,[Expr.Value("b1"); Expr.Value("b2")]) ||> List.fold (fun acc z -> Expr.Application(acc,z))).EvaluateUntyped()
//Expr.Application(Expr.Application(rdC, Expr.Value("b1")), Expr.Value("b2")).EvaluateUntyped()


//let x = ((test_data.[0] |> fst).RawData.ToByteArray() |> bytesToFloats).ToTensor().Reshape([|1;1;28;28|])
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

//match <@ t.X @> with
//| PropertyGet(_,x,_) -> x
//| _ -> failwith "todo"

