namespace Test
open NUnit.Framework

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


module ExpressionFunctions =                                         

    [<Test>]
    let ``static method quotation and simplification``() = 
        let qt = <@ O.Combo() @>
        let minQuote = 
            qt 
            |> Expr.unfoldWhileChanged ExprTransforms.expandWithReflectedDefinition |> Seq.last
            |> Expr.unfoldWhileChanged (ExprTransforms.reduceApplicationsAndLambdas true) |> Seq.last
            |> Expr.Cast<string>
        Assert.AreEqual(minQuote.Evaluate(), qt.Evaluate(), "Static Method Quotation expanded and reduced")
        Assert.AreEqual(minQuote |> Expr.getCallNames, set ["ToString"; "op_Addition"], "Expanded expression should only have ToString and op_Addition calls")

    [<Test>]
    let ``object method quotation and simplification``() = 
        let qt = <@ O("A").Combo0() @>
        let minQuote = 
            qt 
            |> Expr.unfoldWhileChanged ExprTransforms.expandWithReflectedDefinition |> Seq.last
            |> Expr.unfoldWhileChanged (ExprTransforms.reduceApplicationsAndLambdas true) |> Seq.last
            |> Expr.Cast<string>
        Assert.AreEqual(minQuote.Evaluate(), qt.Evaluate(), "Objct Method Quotation expanded and reduced")
        Assert.AreEqual(minQuote |> Expr.getCallNames, set ["ToString"; "op_Addition"], "Expanded expression should only have ToString and op_Addition calls")

