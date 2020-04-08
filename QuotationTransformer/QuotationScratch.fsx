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

// TODO Handle property set?

//type O() =
//    let mutable a = "x"
//    [<ReflectedDefinition>]
//    member this.A with get() = a and set(x) = a <- x
//
//match <@ O().A <- "b" @> with
//| PropertyGet(_,PropertyGetterWithReflectedDefinition (Lambdas(yss,_) as rd),zs) -> failwith "todo"
//| PropertySet(a,PropertySetterWithReflectedDefinition b,c,d) -> (a,b,c,d)
//| _ -> failwith "todo"




