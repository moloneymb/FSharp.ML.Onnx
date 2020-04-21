(* We can support multi-out *)
#load "Base.fsx"
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

type Bar2 = {x:Tensor<int>;y:(Tensor<int>*Tensor<int>)}


let mm = getMM typeof<Bar2>

let inputs = ExprRun.getValueInfo(0,mm) |> snd

<@ [|0|].[1..2] @>

<@ fun (input:Bar2) -> (input.x , (fst input.y) , (snd input.y)) @>
|> Expr.applyTransform ExprTransforms.Simple.builtIns
|> function
| Lambda(x,_) -> x
| _ -> failwith "err"



<@
let (x,y,z) = (10,20,30)
    in (x,y,z)
@>

//Microsoft.FSharp.Reflection.FSharpValue.MakeTuple

<@ fun (input:Bar2) -> 
    let (a,b) = input.y
    input.x + a + b @>

//flattenStructualReturnType<Tensor<int32>,Bar2> <@ {x = p2; y = (p2,p2)} @>

//Expr.Cast<Tensor<int>[]>(Expr.NewArray(typeof<float32>,trans <@ {x = 1; y = (2,3)} @> )).Evaluate()

//Expr.Cast<float32[]>(Expr.NewArray(typeof<float32>,trans <@ (2,3) @> )).Evaluate()
//Expr.Cast<float32[]>(Expr.NewArray(typeof<float32>,trans <@ 2 @> )).Evaluate()


//NewRecord

