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

#r "../protobuf/onnx/csharp/OnnxMLProto.dll"
#r "microsoft.ml.onnxruntime/1.1.2/lib/netstandard1.1/Microsoft.ML.OnnxRuntime.dll"

#load "../ONNXBackend/ProtoBuf.fs"
#load "../ONNXBackend/ONNXAPI.g.fs"
#load "Common.fs"

open Common
open ProtoBuf
open System.IO
open Microsoft.ML.OnnxRuntime.Tensors
open FSharp.Quotations.Evaluator
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open Onnx
open System


type on = ONNXAPI.ONNX

type ONNXAPI.ONNX with
    [<ReflectedDefinition>]
    static member reshape(x: Tensor<float32>,shape: int32[]) = on.reshape(x,(shape |> Array.map int64).ToTensor())


open System
let mnistDir = Path.Combine(__SOURCE_DIRECTORY__,"..","data","mnist")

type MNIST() = 

    let test_data = 
            let f(path: string) = 
                TensorProto.Parser.ParseFrom(File.ReadAllBytes(path))
            [| for i in [0;1;2] ->
                    Path.Combine(mnistDir,sprintf "test_data_set_0") 
                    |> fun dir -> (f(Path.Combine(dir,"input_0.pb")),f(Path.Combine(dir,"output_0.pb")))|]

    let bytesToFloats(buffer : byte[]) = 
        let xs= Array.zeroCreate<float32> (buffer.Length / 4)
        System.Buffer.BlockCopy(buffer, 0, xs, 0, buffer.Length)
        xs

    let getTensorF(name,shape) =
        let dts = File.ReadAllBytes(Path.Combine(mnistDir, name)) |> bytesToFloats
        on.reshape(ArrayTensorExtensions.ToTensor(dts) ,ArrayTensorExtensions.ToTensor(shape))

    let p193 = getTensorF("Parameter193", [|16L; 4L; 4L; 10L|])
    let p87  = getTensorF("Parameter87",  [|16L; 8L; 5L; 5L|])
    let p5   = getTensorF("Parameter5",  [|8L; 1L; 5L; 5L|])
    let p6   = getTensorF("Parameter6", [|8L; 1L; 1L|])
    let p88  = getTensorF("Parameter88", [|16L; 1L; 1L|])
    let p194 = getTensorF("Parameter194", [|1L; 10L|]) 

    [<ReflectedDefinition>]
    member this.Rec(x:Tensor<float32>,p1,p2,k) = 
       on.max_pool(on.relu(on.add(on.conv(x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst

    [<ReflectedDefinition>]
    member this.Forward(x: Tensor<float32>) = 
        on.add(on.mat_mul(on.reshape((this.Rec (this.Rec(x,p5,p6,2L),p87,p88,3L)),[|1;256|]),on.reshape(p193,[|256;10|])),p194)



let onMethods = typeof<on>.GetMethods() |> Array.map (fun x -> x.Name) |> Set

//let rMethods = 
//    typeof<ONNXGraph>.GetMethods() 
//    |> Array.filter (fun x -> x.Name <> "Constant") 
//    |> Array.map (fun x -> (x.Name,x)) |> Map
//
//let cMethods = 
//    typeof<ONNXGraph>.GetMethods() 
//    |> Array.filter (fun x -> x.Name = "Constant") 
//    |> Array.map (fun x -> x.GetParameters().[1].ParameterType.GenericTypeArguments.[0].FullName,x) 
//    |> Map

//
//
//let mnistG = MNISTGraph()

//    static member getExprGraph<'a> (f:Expr<Tensor<'a> -> Tensor<'a>>) : ((Tensor<'a> -> Tensor<'a>) * IDisposable) =
//        let g = Graph.Default()
//        let input = {name = g.GetName("Input"); dt = getDataType(typeof<'a>)}
//        let res = (trans g (Map([input.name,input])) f).EvaluateUntyped() :?> ValueInfo
//        let gp = GraphProto(Name = "G")
//        gp.Input.Add(makeValueInfoProto(input))
//        gp.Output.Add(makeValueInfoProto(res))
//        gp.Node.Add(g.ops)
//        let sess = new InferenceSession(writeModelToStream(gp |> graphToModel))
//        let f x = 
//            use res2 = sess.Run([])
//            (res2 |> Seq.head).AsTensor<'a>().Clone()
//        (f , {new IDisposable with member this.Dispose() = sess.Dispose()})
//sess.Run()
//        let res = (expandCalls f).EvaluateUntyped() :?> ValueInfo
//        let f () = 
//            use res2 = sess.Run([])
//            (res2 |> Seq.head).AsTensor<'a>().Clone()
//        (f , {new IDisposable with member this.Dispose() = sess.Dispose()})
