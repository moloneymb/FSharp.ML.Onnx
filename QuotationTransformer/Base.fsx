#I @"C:\EE\Git\ONNXBackend\ONNXBackend\bin\Debug\net47\"
#r "FSharp.Quotations.Evaluator.dll"
#r "Google.Protobuf.dll"
#r "OnnxMLProto.dll"
#r "Microsoft.ML.OnnxRuntime.dll"
#r "ONNXBackend.exe"

open Common
open Microsoft.ML.OnnxRuntime.Tensors
open ProtoBuf
open System.IO

let mnistDir = Path.Combine(__SOURCE_DIRECTORY__,"..","data","mnist")

type MNIST() = 

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

