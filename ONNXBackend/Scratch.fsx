#I "C:/Users/moloneymb/.nuget/packages"

#r @"system.runtime.compilerservices.unsafe/4.5.2/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"
#r @"system.memory/4.5.3/lib/netstandard2.0/System.Memory.dll"
#r "microsoft.ml.onnxruntime/1.1.2/lib/netstandard1.1/Microsoft.ML.OnnxRuntime.dll"
#r @"google.protobuf/3.11.2/lib/netstandard2.0/Google.Protobuf.dll"
#r "onnx/csharp/OnnxMLProto.dll"
#load "ProtoBuf.fs"

open ProtoBuf
open System.Text
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors
open System.IO
open Onnx
open Google.Protobuf.Collections

let input1 = ArrayTensorExtensions.ToTensor(Array2D.create 1 32 2.f) :> Tensor<float32>
let input2 = ArrayTensorExtensions.ToTensor(Array2D.create 32 1 3.f) :> Tensor<float32>

let input1Int = ArrayTensorExtensions.ToTensor(Array2D.create 1 32 2L) :> Tensor<int64>
let input2Int = ArrayTensorExtensions.ToTensor(Array2D.create 32 1 3L) :> Tensor<int64>

let input4D1 = ArrayTensorExtensions.ToTensor(Array4D.create 3 3 1 3 2.f) :> Tensor<float32>
let input4D2 = ArrayTensorExtensions.ToTensor(Array4D.create 3 3 3 1 1.f) :> Tensor<float32>





