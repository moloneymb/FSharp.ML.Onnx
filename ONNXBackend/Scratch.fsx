#I "C:/Users/moloneymb/.nuget/packages"
#r @"system.runtime.compilerservices.unsafe/4.5.2/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"
#r @"system.memory/4.5.3/lib/netstandard2.0/System.Memory.dll"
#r "microsoft.ml.onnxruntime/1.1.2/lib/netstandard1.1/Microsoft.ML.OnnxRuntime.dll"
#r @"google.protobuf/3.11.2/lib/netstandard2.0/Google.Protobuf.dll"
#r "../protobuf/onnx/csharp/OnnxMLProto.dll"
#load "ProtoBuf.fs"
#load "ONNXAPI.g.fs"

open ProtoBuf
open System.Text
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors
open System.IO
open Google.Protobuf.Collections
open ONNXAPI


#time

type on = ONNX
type Tensor<'a> with
    member this.shape = this.Dimensions.ToArray()

let input1 = ArrayTensorExtensions.ToTensor(Array2D.create 10000 40 -2.f) :> Tensor<float32>
let input2 = ArrayTensorExtensions.ToTensor(Array2D.create 40 10000 -2.f) :> Tensor<float32>

let res1 = on.MatMul(input2,on.Abs(input1))
res1.shape
res1.[0,0] = -40000.f

#time


//res.shape

//3.24 -> 2.0 -> 0.5
//0.00145

//0.5/0.0015

//let res = on.MatMul(on.Relu(input2),input1)

//let input2 = ArrayTensorExtensions.ToTensor(Array2D.create 32 1 3.f) :> Tensor<float32>

//let input1Int = ArrayTensorExtensions.ToTensor(Array2D.create 1 32 2L) :> Tensor<int64>
//let input2Int = ArrayTensorExtensions.ToTensor(Array2D.create 32 1 3L) :> Tensor<int64>
//
//let input4D1 = ArrayTensorExtensions.ToTensor(Array4D.create 3 3 1 3 2.f) :> Tensor<float32>
//let input4D2 = ArrayTensorExtensions.ToTensor(Array4D.create 3 3 3 1 1.f) :> Tensor<float32>
//
