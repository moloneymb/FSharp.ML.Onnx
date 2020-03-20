#I "C:/Users/moloneymb/.nuget/packages"
#r @"system.runtime.compilerservices.unsafe/4.5.2/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"
#r @"system.memory/4.5.3/lib/netstandard2.0/System.Memory.dll"
#r "microsoft.ml.onnxruntime/1.1.2/lib/netstandard1.1/Microsoft.ML.OnnxRuntime.dll"
#r @"google.protobuf/3.11.2/lib/netstandard2.0/Google.Protobuf.dll"
#r "../protobuf/onnx/csharp/OnnxMLProto.dll"
#r "netstandard"

#load "ProtoBuf.fs"
#load "ONNXAPI.g.fs"

#time

open System.IO
open Onnx
open ProtoBuf
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors

type on = ONNXAPI.ONNX

//let mnistDir = Path.Combine(__SOURCE_DIRECTORY__,"..","data","mnist")

