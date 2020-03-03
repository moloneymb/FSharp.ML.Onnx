// Ported from https://github.com/microsoft/onnxruntime/blob/master/csharp/sample/Microsoft.ML.OnnxRuntime.InferenceSample/Program.cs 

open System
open System.Collections.Generic
open System.Text
open System.IO
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors


[<EntryPoint>]
let main argv =
    let mnistDir = Path.Combine(__SOURCE_DIRECTORY__, "..","data","mnist")
    let model = File.ReadAllBytes(Path.Combine(mnistDir, "model.onnx"))
    let x = Onnx.ModelProto.Parser.ParseFrom(model)
    0 



