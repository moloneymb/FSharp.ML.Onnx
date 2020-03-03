// Ported from https://github.com/microsoft/onnxruntime/blob/master/csharp/sample/Microsoft.ML.OnnxRuntime.InferenceSample/Program.cs 

open System
open System.Collections.Generic
open System.Text
open System.IO
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors

let loadTensorFromFile(filename: string) = 
    File.ReadAllLines(filename).[1..]
    |> Array.collect (fun line -> line.Split([|',';'[';']'|], StringSplitOptions.RemoveEmptyEntries))
    |> Array.map Single.Parse

let UseAPI() = 
    let dir = Path.Combine(__SOURCE_DIRECTORY__ ,"..", "data", "squeezenet")
    let modelPath = Path.Combine(dir,"squeezenet.onnx")

    // Optional : Create session options and set the graph optimization level for the session
    let options = new SessionOptions()
    options.GraphOptimizationLevel <- GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    use session = new InferenceSession(modelPath, options)
    let inputMeta = session.InputMetadata
    let inputData = loadTensorFromFile(Path.Combine(dir,"bench.in"))
    let container = 
        [|
            for name in inputMeta.Keys do
                let tensor = new DenseTensor<float32>(Memory.op_Implicit(inputData),ReadOnlySpan.op_Implicit(inputMeta.[name].Dimensions)) 
                yield NamedOnnxValue.CreateFromTensor<float32>(name, tensor)
        |]
    use results = session.Run(container)
    for r in results do
        printfn "Output for %s" r.Name
        printfn "%s" (r.AsTensor<float32>().GetArrayString())


//[<EntryPoint>]
//let main argv =
//    printfn "Using API"
//    UseAPI()
//    printfn "Done"
//    0 


[<EntryPoint>]
let main argv =
    let mnistDir = Path.Combine(__SOURCE_DIRECTORY__, "..","data","mnist")
    let model = File.ReadAllBytes(Path.Combine(mnistDir, "model.onnx"))
    let x = Onnx.ModelProto.Parser.ParseFrom(model)
    0 



