#r "nuget: Google.Protobuf"
#r "nuget: Microsoft.ML.OnnxRuntime"
#r "nuget: FSharp.Quotations.Evaluator"
#r "nuget: System.Drawing.Common"
#r "nuget: TextCopy, 6.2.1"
#r @"C:\EE\Git\FSharp.ML.Onnx\OnnxMlProto.dll"
#r @"C:\EE\Git\FSharp.ML.Onnx\FSharpMLOnnx\bin\Debug\netstandard2.0\FSharpMLOnnx.dll"
#r @"C:\EE\Git\fcsv7\FCSCore\bin\Debug\net8.0\FCSCore.dll"


open Utilities
open System
open System.IO
open Onnx
open FSharp.ML.Onnx.Protobuf
open Microsoft.ML.OnnxRuntime.Tensors
open Microsoft.ML.OnnxRuntime

open Utilities
open System
open FSharp.ML.Onnx.Protobuf
open Microsoft.ML.OnnxRuntime.Tensors
open Microsoft.ML.OnnxRuntime
open Microsoft.FSharp.Quotations
open FSharp.ML.Onnx.Utils
open FSharp.ML.Onnx.Utils.Expr
open FSharp.ML.Onnx.Expr

type on = FSharp.ML.Onnx.API.SnakeCase.Onnx
type DV<'a> = DisposableValue<'a>

let zeroTensor = ArrayTensorExtensions.ToTensor([|0.f|])
let piTensor = ArrayTensorExtensions.ToTensor([|float32 Math.PI|])

let xs = ArrayTensorExtensions.ToTensor([|Single.NegativeInfinity;-1.f;0.f;-0.0001f |])
let ys = ArrayTensorExtensions.ToTensor([|Single.NegativeInfinity;-1.f;0.f;1.f |])

type StubClass() =
    [<FSharp.ML.Onnx.Expr.MethodSubstitution("Atan2")>]
    [<ReflectedDefinition>]
    static member Atan2(graph: Graph, x: ValueInfo, y: ValueInfo) =
        graph.AddNode("Atan2", [|x;y|], [|x.dt|], [||],domain="v1") |> toTuple1

type FSharp.ML.Onnx.API.PascalCase.Onnx with
    // Stub for substitution
    [<FSharp.ML.Onnx.Expr.MethodSubstitution("Atan2")>]
    static member Atan2(x: Tensor<float32>,y: Tensor<float32>) : Tensor<float32> = 
        // Is this needed?
        MV() |> fun mv -> execNode<float32> "Atan2" [|mv.c(x); mv.c(y)|] [||]

type FSharp.ML.Onnx.API.SnakeCase.Onnx with
    [<ReflectedDefinition>]
    static member atan2(x: Tensor<float32>,y: Tensor<float32>) =
      FSharp.ML.Onnx.API.PascalCase.Onnx.Atan2(x,y)

type MappedFunctions = Map<string,(Var list * Expr)>

let extensionFunctions = 
  typeof<StubClass>.GetMethods()
  |> Array.filter filterMethods
  |> Array.choose (fun mi -> 
    match mi with
    | FSharp.ML.Onnx.Expr.TryGetMethodSubstitution(name) ->
      match Expr.TryGetReflectedDefinition(mi) with
      | None -> None
      | Some(DerivedPatterns.Lambdas([xs],t)) -> Some(name,(xs,t))
      | _ -> failwith "err"
    | _ -> None
    )
  |> Map.ofArray

let toOnnxModel<'a,'b>(expr: Expr<'a -> 'b>, mappedFunctions:MappedFunctions)  : ValueInfo[]*ValueInfo[]*Onnx.ModelProto= 
    let mmIn = getMM (typeof<'a>)
    let mmOut = getMM (typeof<'b>)
    OnnxProcessor.BuildGraph(expr,mmIn,mmOut,mappedFunctions)

let toOnnxFunction(f:Expr<'a->'b>,opType:string, domain:string, mappedFunctions:MappedFunctions) =
  let ins,outs,model = toOnnxModel(f,mappedFunctions)
  let g = model.Graph
  let funcProto = Onnx.FunctionProto(Name=opType)
  funcProto.Input.AddRange(g.Input |> Seq.map (fun x -> x.Name))
  funcProto.Node.AddRange(g.Node)
  funcProto.Output.AddRange(g.Output |> Seq.map (fun x -> x.Name))
  funcProto.Domain <- domain
  funcProto.OpsetImport.Add(Onnx.OperatorSetIdProto(Domain="",Version=14L))
  (ins,outs,funcProto)

let funExpr =
  <@ fun (self:Tensor<float32>,other:Tensor<float32>) -> 
      let atan = on.atan(on.div(self,other))
      let cond = on.less(other,zeroTensor)
      let piFactor = on.where(cond, on.mul(on.sign(self),piTensor),zeroTensor)
      on.add(atan,piFactor)
   @>

let _,_,funcProto = toOnnxFunction(funExpr,"Atan2","v1",Map.empty)
let _,_,model = toOnnxModel(<@ fun (x:Tensor<float32>,y:Tensor<float32>) -> on.atan2(x,y) @>,extensionFunctions)

model.Functions.Add(funcProto)
model.OpsetImport.Add(Onnx.OperatorSetIdProto(Domain="v1",Version=14L))


let passed = 
  use sess = new InferenceSession(model.ToArray(),new SessionOptions(LogVerbosityLevel = 3))
  use res = sess.Run([|NamedOnnxValue.CreateFromTensor("Input0",xs);NamedOnnxValue.CreateFromTensor("Input1",ys)|])
  // The nanfs are not great but that's beside the point
  sprintf "%A" (res.[0].AsTensor<float32>().ToDenseTensor().Buffer.ToArray()) = "[|nanf; -2.356194496f; nanf; 1.570896387f|]"

let mnistDir = Path.Combine(__SOURCE_DIRECTORY__,"..","data","mnist")

type ong = FSharp.ML.Onnx.API.Graph.OnnxGraph

open FSharp.ML.Onnx.Extensions
type MNISTGraph() = 

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
    member this.Rec(graph:Graph, x:ValueInfo,p1,p2,k) = 
       ong.MaxPool(graph,ong.Relu(graph,ong.Add(graph,ong.Conv(graph,x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst

    [<ReflectedDefinition>]
    member this.Forward(graph: Graph, x: ValueInfo) = 
        let constant (x:Tensor<float32>) = Constants.constant(graph,x)
        let x = this.Rec(graph, x, constant p5,constant p6,2L)
        let x = this.Rec (graph, x, constant p87,constant p88,3L)
        ong.Add(graph, ong.MatMul(graph, ong.Reshape(graph, x,Constants.constant(graph,[|1L;256L|].ToTensor())),ong.Reshape(graph,constant p193,Constants.constant(graph,[|256L;10L|].ToTensor()))),constant p194)

    [<ReflectedDefinition>]
    member this.Rec(x:Tensor<float32>,p1,p2,k) = 
       on.max_pool(on.relu(on.add(on.conv(x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst

    [<ReflectedDefinition>]
    member this.Forward(x: Tensor<float32>) : Tensor<float32> = 
        let layer (p1,p2,k) (x:Tensor<float32>) : Tensor<float32> = 
            on.max_pool(on.relu(on.add(on.conv(x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst
        on.add(on.mat_mul(on.reshape(x |> layer(p5,p6,2L) |> layer (p87, p88, 3L),[|1;256|]),on.reshape(p193,[|256;10|])),p194)


let mnistG = MNISTGraph()
//let graphFunction : DV<Tensor<float32> -> DV<Tensor<float32>>> =  OnnxProcessor.ToOnnxGraph(<@ mnistG.Forward @>)
let graphFunction =  <@ fun (x:Tensor<float32>) -> mnistG.Forward(x) @>
let mmIn = getMM (typeof<Tensor<float32>>)
let mmOut = getMM (typeof<Tensor<float32>>)
let inputs,outputs,model2 = OnnxProcessor.BuildGraph(graphFunction,mmIn,mmOut)

