// TODO recursively get reflected definition
// TODO code generate helper functions for graph construction


#I @"C:\Users\moloneymb\.nuget\packages\"
#r @"fsharp.compiler.service\25.0.1\lib\net45\FSharp.Compiler.Service.dll"
#r @"fantomas\2.9.2\lib\net452\Fantomas.dll"
#r @"falanx.machinery\0.5.2\lib\netstandard2.0\Falanx.Machinery.dll"
#r @"C:\Users\moloneymb\source\repos\QuotationTesting\packages\FSharp.Quotations.Evaluator.1.1.3\lib\net45\FSharp.Quotations.Evaluator.dll"
#I "C:/Users/moloneymb/.nuget/packages"
#r @"system.runtime.compilerservices.unsafe/4.5.2/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"
#r @"system.memory/4.5.3/lib/netstandard2.0/System.Memory.dll"
#r "microsoft.ml.onnxruntime/1.1.2/lib/netstandard1.1/Microsoft.ML.OnnxRuntime.dll"
#r @"google.protobuf/3.11.2/lib/netstandard2.0/Google.Protobuf.dll"
#r "../protobuf/onnx/csharp/OnnxMLProto.dll"
#r "netstandard"

#load "../ONNXBackend/ProtoBuf.fs"
#load "../ONNXBackend/ONNXAPI.g.fs"

open ProtoBuf
open System.IO
open Microsoft.ML.OnnxRuntime.Tensors
open Microsoft.ML.OnnxRuntime
open FSharp.Quotations.Evaluator
open Fantomas
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape
open Microsoft.FSharp.Quotations.Patterns
open Onnx
open System
open System.Reflection

type Graph = 
    { 
        mutable ops : NodeProto list
        mutable usedNames : Map<string,int>
    } 
    static member Default() = {ops = []; usedNames = Map.empty}
    member this.GetName(name : string) : string = 
            let (x,y) = 
                match this.usedNames.TryFind(name) with
                | None -> name,this.usedNames.Add(name,0)
                | Some(v) -> 
                    let newName = name + string(v + 1)
                    newName,this.usedNames.Add(name,v+1).Add(newName,0)
            this.usedNames <- y
            x

    member this.AddNode(node: NodeProto) = this.ops <- node::this.ops


type R() = 
//    static member Add(name: string, A: string, B: string, output: string) : NodeProto =
//        Node.binaryOp "Add" [||] (name,A,B,output)

    static member Constant(graph: Graph, t: Tensor<float32>) : string =
        let output = (graph.GetName("Constant_Output"))
        graph.AddNode(Node.simple "Constant" (graph.GetName("Constant"),[||],[|output|],[|Attr.tensor(t).Value|]))
        output

    static member Relu(graph : Graph, A: string) : string =
        let output = (graph.GetName("Relu_Output"))
        graph.AddNode(Node.unaryOp "Relu" [||] (graph.GetName("Relu"),A,output))
        output

    static member Add(graph : Graph, A: string, B: string) : string =
        let output = (graph.GetName("Add_Output"))
        graph.AddNode(Node.binaryOp "Add" [||] (graph.GetName("Add"),A,B,output))
        output

type on = ONNXAPI.ONNX
let g = Graph.Default()


let t1 = ArrayTensorExtensions.ToTensor(Array.init 100 (fun i -> float32(i % 20)))
let t2 = ArrayTensorExtensions.ToTensor(Array.init 100 (fun i -> float32((i % 20)-10)))
let f1 = <@@ fun (x:Tensor<float32>,y:Tensor<float32>) -> on.Relu(on.Add(x,y)) @@>

let f2 = f1.EvaluateUntyped() :?> (Tensor<float32>*Tensor<float32>->Tensor<float32>)


let rec trans (exp : Expr) =
    failwith "todo"

    //match exp with
    //| Call

let mnistDir = Path.Combine(__SOURCE_DIRECTORY__,"..","data","mnist")

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
    on.Reshape(ArrayTensorExtensions.ToTensor(dts) ,ArrayTensorExtensions.ToTensor(shape))

type 'a``[]`` with
    member x.ToTensor() = ArrayTensorExtensions.ToTensor(x)

type Tensor<'a> with
    member x.Reshape(shape:int[]) = 
        x.Reshape(System.ReadOnlyMemory.op_Implicit(shape).Span)

type ONNXAPI.ONNX with
    static member Reshape(x: Tensor<float32>,shape: int32[]) = on.Reshape(x,(shape |> Array.map int64).ToTensor())

type MNIST() = 
    let p193 = getTensorF("Parameter193", [|16L; 4L; 4L; 10L|])
    let p87  = getTensorF("Parameter87",  [|16L; 8L; 5L; 5L|])
    let p5   = getTensorF("Parameter5",  [|8L; 1L; 5L; 5L|])
    let p6   = getTensorF("Parameter6", [|8L; 1L; 1L|])
    let p88  = getTensorF("Parameter88", [|16L; 1L; 1L|])
    let p194 = getTensorF("Parameter194", [|1L; 10L|]) 
    [<ReflectedDefinition>]
    member this.Forward(x: Tensor<float32>) = 
        let f x p1 p2 k = on.MaxPool(on.Relu(on.Add(on.Conv(x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst
        on.Add(on.MatMul(on.Reshape((f (f x p5 p6 2L) p87 p88 3L),[|1;256|]),on.Reshape(p193,[|256;10|])),p194) 

let rd = Expr.TryGetReflectedDefinition(typeof<MNIST>.GetMember("Forward").[0] :?> MethodBase).Value

//rd

//let g = rd.EvaluateUntyped() :?> (MNIST -> Tensor<float32> -> Tensor<float32>)


let x = ((test_data.[0] |> fst).RawData.ToByteArray() |> bytesToFloats).ToTensor().Reshape([|1;1;28;28|])

let mn = MNIST()

mn.Forward(x)

//let y = g mn x

//rd
//|> Falanx.Machinery.Quotations.ToAst
//|> snd
//|> fun x -> CodeFormatter.FormatAST(x, "helloWorld.fs", None, FormatConfig.FormatConfig.Default)
//|> Falanx.Machinery.Expr.cleanUpTypeName
