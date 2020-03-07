namespace Test
open Microsoft.ML.OnnxRuntime.Tensors
open NUnit.Framework
open Onnx
open ProtoBuf
open System
open System.IO
open Microsoft.ML.OnnxRuntime



module MiniGraphs = 
    type on = ONNXAPI.ONNX
    type Tensor<'a> with
        member this.shape = this.Dimensions.ToArray()

    let input1 = ArrayTensorExtensions.ToTensor(Array2D.create 1 32 2.f) :> Tensor<float32>
    let input2 = ArrayTensorExtensions.ToTensor(Array2D.create 32 1 3.f) :> Tensor<float32>

    let input1Int = ArrayTensorExtensions.ToTensor(Array2D.create 1 32 2L) :> Tensor<int64>
    let input2Int = ArrayTensorExtensions.ToTensor(Array2D.create 32 1 3L) :> Tensor<int64>

    let input4D1 = ArrayTensorExtensions.ToTensor(Array4D.create 3 3 1 3 2.f) :> Tensor<float32>
    let input4D2 = ArrayTensorExtensions.ToTensor(Array4D.create 3 3 3 1 1.f) :> Tensor<float32>


    [<Test>]
    let ``add float``() = 
        let add x y = buildAndRunBinary "Add" x y [||]
        let res1 = add input1 input2
        let res2 = add input1Int input2Int
        if res1.Dimensions.ToArray() <> [|32;32|] then failwith "Incorrect dimmesions"
        if res1 |> Seq.exists (fun x -> x <> 5.f) then failwith "An incorrect value"
        if res2.Dimensions.ToArray() <> [|32;32|] then failwith "Incorrect dimmesions"
        if res2 |> Seq.exists (fun x -> x <> 5L) then failwith "An incorrect value"

    [<Test>]
    let relu() = 
        let xx = Array2D.create 2 2 0.f
        xx.[0,0] <- -1.0f
        xx.[1,1] <- 1.0f
        let res = buildAndRunUnary "Relu" (ArrayTensorExtensions.ToTensor(xx) :> Tensor<float32>) [||]
        Assert.AreEqual(float res.[0,0],float 0.0f,0.001)
        Assert.AreEqual(float res.[1,1],float 1.0f,0.001)

    [<Test>]
    let convolution() = 
        let buildAndRunConv (input:Tensor<'a>, kernel:Tensor<'a>, kernel_shape: int64[], strides: int64[], auto_pad: string, group: int64, dilations: int64[]) = 
            runSingleOutputNode (cnn("Op", "Input1", "Input2" , "Output1",kernel_shape, strides, auto_pad,group,dilations)) [|kernel;kernel|]

        let img = ArrayTensorExtensions.ToTensor(Array4D.create 1 1 32 32 1.f) :> Tensor<float32>
        let kernel = ArrayTensorExtensions.ToTensor(Array4D.create 8 1 5 5 1.f) :> Tensor<float32>
        let convRes = buildAndRunConv(img,kernel,[|5L;5L|],[|1L;1L;1L;1L|],"SAME_UPPER",1L,[|1L;1L|])
        Assert.AreEqual(float convRes.[0,0,0,0], 9.0, 0.001)
        Assert.AreEqual(float convRes.[0,0,5,5], 12.0, 0.001)

    let matmul x y = buildAndRunBinary "MatMul" x y [||]

    [<Test>]
    let ``matmul broadcast``() = 
        let res1 = matmul input1 input2
        Assert.AreEqual(res1.Dimensions.ToArray(), [|1;1|])
        Assert.AreEqual(res1.[0], 192.0f)
        let res2 = matmul input2 input1
        Assert.AreEqual(res2.Dimensions.ToArray(), [|32;32|])
        Assert.AreEqual(res2.[0], 6.0f)

    [<Test>]
    let ``matmul batch``() = 
        let res1 = matmul input4D1 input4D2
        Assert.AreEqual(res1.Dimensions.ToArray(), [|3;3;1;1|])
        Assert.AreEqual(res1.[0], 6.0f)

    /// This 
    [<Test>]
    let ``eager api``() =
        let input1 = ArrayTensorExtensions.ToTensor(Array2D.create 10000 40 -2.f) :> Tensor<float32>
        let input2 = ArrayTensorExtensions.ToTensor(Array2D.create 40 10000 -2.f) :> Tensor<float32>
        let res = on.MatMul(input2,on.Abs(input1))
        Assert.AreEqual(res.shape, [|40;40|], "testing shape")
        Assert.AreEqual(float res.[0,0], -40000., 0.00001, "testing math")
        

module FullModel = 

    let shouldEqual (msg: string) (v1: 'T) (v2: 'T) = 
        if v1 <> v2 then 
            Assert.Fail(sprintf "fail %s: expected %A, got %A" msg v1 v2)

    let mnistDir = Path.Combine(__SOURCE_DIRECTORY__,"..","data","mnist")

    let test_data = 
        lazy
            let f(path: string) = 
                TensorProto.Parser.ParseFrom(File.ReadAllBytes(path))
            [| for i in [0;1;2] ->
                    Path.Combine(mnistDir,sprintf "test_data_set_0") 
                    |> fun dir -> (f(Path.Combine(dir,"input_0.pb")),f(Path.Combine(dir,"output_0.pb")))|]

    let testModel(model : byte[]) = 
        use sess = new InferenceSession(model)
        for (index,(input,output)) in test_data.Force() |> Array.indexed do
            use values2 = sess.Run([|NamedOnnxValue.CreateFromTensor("Input3",Tensor.FromTensorProtoFloat32(input))|])
            let diff = 
                (values2 |> Seq.toArray |> Array.head |> fun v -> v.AsTensor<float32>() |> Seq.toArray, Tensor.FromTensorProtoFloat32(output) |> Seq.toArray)
                ||> Array.zip
                |> Array.sumBy (fun (x,y) -> System.Math.Abs(x-y))
            if diff > 0.1f then failwithf "Unexpected result in example %i with a difference of %f" index diff


    [<Test>]
    let ``prebuilt model``() = 
        File.ReadAllBytes(Path.Combine(mnistDir, "model.onnx"))
        |> testModel


    /// This is a full MNist example that exactly matches the pre-trained model
    [<Test>]
    let ``code model``() =
        let nodes = 
            [|
                reshape ("Times212_reshape1","Parameter193", "Parameter193_reshape1_shape","Parameter193_reshape1")
                cnn("Convolution28","Input3","Parameter5","Convolution28_Output_0",[|5L;5L|],[|1L;1L|],"SAME_UPPER",1L,[|1L;1L|])
                add ("Plus30", "Convolution28_Output_0", "Parameter6","Plus30_Output_0")
                relu("ReLU32","Plus30_Output_0","ReLU32_Output_0")
                maxPool("Pooling66","ReLU32_Output_0", "Pooling66_Output_0", [|2L;2L|],[|2L;2L|],[|0L;0L;0L;0L|],"NOTSET")
                cnn("Convolution110","Pooling66_Output_0","Parameter87","Convolution110_Output_0",[|5L;5L|],[|1L;1L|],"SAME_UPPER",1L,[|1L;1L|])
                add ("Plus112", "Convolution110_Output_0", "Parameter88" ,"Plus112_Output_0")
                relu("ReLU114", "Plus112_Output_0", "ReLU114_Output_0")
                maxPool("Pooling160","ReLU114_Output_0", "Pooling160_Output_0", [|3L;3L|],[|3L;3L|],[|0L;0L;0L;0L|],"NOTSET")
                reshape("Times212_reshape0","Pooling160_Output_0", "Pooling160_Output_0_reshape0_shape","Pooling160_Output_0_reshape0")
                matmul("Times212", "Pooling160_Output_0_reshape0", "Parameter193_reshape1", "Times212_Output_0")
                add("Plus214", "Times212_Output_0", "Parameter194" , "Plus214_Output_0")
            |]

        let tensorProtos = 
            [|
                "Parameter193", DataType.FLOAT32, [|16L; 4L; 4L; 10L|]
                "Parameter87", DataType.FLOAT32, [|16L; 8L; 5L; 5L|]
                "Parameter5", DataType.FLOAT32, [|8L; 1L; 5L; 5L|]
                "Parameter6", DataType.FLOAT32, [|8L; 1L; 1L|]
                "Parameter88", DataType.FLOAT32, [|16L; 1L; 1L|]
                "Pooling160_Output_0_reshape0_shape", DataType.INT64, [|2L|]
                "Parameter193_reshape1_shape", DataType.INT64, [|2L|]
                "Parameter194", DataType.FLOAT32, [|1L; 10L|]
            |] |> Array.map (fun (name, dt,dims) -> 
                let tp = TensorProto(DataType = int dt, Name = name)
                tp.Dims.AddRange(dims)
                let path = Path.Combine(mnistDir, name)
                let data = File.ReadAllBytes(path)
                match dt with
                | DataType.FLOAT32 -> 
                    tp.FloatData.AddRange(data |> bytesToFloats)
                | DataType.INT64 -> 
                    tp.Int64Data.AddRange(data |> bytesToInts)
                | _ -> failwith "err"
                tp)

        let inputs = 
            [| 
                "Input3", DataType.FLOAT32, [|1L;1L;28L;28L|]
                "Parameter5", DataType.FLOAT32, [|8L;1L;5L;5L|]
                "Parameter6", DataType.FLOAT32, [|8L;1L;1L|]
                "Parameter87", DataType.FLOAT32, [|16L;8L;5L;5L|]
                "Parameter88", DataType.FLOAT32, [|16L;1L;1L|]
                "Pooling160_Output_0_reshape0_shape", DataType.INT64, [|2L|]
                "Parameter193",DataType.FLOAT32,[|16L;4L;4L;10L|]
                "Parameter193_reshape1_shape", DataType.INT64,[|2L|]
                "Parameter194", DataType.FLOAT32,[|1L;10L|]
            |]
            |> Array.map (fun (name,dt,shape) -> ValueInfoProto(DocString = "", Name = name, Type = TypeProto(TensorType = TypeProto.Types.Tensor(ElemType = int32 dt, Shape = makeShape shape))))

        let outputs =
            [|"Plus214_Output_0", DataType.FLOAT32,[|1L;10L|]|]
            |> Array.map (fun (name,dt,shape) -> ValueInfoProto(DocString = "", Name = name, Type = TypeProto(TensorType = TypeProto.Types.Tensor(ElemType = int32 dt, Shape = makeShape shape))))

        let valueInfo = 
            [|
                "Parameter193_reshape1", DataType.FLOAT32, [|256L;10L|]
                "Convolution28_Output_0", DataType.FLOAT32, [|1L;8L;28L;28L|]
                "Plus30_Output_0", DataType.FLOAT32, [|1L;8L;28L;28L|]
                "ReLU32_Output_0", DataType.FLOAT32, [|1L;8L;28L;28L|]
                "Pooling66_Output_0", DataType.FLOAT32, [|1L;8L;14L;14L|]
                "Convolution110_Output_0", DataType.FLOAT32, [|1L;16L;14L;14L|]
                "Plus112_Output_0", DataType.FLOAT32, [|1L;16L;14L;14L|]
                "ReLU114_Output_0", DataType.FLOAT32, [|1L;16L;14L;14L|]
                "Pooling160_Output_0", DataType.FLOAT32, [|1L;16L;4L;4L|]
                "Pooling160_Output_0_reshape0", DataType.FLOAT32, [|1L; 256L|]
                "Times212_Output_0", DataType.FLOAT32, [|1L;10L|]
            |]
            |> Array.map (fun (name,dt,shape) -> ValueInfoProto(DocString = "", Name = name, Type = TypeProto(TensorType = TypeProto.Types.Tensor(ElemType = int32 dt, Shape = makeShape shape))))

        let mp = 
            let graph = GraphProto(Name = "CNTKGraph")
            graph.Input.AddRange(inputs)
            graph.Output.AddRange(outputs)
            graph.ValueInfo.AddRange(valueInfo)
            graph.Node.AddRange(nodes)
            graph.Initializer.AddRange(tensorProtos)
            let mp = 
                ModelProto(DocString = "",
                    Domain = "ai.cntk",
                    IrVersion = 3L,
                    ModelVersion = 1L,
                    ProducerName = "CNTK",
                    ProducerVersion = "2.5.1",
                    Graph = graph)
            mp.OpsetImport.Add(OperatorSetIdProto(Version = 8L))
            mp

        let mpData = writeModelToStream(mp)

        mpData |> testModel


//    [<Test>]
//    let ``eager mnist``() = 
//        failwith "in-progress"
        //let input = 
//        reshape ("Times212_reshape1","Parameter193", "Parameter193_reshape1_shape","Parameter193_reshape1")
//        cnn("Convolution28","Input3","Parameter5","Convolution28_Output_0",[|5L;5L|],[|1L;1L|],"SAME_UPPER",1L,[|1L;1L|])
//        add ("Plus30", "Convolution28_Output_0", "Parameter6","Plus30_Output_0")
//        relu("ReLU32","Plus30_Output_0","ReLU32_Output_0")
//        maxPool("Pooling66","ReLU32_Output_0", "Pooling66_Output_0", [|2L;2L|],[|2L;2L|],[|0L;0L;0L;0L|],"NOTSET")
//        cnn("Convolution110","Pooling66_Output_0","Parameter87","Convolution110_Output_0",[|5L;5L|],[|1L;1L|],"SAME_UPPER",1L,[|1L;1L|])
//        add ("Plus112", "Convolution110_Output_0", "Parameter88" ,"Plus112_Output_0")
//        relu("ReLU114", "Plus112_Output_0", "ReLU114_Output_0")
//        maxPool("Pooling160","ReLU114_Output_0", "Pooling160_Output_0", [|3L;3L|],[|3L;3L|],[|0L;0L;0L;0L|],"NOTSET")
//        reshape("Times212_reshape0","Pooling160_Output_0", "Pooling160_Output_0_reshape0_shape","Pooling160_Output_0_reshape0")
//        matmul("Times212", "Pooling160_Output_0_reshape0", "Parameter193_reshape1", "Times212_Output_0")
//        add("Plus214", "Times212_Output_0", "Parameter194" , "Plus214_Output_0")


module ONNXExample = 
    [<Test>]
    let ``squeezenet example``() = 
        let loadTensorFromFile(filename: string) = 
            File.ReadAllLines(filename).[1..]
            |> Array.collect (fun line -> line.Split([|',';'[';']'|], StringSplitOptions.RemoveEmptyEntries))
            |> Array.map Single.Parse

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

        // TODO verify output
        ()
//        for r in results do
//            printfn "Output for %s" r.Name
//            printfn "%s" (r.AsTensor<float32>().GetArrayString())





