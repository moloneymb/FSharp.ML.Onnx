module ONNXAPIGraph

open System
open System.Numerics
open System.IO
open System.Text
open Onnx
open Google.Protobuf.Collections
open Microsoft.ML.OnnxRuntime.Tensors
open Microsoft.ML.OnnxRuntime
open ProtoBuf

type ONNXGraph() =
    static member linear_regressor(graph: Graph, X: ValueInfo, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        graph.AddNode("LinearRegressor", [|X|], [||], [|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|]).[0]
    static member imputer(graph: Graph, X: ValueInfo, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        graph.AddNode("Imputer", [|X|], [||], [|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|]).[0]
    static member feature_vectorizer(graph: Graph, [<ParamArray>]X: ValueInfo[], ?inputdimensions: int64[]) =
        graph.AddNode("FeatureVectorizer", (X), [||], [|Attr.ints("inputdimensions", inputdimensions)|]).[0]
    static member binarizer(graph: Graph, X: ValueInfo, ?threshold: float32) =
        graph.AddNode("Binarizer", [|X|], [||], [|Attr.float("threshold", threshold, 0.0f)|]).[0]
    static member array_feature_extractor(graph: Graph, X: ValueInfo, Y: ValueInfo) =
        graph.AddNode("ArrayFeatureExtractor", [|X; Y|], [||], [||]).[0]
    static member svm_regressor(graph: Graph, X: ValueInfo, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        graph.AddNode("SVMRegressor", [|X|], [||], [|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|]).[0]
    static member det(graph: Graph, X: ValueInfo) =
        graph.AddNode("Det", [|X|], [||], [||]).[0]
    static member tree_ensemble_regressor(graph: Graph, X: ValueInfo, ?aggregate_function: string, ?base_values: float32[], ?n_targets: int64, ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string, ?target_ids: int64[], ?target_nodeids: int64[], ?target_treeids: int64[], ?target_weights: float32[]) =
        graph.AddNode("TreeEnsembleRegressor", [|X|], [||], [|Attr.string("aggregate_function", aggregate_function, "SUM"); Attr.floats("base_values", base_values); Attr.int("n_targets", n_targets); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE"); Attr.ints("target_ids", target_ids); Attr.ints("target_nodeids", target_nodeids); Attr.ints("target_treeids", target_treeids); Attr.floats("target_weights", target_weights)|]).[0]
    static member round(graph: Graph, X: ValueInfo) =
        graph.AddNode("Round", [|X|], [||], [||]).[0]
    static member range(graph: Graph, start: ValueInfo, limit: ValueInfo, delta: ValueInfo) =
        graph.AddNode("Range", [|start; limit; delta|], [||], [||]).[0]
    static member thresholded_relu(graph: Graph, X: ValueInfo, ?alpha: float32) =
        graph.AddNode("ThresholdedRelu", [|X|], [||], [|Attr.float("alpha", alpha, 1.0f)|]).[0]
    static member mean_variance_normalization(graph: Graph, X: ValueInfo, ?axes: int64[]) =
        graph.AddNode("MeanVarianceNormalization", [|X|], [||], [|Attr.ints("axes", axes, [|0L;2L;3L|])|]).[0]
    static member non_zero(graph: Graph, X: ValueInfo) =
        graph.AddNode("NonZero", [|X|], [||], [||]).[0]
    static member shrink(graph: Graph, input: ValueInfo, ?bias: float32, ?lambd: float32) =
        graph.AddNode("Shrink", [|input|], [||], [|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|]).[0]
    static member erf(graph: Graph, input: ValueInfo) =
        graph.AddNode("Erf", [|input|], [||], [||]).[0]
    static member atanh(graph: Graph, input: ValueInfo) =
        graph.AddNode("Atanh", [|input|], [||], [||]).[0]
    static member acosh(graph: Graph, input: ValueInfo) =
        graph.AddNode("Acosh", [|input|], [||], [||]).[0]
    static member expand(graph: Graph, input: ValueInfo, shape: ValueInfo) =
        graph.AddNode("Expand", [|input; shape|], [||], [||]).[0]
    static member atan(graph: Graph, input: ValueInfo) =
        graph.AddNode("Atan", [|input|], [||], [||]).[0]
    static member asin(graph: Graph, input: ValueInfo) =
        graph.AddNode("Asin", [|input|], [||], [||]).[0]
    static member lp_normalization(graph: Graph, input: ValueInfo, ?axis: int64, ?p: int64) =
        graph.AddNode("LpNormalization", [|input|], [||], [|Attr.int("axis", axis, -1L); Attr.int("p", p, 2L)|]).[0]
    static member ceil(graph: Graph, X: ValueInfo) =
        graph.AddNode("Ceil", [|X|], [||], [||]).[0]
    static member log_softmax(graph: Graph, input: ValueInfo, ?axis: int64) =
        graph.AddNode("LogSoftmax", [|input|], [||], [|Attr.int("axis", axis, 1L)|]).[0]
    static member mat_mul(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("MatMul", [|A; B|], [||], [||]).[0]
    static member bit_shift(graph: Graph, X: ValueInfo, Y: ValueInfo, direction: string) =
        graph.AddNode("BitShift", [|X; Y|], [||], [|Attr.string("direction", direction)|]).[0]
    static member sinh(graph: Graph, input: ValueInfo) =
        graph.AddNode("Sinh", [|input|], [||], [||]).[0]
    static member acos(graph: Graph, input: ValueInfo) =
        graph.AddNode("Acos", [|input|], [||], [||]).[0]
    static member identity(graph: Graph, input: ValueInfo) =
        graph.AddNode("Identity", [|input|], [||], [||]).[0]
    static member pow(graph: Graph, X: ValueInfo, Y: ValueInfo) =
        graph.AddNode("Pow", [|X; Y|], [||], [||]).[0]
    static member mod_(graph: Graph, A: ValueInfo, B: ValueInfo, ?fmod: int64) =
        graph.AddNode("Mod", [|A; B|], [||], [|Attr.int("fmod", fmod, 0L)|]).[0]
    static member softplus(graph: Graph, X: ValueInfo) =
        graph.AddNode("Softplus", [|X|], [||], [||]).[0]
    static member normalizer(graph: Graph, X: ValueInfo, ?norm: string) =
        graph.AddNode("Normalizer", [|X|], [||], [|Attr.string("norm", norm, "MAX")|]).[0]
    static member hardmax(graph: Graph, input: ValueInfo, ?axis: int64) =
        graph.AddNode("Hardmax", [|input|], [||], [|Attr.int("axis", axis, 1L)|]).[0]
    static member hard_sigmoid(graph: Graph, X: ValueInfo, ?alpha: float32, ?beta: float32) =
        graph.AddNode("HardSigmoid", [|X|], [||], [|Attr.float("alpha", alpha, 0.20000000298023224f); Attr.float("beta", beta, 0.5f)|]).[0]
    static member lp_pool(graph: Graph, X: ValueInfo, kernel_shape: int64[], ?auto_pad: string, ?p: int64, ?pads: int64[], ?strides: int64[]) =
        graph.AddNode("LpPool", [|X|], [||], [|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("p", p, 2L); Attr.ints("pads", pads); Attr.ints("strides", strides)|]).[0]
    static member min(graph: Graph, [<ParamArray>]data_0: ValueInfo[]) =
        graph.AddNode("Min", (data_0), [||], [||]).[0]
    static member sum(graph: Graph, [<ParamArray>]data_0: ValueInfo[]) =
        graph.AddNode("Sum", (data_0), [||], [||]).[0]
    static member transpose(graph: Graph, data: ValueInfo, ?perm: int64[]) =
        graph.AddNode("Transpose", [|data|], [||], [|Attr.ints("perm", perm)|]).[0]
    static member scatternd(graph: Graph, data: ValueInfo, indices: ValueInfo, updates: ValueInfo) =
        graph.AddNode("ScatterND", [|data; indices; updates|], [||], [||]).[0]
    static member global_lp_pool(graph: Graph, X: ValueInfo, ?p: int64) =
        graph.AddNode("GlobalLpPool", [|X|], [||], [|Attr.int("p", p, 2L)|]).[0]
    static member gemm(graph: Graph, A: ValueInfo, B: ValueInfo, ?C: ValueInfo, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        graph.AddNode("Gemm", ([|Some(A); Some(B); C|] |> Array.choose id), [||], [|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|]).[0]
    static member instance_normalization(graph: Graph, input: ValueInfo, scale: ValueInfo, B: ValueInfo, ?epsilon: float32) =
        graph.AddNode("InstanceNormalization", [|input; scale; B|], [||], [|Attr.float("epsilon", epsilon, 9.999999747378752e-06f)|]).[0]
    static member average_pool(graph: Graph, X: ValueInfo, kernel_shape: int64[], ?auto_pad: string, ?count_include_pad: int64, ?pads: int64[], ?strides: int64[]) =
        graph.AddNode("AveragePool", [|X|], [||], [|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("count_include_pad", count_include_pad, 0L); Attr.ints("pads", pads); Attr.ints("strides", strides)|]).[0]
    static member sign(graph: Graph, input: ValueInfo) =
        graph.AddNode("Sign", [|input|], [||], [||]).[0]
    static member clip(graph: Graph, input: ValueInfo, ?min: ValueInfo, ?max: ValueInfo) =
        graph.AddNode("Clip", ([|Some(input); min; max|] |> Array.choose id), [||], [||]).[0]
    static member dequantize_linear(graph: Graph, x: ValueInfo, x_scale: ValueInfo, ?x_zero_point: ValueInfo) =
        graph.AddNode("DequantizeLinear", ([|Some(x); Some(x_scale); x_zero_point|] |> Array.choose id), [||], [||]).[0]
    static member lrn(graph: Graph, X: ValueInfo, size: int64, ?alpha: float32, ?beta: float32, ?bias: float32) =
        graph.AddNode("LRN", [|X|], [||], [|Attr.int("size", size); Attr.float("alpha", alpha, 9.999999747378752e-05f); Attr.float("beta", beta, 0.75f); Attr.float("bias", bias, 1.0f)|]).[0]
    static member elu(graph: Graph, X: ValueInfo, ?alpha: float32) =
        graph.AddNode("Elu", [|X|], [||], [|Attr.float("alpha", alpha, 1.0f)|]).[0]
    static member sin(graph: Graph, input: ValueInfo) =
        graph.AddNode("Sin", [|input|], [||], [||]).[0]
    static member pad(graph: Graph, data: ValueInfo, pads: ValueInfo, ?constant_value: ValueInfo, ?mode: string) =
        graph.AddNode("Pad", ([|Some(data); Some(pads); constant_value|] |> Array.choose id), [||], [|Attr.string("mode", mode, "constant")|]).[0]
    static member gathernd(graph: Graph, data: ValueInfo, indices: ValueInfo) =
        graph.AddNode("GatherND", [|data; indices|], [||], [||]).[0]
    static member relu(graph: Graph, X: ValueInfo) =
        graph.AddNode("Relu", [|X|], [||], [||]).[0]
    static member conv(graph: Graph, X: ValueInfo, W: ValueInfo, ?B: ValueInfo, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        graph.AddNode("Conv", ([|Some(X); Some(W); B|] |> Array.choose id), [||], [|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|]).[0]
    static member arg_max(graph: Graph, data: ValueInfo, ?axis: int64, ?keepdims: int64) =
        graph.AddNode("ArgMax", [|data|], [||], [|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|]).[0]
    static member div(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("Div", [|A; B|], [||], [||]).[0]
    static member max_roi_pool(graph: Graph, X: ValueInfo, rois: ValueInfo, pooled_shape: int64[], ?spatial_scale: float32) =
        graph.AddNode("MaxRoiPool", [|X; rois|], [||], [|Attr.ints("pooled_shape", pooled_shape); Attr.float("spatial_scale", spatial_scale, 1.0f)|]).[0]
    static member add(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("Add", [|A; B|], [||], [||]).[0]
    static member leaky_relu(graph: Graph, X: ValueInfo, ?alpha: float32) =
        graph.AddNode("LeakyRelu", [|X|], [||], [|Attr.float("alpha", alpha, 0.009999999776482582f)|]).[0]
    static member reduce_log_sum(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceLogSum", [|data|], [||], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]).[0]
    static member floor(graph: Graph, X: ValueInfo) =
        graph.AddNode("Floor", [|X|], [||], [||]).[0]
    static member arg_min(graph: Graph, data: ValueInfo, ?axis: int64, ?keepdims: int64) =
        graph.AddNode("ArgMin", [|data|], [||], [|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|]).[0]
    static member depth_to_space(graph: Graph, input: ValueInfo, blocksize: int64, ?mode: string) =
        graph.AddNode("DepthToSpace", [|input|], [||], [|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|]).[0]
    static member tan(graph: Graph, input: ValueInfo) =
        graph.AddNode("Tan", [|input|], [||], [||]).[0]
    static member reduce_sum(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceSum", [|data|], [||], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]).[0]
    static member concat(graph: Graph, axis: int64, [<ParamArray>]inputs: ValueInfo[]) =
        graph.AddNode("Concat", (inputs), [||], [|Attr.int("axis", axis)|]).[0]
    static member one_hot_encoder(graph: Graph, X: ValueInfo, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        graph.AddNode("OneHotEncoder", [|X|], [||], [|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|]).[0]
    static member conv_transpose(graph: Graph, X: ValueInfo, W: ValueInfo, ?B: ValueInfo, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?output_padding: int64[], ?output_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        graph.AddNode("ConvTranspose", ([|Some(X); Some(W); B|] |> Array.choose id), [||], [|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("output_padding", output_padding); Attr.ints("output_shape", output_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|]).[0]
    static member reverse_sequence(graph: Graph, input: ValueInfo, sequence_lens: ValueInfo, ?batch_axis: int64, ?time_axis: int64) =
        graph.AddNode("ReverseSequence", [|input; sequence_lens|], [||], [|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|]).[0]
    static member max(graph: Graph, [<ParamArray>]data_0: ValueInfo[]) =
        graph.AddNode("Max", (data_0), [||], [||]).[0]
    static member global_max_pool(graph: Graph, X: ValueInfo) =
        graph.AddNode("GlobalMaxPool", [|X|], [||], [||]).[0]
    static member exp(graph: Graph, input: ValueInfo) =
        graph.AddNode("Exp", [|input|], [||], [||]).[0]
    static member reshape(graph: Graph, data: ValueInfo, shape: ValueInfo) =
        graph.AddNode("Reshape", [|data; shape|], [||], [||]).[0]
    static member global_average_pool(graph: Graph, X: ValueInfo) =
        graph.AddNode("GlobalAveragePool", [|X|], [||], [||]).[0]
    static member mean(graph: Graph, [<ParamArray>]data_0: ValueInfo[]) =
        graph.AddNode("Mean", (data_0), [||], [||]).[0]
    static member mul(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("Mul", [|A; B|], [||], [||]).[0]
    static member neg(graph: Graph, X: ValueInfo) =
        graph.AddNode("Neg", [|X|], [||], [||]).[0]
    static member not_(graph: Graph, X: ValueInfo) =
        graph.AddNode("Not", [|X|], [||], [||]).[0]
    static member reducel1(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceL1", [|data|], [||], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]).[0]
    static member flatten(graph: Graph, input: ValueInfo, ?axis: int64) =
        graph.AddNode("Flatten", [|input|], [||], [|Attr.int("axis", axis, 1L)|]).[0]
    static member p_relu(graph: Graph, X: ValueInfo, slope: ValueInfo) =
        graph.AddNode("PRelu", [|X; slope|], [||], [||]).[0]
    static member unsqueeze(graph: Graph, data: ValueInfo, axes: int64[]) =
        graph.AddNode("Unsqueeze", [|data|], [||], [|Attr.ints("axes", axes)|]).[0]
    static member tanh(graph: Graph, input: ValueInfo) =
        graph.AddNode("Tanh", [|input|], [||], [||]).[0]
    static member abs(graph: Graph, X: ValueInfo) =
        graph.AddNode("Abs", [|X|], [||], [||]).[0]
    static member reciprocal(graph: Graph, X: ValueInfo) =
        graph.AddNode("Reciprocal", [|X|], [||], [||]).[0]
    static member reduce_log_sum_exp(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceLogSumExp", [|data|], [||], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]).[0]
    static member reduce_max(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceMax", [|data|], [||], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]).[0]
    static member reduce_mean(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceMean", [|data|], [||], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]).[0]
    static member cosh(graph: Graph, input: ValueInfo) =
        graph.AddNode("Cosh", [|input|], [||], [||]).[0]
    static member reduce_min(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceMin", [|data|], [||], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]).[0]
    static member reduce_prod(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceProd", [|data|], [||], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]).[0]
    static member squeeze(graph: Graph, data: ValueInfo, ?axes: int64[]) =
        graph.AddNode("Squeeze", [|data|], [||], [|Attr.ints("axes", axes)|]).[0]
    static member selu(graph: Graph, X: ValueInfo, ?alpha: float32, ?gamma: float32) =
        graph.AddNode("Selu", [|X|], [||], [|Attr.float("alpha", alpha, 1.6732631921768188f); Attr.float("gamma", gamma, 1.0507010221481323f)|]).[0]
    static member sigmoid(graph: Graph, X: ValueInfo) =
        graph.AddNode("Sigmoid", [|X|], [||], [||]).[0]
    static member reduce_sum_square(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceSumSquare", [|data|], [||], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]).[0]
    static member softmax(graph: Graph, input: ValueInfo, ?axis: int64) =
        graph.AddNode("Softmax", [|input|], [||], [|Attr.int("axis", axis, 1L)|]).[0]
    static member softsign(graph: Graph, input: ValueInfo) =
        graph.AddNode("Softsign", [|input|], [||], [||]).[0]
    static member cos(graph: Graph, input: ValueInfo) =
        graph.AddNode("Cos", [|input|], [||], [||]).[0]
    static member space_to_depth(graph: Graph, input: ValueInfo, blocksize: int64) =
        graph.AddNode("SpaceToDepth", [|input|], [||], [|Attr.int("blocksize", blocksize)|]).[0]
    static member asinh(graph: Graph, input: ValueInfo) =
        graph.AddNode("Asinh", [|input|], [||], [||]).[0]
    static member reducel2(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceL2", [|data|], [||], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]).[0]
    static member sqrt(graph: Graph, X: ValueInfo) =
        graph.AddNode("Sqrt", [|X|], [||], [||]).[0]
    static member log(graph: Graph, input: ValueInfo) =
        graph.AddNode("Log", [|input|], [||], [||]).[0]
    static member sub(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("Sub", [|A; B|], [||], [||]).[0]
    static member scaler(graph: Graph, X: ValueInfo, ?offset: float32[], ?scale: float32[]) =
        graph.AddNode("Scaler", [|X|], [||], [|Attr.floats("offset", offset); Attr.floats("scale", scale)|]).[0]
    static member upsample(graph: Graph, X: ValueInfo, scales: ValueInfo, ?mode: string) =
        graph.AddNode("Upsample", [|X; scales|], [||], [|Attr.string("mode", mode, "nearest")|]).[0]
