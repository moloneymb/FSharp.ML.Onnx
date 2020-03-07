module ONNXAPI

open System.Text
open System.IO
open Onnx
open Google.Protobuf.Collections
open Microsoft.ML.OnnxRuntime.Tensors
open Microsoft.ML.OnnxRuntime
open ProtoBuf

type ONNX() =
    static member LinearRegressor(X: Tensor<float32>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        buildAndRunUnary "LinearRegressor" X ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member LinearRegressor(X: Tensor<int64>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        buildAndRunUnary "LinearRegressor" X ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member LinearRegressor(X: Tensor<int>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        buildAndRunUnary "LinearRegressor" X ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member Imputer(X: Tensor<float32>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        buildAndRunUnary "Imputer" X ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member Imputer(X: Tensor<int64>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        buildAndRunUnary "Imputer" X ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member Imputer(X: Tensor<int>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        buildAndRunUnary "Imputer" X ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member Binarizer(X: Tensor<float32>, ?threshold: float32) =
        buildAndRunUnary "Binarizer" X ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member Binarizer(X: Tensor<int64>, ?threshold: float32) =
        buildAndRunUnary "Binarizer" X ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member Binarizer(X: Tensor<int>, ?threshold: float32) =
        buildAndRunUnary "Binarizer" X ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member SVMRegressor(X: Tensor<float32>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        buildAndRunUnary "SVMRegressor" X ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member SVMRegressor(X: Tensor<int64>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        buildAndRunUnary "SVMRegressor" X ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member SVMRegressor(X: Tensor<int>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        buildAndRunUnary "SVMRegressor" X ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member Det(X: Tensor<float32>) =        buildAndRunUnary "Det" X [||]
    static member TreeEnsembleRegressor(X: Tensor<float32>, ?aggregate_function: string, ?base_values: float32[], ?n_targets: int64, ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string, ?target_ids: int64[], ?target_nodeids: int64[], ?target_treeids: int64[], ?target_weights: float32[]) =
        buildAndRunUnary "TreeEnsembleRegressor" X ([|Attr.string("aggregate_function", aggregate_function, "SUM"); Attr.floats("base_values", base_values); Attr.int("n_targets", n_targets); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE"); Attr.ints("target_ids", target_ids); Attr.ints("target_nodeids", target_nodeids); Attr.ints("target_treeids", target_treeids); Attr.floats("target_weights", target_weights)|] |> Array.choose id)
    static member TreeEnsembleRegressor(X: Tensor<int64>, ?aggregate_function: string, ?base_values: float32[], ?n_targets: int64, ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string, ?target_ids: int64[], ?target_nodeids: int64[], ?target_treeids: int64[], ?target_weights: float32[]) =
        buildAndRunUnary "TreeEnsembleRegressor" X ([|Attr.string("aggregate_function", aggregate_function, "SUM"); Attr.floats("base_values", base_values); Attr.int("n_targets", n_targets); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE"); Attr.ints("target_ids", target_ids); Attr.ints("target_nodeids", target_nodeids); Attr.ints("target_treeids", target_treeids); Attr.floats("target_weights", target_weights)|] |> Array.choose id)
    static member TreeEnsembleRegressor(X: Tensor<int>, ?aggregate_function: string, ?base_values: float32[], ?n_targets: int64, ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string, ?target_ids: int64[], ?target_nodeids: int64[], ?target_treeids: int64[], ?target_weights: float32[]) =
        buildAndRunUnary "TreeEnsembleRegressor" X ([|Attr.string("aggregate_function", aggregate_function, "SUM"); Attr.floats("base_values", base_values); Attr.int("n_targets", n_targets); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE"); Attr.ints("target_ids", target_ids); Attr.ints("target_nodeids", target_nodeids); Attr.ints("target_treeids", target_treeids); Attr.floats("target_weights", target_weights)|] |> Array.choose id)
    static member Round(X: Tensor<float32>) =        buildAndRunUnary "Round" X [||]
    static member ThresholdedRelu(X: Tensor<float32>, ?alpha: float32) =
        buildAndRunUnary "ThresholdedRelu" X ([|Attr.float("alpha", alpha, 1.0f)|] |> Array.choose id)
    static member MeanVarianceNormalization(X: Tensor<float32>, ?axes: int64[]) =
        buildAndRunUnary "MeanVarianceNormalization" X ([|Attr.ints("axes", axes, [|0L;2L;3L|])|] |> Array.choose id)
    static member NonZero(X: Tensor<uint8>) =        buildAndRunUnary "NonZero" X [||]
    static member NonZero(X: Tensor<int8>) =        buildAndRunUnary "NonZero" X [||]
    static member NonZero(X: Tensor<int>) =        buildAndRunUnary "NonZero" X [||]
    static member NonZero(X: Tensor<int64>) =        buildAndRunUnary "NonZero" X [||]
    static member NonZero(X: Tensor<float32>) =        buildAndRunUnary "NonZero" X [||]
    static member NonZero(X: Tensor<string>) =        buildAndRunUnary "NonZero" X [||]
    static member NonZero(X: Tensor<bool>) =        buildAndRunUnary "NonZero" X [||]
    static member Shrink(input: Tensor<uint8>, ?bias: float32, ?lambd: float32) =
        buildAndRunUnary "Shrink" input ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<int8>, ?bias: float32, ?lambd: float32) =
        buildAndRunUnary "Shrink" input ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<int>, ?bias: float32, ?lambd: float32) =
        buildAndRunUnary "Shrink" input ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<int64>, ?bias: float32, ?lambd: float32) =
        buildAndRunUnary "Shrink" input ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<float32>, ?bias: float32, ?lambd: float32) =
        buildAndRunUnary "Shrink" input ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Erf(input: Tensor<uint8>) =        buildAndRunUnary "Erf" input [||]
    static member Erf(input: Tensor<int8>) =        buildAndRunUnary "Erf" input [||]
    static member Erf(input: Tensor<int>) =        buildAndRunUnary "Erf" input [||]
    static member Erf(input: Tensor<int64>) =        buildAndRunUnary "Erf" input [||]
    static member Erf(input: Tensor<float32>) =        buildAndRunUnary "Erf" input [||]
    static member Atanh(input: Tensor<float32>) =        buildAndRunUnary "Atanh" input [||]
    static member Acosh(input: Tensor<float32>) =        buildAndRunUnary "Acosh" input [||]
    static member Atan(input: Tensor<float32>) =        buildAndRunUnary "Atan" input [||]
    static member Asin(input: Tensor<float32>) =        buildAndRunUnary "Asin" input [||]
    static member LpNormalization(input: Tensor<float32>, ?axis: int64, ?p: int64) =
        buildAndRunUnary "LpNormalization" input ([|Attr.int("axis", axis, -1L); Attr.int("p", p, 2L)|] |> Array.choose id)
    static member Ceil(X: Tensor<float32>) =        buildAndRunUnary "Ceil" X [||]
    static member LogSoftmax(input: Tensor<float32>, ?axis: int64) =
        buildAndRunUnary "LogSoftmax" input ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Sinh(input: Tensor<float32>) =        buildAndRunUnary "Sinh" input [||]
    static member Acos(input: Tensor<float32>) =        buildAndRunUnary "Acos" input [||]
    static member Identity(input: Tensor<uint8>) =        buildAndRunUnary "Identity" input [||]
    static member Identity(input: Tensor<int8>) =        buildAndRunUnary "Identity" input [||]
    static member Identity(input: Tensor<int>) =        buildAndRunUnary "Identity" input [||]
    static member Identity(input: Tensor<int64>) =        buildAndRunUnary "Identity" input [||]
    static member Identity(input: Tensor<float32>) =        buildAndRunUnary "Identity" input [||]
    static member Identity(input: Tensor<string>) =        buildAndRunUnary "Identity" input [||]
    static member Identity(input: Tensor<bool>) =        buildAndRunUnary "Identity" input [||]
    static member Softplus(X: Tensor<float32>) =        buildAndRunUnary "Softplus" X [||]
    static member Normalizer(X: Tensor<float32>, ?norm: string) =
        buildAndRunUnary "Normalizer" X ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member Normalizer(X: Tensor<int64>, ?norm: string) =
        buildAndRunUnary "Normalizer" X ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member Normalizer(X: Tensor<int>, ?norm: string) =
        buildAndRunUnary "Normalizer" X ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member Hardmax(input: Tensor<float32>, ?axis: int64) =
        buildAndRunUnary "Hardmax" input ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member HardSigmoid(X: Tensor<float32>, ?alpha: float32, ?beta: float32) =
        buildAndRunUnary "HardSigmoid" X ([|Attr.float("alpha", alpha, 0.20000000298023224f); Attr.float("beta", beta, 0.5f)|] |> Array.choose id)
    static member LpPool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?p: int64, ?pads: int64[], ?strides: int64[]) =
        buildAndRunUnary "LpPool" X ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("p", p, 2L); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Transpose(data: Tensor<uint8>, ?perm: int64[]) =
        buildAndRunUnary "Transpose" data ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<int8>, ?perm: int64[]) =
        buildAndRunUnary "Transpose" data ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<int>, ?perm: int64[]) =
        buildAndRunUnary "Transpose" data ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<int64>, ?perm: int64[]) =
        buildAndRunUnary "Transpose" data ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<float32>, ?perm: int64[]) =
        buildAndRunUnary "Transpose" data ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<string>, ?perm: int64[]) =
        buildAndRunUnary "Transpose" data ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<bool>, ?perm: int64[]) =
        buildAndRunUnary "Transpose" data ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member GlobalLpPool(X: Tensor<float32>, ?p: int64) =
        buildAndRunUnary "GlobalLpPool" X ([|Attr.int("p", p, 2L)|] |> Array.choose id)
    static member AveragePool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?ceil_mode: int64, ?count_include_pad: int64, ?pads: int64[], ?strides: int64[]) =
        buildAndRunUnary "AveragePool" X ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("ceil_mode", ceil_mode, 0L); Attr.int("count_include_pad", count_include_pad, 0L); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Sign(input: Tensor<uint8>) =        buildAndRunUnary "Sign" input [||]
    static member Sign(input: Tensor<int8>) =        buildAndRunUnary "Sign" input [||]
    static member Sign(input: Tensor<int>) =        buildAndRunUnary "Sign" input [||]
    static member Sign(input: Tensor<int64>) =        buildAndRunUnary "Sign" input [||]
    static member Sign(input: Tensor<float32>) =        buildAndRunUnary "Sign" input [||]
    static member LRN(X: Tensor<float32>, size: int64, ?alpha: float32, ?beta: float32, ?bias: float32) =
        buildAndRunUnary "LRN" X ([|Attr.int("size", size); Attr.float("alpha", alpha, 9.999999747378752e-05f); Attr.float("beta", beta, 0.75f); Attr.float("bias", bias, 1.0f)|] |> Array.choose id)
    static member Elu(X: Tensor<float32>, ?alpha: float32) =
        buildAndRunUnary "Elu" X ([|Attr.float("alpha", alpha, 1.0f)|] |> Array.choose id)
    static member Sin(input: Tensor<float32>) =        buildAndRunUnary "Sin" input [||]
    static member Relu(X: Tensor<float32>) =        buildAndRunUnary "Relu" X [||]
    static member ArgMax(data: Tensor<uint8>, ?axis: int64, ?keepdims: int64) =
        buildAndRunUnary "ArgMax" data ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<int8>, ?axis: int64, ?keepdims: int64) =
        buildAndRunUnary "ArgMax" data ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<int>, ?axis: int64, ?keepdims: int64) =
        buildAndRunUnary "ArgMax" data ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<int64>, ?axis: int64, ?keepdims: int64) =
        buildAndRunUnary "ArgMax" data ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<float32>, ?axis: int64, ?keepdims: int64) =
        buildAndRunUnary "ArgMax" data ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member LeakyRelu(X: Tensor<float32>, ?alpha: float32) =
        buildAndRunUnary "LeakyRelu" X ([|Attr.float("alpha", alpha, 0.009999999776482582f)|] |> Array.choose id)
    static member ReduceLogSum(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceLogSum" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceLogSum(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceLogSum" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceLogSum(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceLogSum" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Floor(X: Tensor<float32>) =        buildAndRunUnary "Floor" X [||]
    static member ArgMin(data: Tensor<uint8>, ?axis: int64, ?keepdims: int64) =
        buildAndRunUnary "ArgMin" data ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<int8>, ?axis: int64, ?keepdims: int64) =
        buildAndRunUnary "ArgMin" data ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<int>, ?axis: int64, ?keepdims: int64) =
        buildAndRunUnary "ArgMin" data ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<int64>, ?axis: int64, ?keepdims: int64) =
        buildAndRunUnary "ArgMin" data ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<float32>, ?axis: int64, ?keepdims: int64) =
        buildAndRunUnary "ArgMin" data ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<uint8>, blocksize: int64, ?mode: string) =
        buildAndRunUnary "DepthToSpace" input ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<int8>, blocksize: int64, ?mode: string) =
        buildAndRunUnary "DepthToSpace" input ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<int>, blocksize: int64, ?mode: string) =
        buildAndRunUnary "DepthToSpace" input ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<int64>, blocksize: int64, ?mode: string) =
        buildAndRunUnary "DepthToSpace" input ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<float32>, blocksize: int64, ?mode: string) =
        buildAndRunUnary "DepthToSpace" input ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<string>, blocksize: int64, ?mode: string) =
        buildAndRunUnary "DepthToSpace" input ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<bool>, blocksize: int64, ?mode: string) =
        buildAndRunUnary "DepthToSpace" input ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member Tan(input: Tensor<float32>) =        buildAndRunUnary "Tan" input [||]
    static member ReduceSum(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceSum" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceSum(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceSum" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceSum(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceSum" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member OneHotEncoder(X: Tensor<string>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        buildAndRunUnary "OneHotEncoder" X ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member OneHotEncoder(X: Tensor<int64>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        buildAndRunUnary "OneHotEncoder" X ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member OneHotEncoder(X: Tensor<int>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        buildAndRunUnary "OneHotEncoder" X ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member OneHotEncoder(X: Tensor<float32>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        buildAndRunUnary "OneHotEncoder" X ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member GlobalMaxPool(X: Tensor<float32>) =        buildAndRunUnary "GlobalMaxPool" X [||]
    static member Exp(input: Tensor<float32>) =        buildAndRunUnary "Exp" input [||]
    static member GlobalAveragePool(X: Tensor<float32>) =        buildAndRunUnary "GlobalAveragePool" X [||]
    static member Neg(X: Tensor<float32>) =        buildAndRunUnary "Neg" X [||]
    static member Neg(X: Tensor<int>) =        buildAndRunUnary "Neg" X [||]
    static member Neg(X: Tensor<int8>) =        buildAndRunUnary "Neg" X [||]
    static member Neg(X: Tensor<int64>) =        buildAndRunUnary "Neg" X [||]
    static member Not(X: Tensor<bool>) =        buildAndRunUnary "Not" X [||]
    static member ReduceL1(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceL1" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceL1(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceL1" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceL1(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceL1" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<uint8>, ?axis: int64) =
        buildAndRunUnary "Flatten" input ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<int8>, ?axis: int64) =
        buildAndRunUnary "Flatten" input ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<int>, ?axis: int64) =
        buildAndRunUnary "Flatten" input ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<int64>, ?axis: int64) =
        buildAndRunUnary "Flatten" input ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<float32>, ?axis: int64) =
        buildAndRunUnary "Flatten" input ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<string>, ?axis: int64) =
        buildAndRunUnary "Flatten" input ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<bool>, ?axis: int64) =
        buildAndRunUnary "Flatten" input ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<uint8>, axes: int64[]) =
        buildAndRunUnary "Unsqueeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<int8>, axes: int64[]) =
        buildAndRunUnary "Unsqueeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<int>, axes: int64[]) =
        buildAndRunUnary "Unsqueeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<int64>, axes: int64[]) =
        buildAndRunUnary "Unsqueeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<float32>, axes: int64[]) =
        buildAndRunUnary "Unsqueeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<string>, axes: int64[]) =
        buildAndRunUnary "Unsqueeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<bool>, axes: int64[]) =
        buildAndRunUnary "Unsqueeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Tanh(input: Tensor<float32>) =        buildAndRunUnary "Tanh" input [||]
    static member Abs(X: Tensor<uint8>) =        buildAndRunUnary "Abs" X [||]
    static member Abs(X: Tensor<int8>) =        buildAndRunUnary "Abs" X [||]
    static member Abs(X: Tensor<int>) =        buildAndRunUnary "Abs" X [||]
    static member Abs(X: Tensor<int64>) =        buildAndRunUnary "Abs" X [||]
    static member Abs(X: Tensor<float32>) =        buildAndRunUnary "Abs" X [||]
    static member Reciprocal(X: Tensor<float32>) =        buildAndRunUnary "Reciprocal" X [||]
    static member ReduceLogSumExp(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceLogSumExp" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceLogSumExp(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceLogSumExp" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceLogSumExp(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceLogSumExp" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceMax" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceMax" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceMax" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMean(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceMean" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMean(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceMean" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMean(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceMean" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Cosh(input: Tensor<float32>) =        buildAndRunUnary "Cosh" input [||]
    static member ReduceMin(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceMin" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMin(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceMin" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMin(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceMin" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceProd(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceProd" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceProd(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceProd" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceProd(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceProd" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Squeeze(data: Tensor<uint8>, ?axes: int64[]) =
        buildAndRunUnary "Squeeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Squeeze(data: Tensor<int8>, ?axes: int64[]) =
        buildAndRunUnary "Squeeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Squeeze(data: Tensor<int>, ?axes: int64[]) =
        buildAndRunUnary "Squeeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Squeeze(data: Tensor<int64>, ?axes: int64[]) =
        buildAndRunUnary "Squeeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Squeeze(data: Tensor<float32>, ?axes: int64[]) =
        buildAndRunUnary "Squeeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Squeeze(data: Tensor<string>, ?axes: int64[]) =
        buildAndRunUnary "Squeeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Squeeze(data: Tensor<bool>, ?axes: int64[]) =
        buildAndRunUnary "Squeeze" data ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Selu(X: Tensor<float32>, ?alpha: float32, ?gamma: float32) =
        buildAndRunUnary "Selu" X ([|Attr.float("alpha", alpha, 1.6732631921768188f); Attr.float("gamma", gamma, 1.0507010221481323f)|] |> Array.choose id)
    static member Sigmoid(X: Tensor<float32>) =        buildAndRunUnary "Sigmoid" X [||]
    static member ReduceSumSquare(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceSumSquare" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceSumSquare(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceSumSquare" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceSumSquare(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceSumSquare" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Softmax(input: Tensor<float32>, ?axis: int64) =
        buildAndRunUnary "Softmax" input ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Softsign(input: Tensor<float32>) =        buildAndRunUnary "Softsign" input [||]
    static member Cos(input: Tensor<float32>) =        buildAndRunUnary "Cos" input [||]
    static member SpaceToDepth(input: Tensor<uint8>, blocksize: int64) =
        buildAndRunUnary "SpaceToDepth" input ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<int8>, blocksize: int64) =
        buildAndRunUnary "SpaceToDepth" input ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<int>, blocksize: int64) =
        buildAndRunUnary "SpaceToDepth" input ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<int64>, blocksize: int64) =
        buildAndRunUnary "SpaceToDepth" input ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<float32>, blocksize: int64) =
        buildAndRunUnary "SpaceToDepth" input ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<string>, blocksize: int64) =
        buildAndRunUnary "SpaceToDepth" input ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<bool>, blocksize: int64) =
        buildAndRunUnary "SpaceToDepth" input ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member Asinh(input: Tensor<float32>) =        buildAndRunUnary "Asinh" input [||]
    static member ReduceL2(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceL2" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceL2(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceL2" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceL2(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        buildAndRunUnary "ReduceL2" data ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Sqrt(X: Tensor<float32>) =        buildAndRunUnary "Sqrt" X [||]
    static member Log(input: Tensor<float32>) =        buildAndRunUnary "Log" input [||]
    static member Scaler(X: Tensor<float32>, ?offset: float32[], ?scale: float32[]) =
        buildAndRunUnary "Scaler" X ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member Scaler(X: Tensor<int64>, ?offset: float32[], ?scale: float32[]) =
        buildAndRunUnary "Scaler" X ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member Scaler(X: Tensor<int>, ?offset: float32[], ?scale: float32[]) =
        buildAndRunUnary "Scaler" X ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member ArrayFeatureExtractor(X: Tensor<float32>, Y: Tensor<float32>) =
        buildAndRunBinary "ArrayFeatureExtractor" X Y ([||] |> Array.choose id)
    static member ArrayFeatureExtractor(X: Tensor<int64>, Y: Tensor<int64>) =
        buildAndRunBinary "ArrayFeatureExtractor" X Y ([||] |> Array.choose id)
    static member ArrayFeatureExtractor(X: Tensor<int>, Y: Tensor<int>) =
        buildAndRunBinary "ArrayFeatureExtractor" X Y ([||] |> Array.choose id)
    static member ArrayFeatureExtractor(X: Tensor<string>, Y: Tensor<string>) =
        buildAndRunBinary "ArrayFeatureExtractor" X Y ([||] |> Array.choose id)
    static member Expand(input: Tensor<uint8>, shape: Tensor<uint8>) =
        buildAndRunBinary "Expand" input shape ([||] |> Array.choose id)
    static member Expand(input: Tensor<int8>, shape: Tensor<int8>) =
        buildAndRunBinary "Expand" input shape ([||] |> Array.choose id)
    static member Expand(input: Tensor<int>, shape: Tensor<int>) =
        buildAndRunBinary "Expand" input shape ([||] |> Array.choose id)
    static member Expand(input: Tensor<int64>, shape: Tensor<int64>) =
        buildAndRunBinary "Expand" input shape ([||] |> Array.choose id)
    static member Expand(input: Tensor<float32>, shape: Tensor<float32>) =
        buildAndRunBinary "Expand" input shape ([||] |> Array.choose id)
    static member Expand(input: Tensor<string>, shape: Tensor<string>) =
        buildAndRunBinary "Expand" input shape ([||] |> Array.choose id)
    static member Expand(input: Tensor<bool>, shape: Tensor<bool>) =
        buildAndRunBinary "Expand" input shape ([||] |> Array.choose id)
    static member MatMul(A: Tensor<float32>, B: Tensor<float32>) =
        buildAndRunBinary "MatMul" A B ([||] |> Array.choose id)
    static member MatMul(A: Tensor<int>, B: Tensor<int>) =
        buildAndRunBinary "MatMul" A B ([||] |> Array.choose id)
    static member MatMul(A: Tensor<int64>, B: Tensor<int64>) =
        buildAndRunBinary "MatMul" A B ([||] |> Array.choose id)
    static member BitShift(X: Tensor<uint8>, Y: Tensor<uint8>, direction: string) =
        buildAndRunBinary "BitShift" X Y ([|Attr.string("direction", direction)|] |> Array.choose id)
    static member Pow(X: Tensor<float32>, Y: Tensor<float32>) =
        buildAndRunBinary "Pow" X Y ([||] |> Array.choose id)
    static member Mod(A: Tensor<uint8>, B: Tensor<uint8>, ?fmod: int64) =
        buildAndRunBinary "Mod" A B ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<int8>, B: Tensor<int8>, ?fmod: int64) =
        buildAndRunBinary "Mod" A B ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<int>, B: Tensor<int>, ?fmod: int64) =
        buildAndRunBinary "Mod" A B ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<int64>, B: Tensor<int64>, ?fmod: int64) =
        buildAndRunBinary "Mod" A B ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<float32>, B: Tensor<float32>, ?fmod: int64) =
        buildAndRunBinary "Mod" A B ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<uint8>, indices: Tensor<uint8>) =
        buildAndRunBinary "GatherND" data indices ([||] |> Array.choose id)
    static member GatherND(data: Tensor<int8>, indices: Tensor<int8>) =
        buildAndRunBinary "GatherND" data indices ([||] |> Array.choose id)
    static member GatherND(data: Tensor<int>, indices: Tensor<int>) =
        buildAndRunBinary "GatherND" data indices ([||] |> Array.choose id)
    static member GatherND(data: Tensor<int64>, indices: Tensor<int64>) =
        buildAndRunBinary "GatherND" data indices ([||] |> Array.choose id)
    static member GatherND(data: Tensor<float32>, indices: Tensor<float32>) =
        buildAndRunBinary "GatherND" data indices ([||] |> Array.choose id)
    static member GatherND(data: Tensor<string>, indices: Tensor<string>) =
        buildAndRunBinary "GatherND" data indices ([||] |> Array.choose id)
    static member GatherND(data: Tensor<bool>, indices: Tensor<bool>) =
        buildAndRunBinary "GatherND" data indices ([||] |> Array.choose id)
    static member Div(A: Tensor<int>, B: Tensor<int>) =
        buildAndRunBinary "Div" A B ([||] |> Array.choose id)
    static member Div(A: Tensor<int64>, B: Tensor<int64>) =
        buildAndRunBinary "Div" A B ([||] |> Array.choose id)
    static member Div(A: Tensor<float32>, B: Tensor<float32>) =
        buildAndRunBinary "Div" A B ([||] |> Array.choose id)
    static member MaxRoiPool(X: Tensor<float32>, rois: Tensor<float32>, pooled_shape: int64[], ?spatial_scale: float32) =
        buildAndRunBinary "MaxRoiPool" X rois ([|Attr.ints("pooled_shape", pooled_shape); Attr.float("spatial_scale", spatial_scale, 1.0f)|] |> Array.choose id)
    static member Add(A: Tensor<int>, B: Tensor<int>) =
        buildAndRunBinary "Add" A B ([||] |> Array.choose id)
    static member Add(A: Tensor<int64>, B: Tensor<int64>) =
        buildAndRunBinary "Add" A B ([||] |> Array.choose id)
    static member Add(A: Tensor<float32>, B: Tensor<float32>) =
        buildAndRunBinary "Add" A B ([||] |> Array.choose id)
    static member ReverseSequence(input: Tensor<uint8>, sequence_lens: Tensor<uint8>, ?batch_axis: int64, ?time_axis: int64) =
        buildAndRunBinary "ReverseSequence" input sequence_lens ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<int8>, sequence_lens: Tensor<int8>, ?batch_axis: int64, ?time_axis: int64) =
        buildAndRunBinary "ReverseSequence" input sequence_lens ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<int>, sequence_lens: Tensor<int>, ?batch_axis: int64, ?time_axis: int64) =
        buildAndRunBinary "ReverseSequence" input sequence_lens ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<int64>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        buildAndRunBinary "ReverseSequence" input sequence_lens ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<float32>, sequence_lens: Tensor<float32>, ?batch_axis: int64, ?time_axis: int64) =
        buildAndRunBinary "ReverseSequence" input sequence_lens ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<string>, sequence_lens: Tensor<string>, ?batch_axis: int64, ?time_axis: int64) =
        buildAndRunBinary "ReverseSequence" input sequence_lens ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<bool>, sequence_lens: Tensor<bool>, ?batch_axis: int64, ?time_axis: int64) =
        buildAndRunBinary "ReverseSequence" input sequence_lens ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member Reshape(data: Tensor<uint8>, shape: Tensor<uint8>) =
        buildAndRunBinary "Reshape" data shape ([||] |> Array.choose id)
    static member Reshape(data: Tensor<int8>, shape: Tensor<int8>) =
        buildAndRunBinary "Reshape" data shape ([||] |> Array.choose id)
    static member Reshape(data: Tensor<int>, shape: Tensor<int>) =
        buildAndRunBinary "Reshape" data shape ([||] |> Array.choose id)
    static member Reshape(data: Tensor<int64>, shape: Tensor<int64>) =
        buildAndRunBinary "Reshape" data shape ([||] |> Array.choose id)
    static member Reshape(data: Tensor<float32>, shape: Tensor<float32>) =
        buildAndRunBinary "Reshape" data shape ([||] |> Array.choose id)
    static member Reshape(data: Tensor<string>, shape: Tensor<string>) =
        buildAndRunBinary "Reshape" data shape ([||] |> Array.choose id)
    static member Reshape(data: Tensor<bool>, shape: Tensor<bool>) =
        buildAndRunBinary "Reshape" data shape ([||] |> Array.choose id)
    static member Mul(A: Tensor<int>, B: Tensor<int>) =
        buildAndRunBinary "Mul" A B ([||] |> Array.choose id)
    static member Mul(A: Tensor<int64>, B: Tensor<int64>) =
        buildAndRunBinary "Mul" A B ([||] |> Array.choose id)
    static member Mul(A: Tensor<float32>, B: Tensor<float32>) =
        buildAndRunBinary "Mul" A B ([||] |> Array.choose id)
    static member PRelu(X: Tensor<float32>, slope: Tensor<float32>) =
        buildAndRunBinary "PRelu" X slope ([||] |> Array.choose id)
    static member PRelu(X: Tensor<int>, slope: Tensor<int>) =
        buildAndRunBinary "PRelu" X slope ([||] |> Array.choose id)
    static member PRelu(X: Tensor<int64>, slope: Tensor<int64>) =
        buildAndRunBinary "PRelu" X slope ([||] |> Array.choose id)
    static member Sub(A: Tensor<int>, B: Tensor<int>) =
        buildAndRunBinary "Sub" A B ([||] |> Array.choose id)
    static member Sub(A: Tensor<int64>, B: Tensor<int64>) =
        buildAndRunBinary "Sub" A B ([||] |> Array.choose id)
    static member Sub(A: Tensor<float32>, B: Tensor<float32>) =
        buildAndRunBinary "Sub" A B ([||] |> Array.choose id)
    static member Upsample(X: Tensor<uint8>, scales: Tensor<uint8>, ?mode: string) =
        buildAndRunBinary "Upsample" X scales ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<int8>, scales: Tensor<int8>, ?mode: string) =
        buildAndRunBinary "Upsample" X scales ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<int>, scales: Tensor<int>, ?mode: string) =
        buildAndRunBinary "Upsample" X scales ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<int64>, scales: Tensor<int64>, ?mode: string) =
        buildAndRunBinary "Upsample" X scales ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<float32>, scales: Tensor<float32>, ?mode: string) =
        buildAndRunBinary "Upsample" X scales ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<string>, scales: Tensor<string>, ?mode: string) =
        buildAndRunBinary "Upsample" X scales ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<bool>, scales: Tensor<bool>, ?mode: string) =
        execNode "Upsample" [|X;scales|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
