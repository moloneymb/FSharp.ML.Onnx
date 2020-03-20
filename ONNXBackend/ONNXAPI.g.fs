module ONNXAPI

open System
open System.IO
open System.Text
open Onnx
open Google.Protobuf.Collections
open Microsoft.ML.OnnxRuntime.Tensors
open Microsoft.ML.OnnxRuntime
open ProtoBuf

type ONNX() =
    static member LinearRegressor(X: Tensor<float32>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        MV() |> fun mv -> execNode<float32> "LinearRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member LinearRegressor(X: Tensor<int64>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        MV() |> fun mv -> execNode<int64> "LinearRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member LinearRegressor(X: Tensor<int>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        MV() |> fun mv -> execNode<int> "LinearRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member Imputer(X: Tensor<float32>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        MV() |> fun mv -> execNode<float32> "Imputer" [|mv.c(X)|] ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member Imputer(X: Tensor<int64>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        MV() |> fun mv -> execNode<int64> "Imputer" [|mv.c(X)|] ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member Imputer(X: Tensor<int>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        MV() |> fun mv -> execNode<int> "Imputer" [|mv.c(X)|] ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member FeatureVectorizer([<ParamArray>]X: Tensor<int>[], ?inputdimensions: int64[]) =
        MV() |> fun mv -> execNode<int> "FeatureVectorizer" (mv.c(X)) ([|Attr.ints("inputdimensions", inputdimensions)|] |> Array.choose id)
    static member FeatureVectorizer([<ParamArray>]X: Tensor<int64>[], ?inputdimensions: int64[]) =
        MV() |> fun mv -> execNode<int64> "FeatureVectorizer" (mv.c(X)) ([|Attr.ints("inputdimensions", inputdimensions)|] |> Array.choose id)
    static member FeatureVectorizer([<ParamArray>]X: Tensor<float32>[], ?inputdimensions: int64[]) =
        MV() |> fun mv -> execNode<float32> "FeatureVectorizer" (mv.c(X)) ([|Attr.ints("inputdimensions", inputdimensions)|] |> Array.choose id)
    static member Binarizer(X: Tensor<float32>, ?threshold: float32) =
        MV() |> fun mv -> execNode<float32> "Binarizer" [|mv.c(X)|] ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member Binarizer(X: Tensor<int64>, ?threshold: float32) =
        MV() |> fun mv -> execNode<int64> "Binarizer" [|mv.c(X)|] ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member Binarizer(X: Tensor<int>, ?threshold: float32) =
        MV() |> fun mv -> execNode<int> "Binarizer" [|mv.c(X)|] ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member ArrayFeatureExtractor(X: Tensor<float32>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member ArrayFeatureExtractor(X: Tensor<int64>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member ArrayFeatureExtractor(X: Tensor<int>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member ArrayFeatureExtractor(X: Tensor<string>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member SVMRegressor(X: Tensor<float32>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        MV() |> fun mv -> execNode<float32> "SVMRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member SVMRegressor(X: Tensor<int64>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        MV() |> fun mv -> execNode<int64> "SVMRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member SVMRegressor(X: Tensor<int>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        MV() |> fun mv -> execNode<int> "SVMRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member Det(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Det" [|mv.c(X)|] [||]
    static member TreeEnsembleRegressor(X: Tensor<float32>, ?aggregate_function: string, ?base_values: float32[], ?n_targets: int64, ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string, ?target_ids: int64[], ?target_nodeids: int64[], ?target_treeids: int64[], ?target_weights: float32[]) =
        MV() |> fun mv -> execNode<float32> "TreeEnsembleRegressor" [|mv.c(X)|] ([|Attr.string("aggregate_function", aggregate_function, "SUM"); Attr.floats("base_values", base_values); Attr.int("n_targets", n_targets); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE"); Attr.ints("target_ids", target_ids); Attr.ints("target_nodeids", target_nodeids); Attr.ints("target_treeids", target_treeids); Attr.floats("target_weights", target_weights)|] |> Array.choose id)
    static member TreeEnsembleRegressor(X: Tensor<int64>, ?aggregate_function: string, ?base_values: float32[], ?n_targets: int64, ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string, ?target_ids: int64[], ?target_nodeids: int64[], ?target_treeids: int64[], ?target_weights: float32[]) =
        MV() |> fun mv -> execNode<int64> "TreeEnsembleRegressor" [|mv.c(X)|] ([|Attr.string("aggregate_function", aggregate_function, "SUM"); Attr.floats("base_values", base_values); Attr.int("n_targets", n_targets); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE"); Attr.ints("target_ids", target_ids); Attr.ints("target_nodeids", target_nodeids); Attr.ints("target_treeids", target_treeids); Attr.floats("target_weights", target_weights)|] |> Array.choose id)
    static member TreeEnsembleRegressor(X: Tensor<int>, ?aggregate_function: string, ?base_values: float32[], ?n_targets: int64, ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string, ?target_ids: int64[], ?target_nodeids: int64[], ?target_treeids: int64[], ?target_weights: float32[]) =
        MV() |> fun mv -> execNode<int> "TreeEnsembleRegressor" [|mv.c(X)|] ([|Attr.string("aggregate_function", aggregate_function, "SUM"); Attr.floats("base_values", base_values); Attr.int("n_targets", n_targets); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE"); Attr.ints("target_ids", target_ids); Attr.ints("target_nodeids", target_nodeids); Attr.ints("target_treeids", target_treeids); Attr.floats("target_weights", target_weights)|] |> Array.choose id)
    static member Round(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Round" [|mv.c(X)|] [||]
    static member Range(start: Tensor<float32>, limit: Tensor<float32>, delta: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Range" [|mv.c(start); mv.c(limit); mv.c(delta)|] [||]
    static member Range(start: Tensor<int>, limit: Tensor<int>, delta: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Range" [|mv.c(start); mv.c(limit); mv.c(delta)|] [||]
    static member Range(start: Tensor<int64>, limit: Tensor<int64>, delta: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Range" [|mv.c(start); mv.c(limit); mv.c(delta)|] [||]
    static member ThresholdedRelu(X: Tensor<float32>, ?alpha: float32) =
        MV() |> fun mv -> execNode<float32> "ThresholdedRelu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.0f)|] |> Array.choose id)
    static member MeanVarianceNormalization(X: Tensor<float32>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<float32> "MeanVarianceNormalization" [|mv.c(X)|] ([|Attr.ints("axes", axes, [|0L;2L;3L|])|] |> Array.choose id)
    static member NonZero(X: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<string>) =
        MV() |> fun mv -> execNode<string> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "NonZero" [|mv.c(X)|] [||]
    static member Shrink(input: Tensor<uint8>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<uint8> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<int8>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<int8> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<int>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<int> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<int64>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<int64> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<float32>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<float32> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Erf(input: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Erf" [|mv.c(input)|] [||]
    static member Erf(input: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Erf" [|mv.c(input)|] [||]
    static member Erf(input: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Erf" [|mv.c(input)|] [||]
    static member Erf(input: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Erf" [|mv.c(input)|] [||]
    static member Erf(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Erf" [|mv.c(input)|] [||]
    static member Atanh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Atanh" [|mv.c(input)|] [||]
    static member Acosh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Acosh" [|mv.c(input)|] [||]
    static member Expand(input: Tensor<uint8>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<int8>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<int>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<int64>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<float32>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<string>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<bool>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Atan(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Atan" [|mv.c(input)|] [||]
    static member Asin(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Asin" [|mv.c(input)|] [||]
    static member LpNormalization(input: Tensor<float32>, ?axis: int64, ?p: int64) =
        MV() |> fun mv -> execNode<float32> "LpNormalization" [|mv.c(input)|] ([|Attr.int("axis", axis, -1L); Attr.int("p", p, 2L)|] |> Array.choose id)
    static member Ceil(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Ceil" [|mv.c(X)|] [||]
    static member LogSoftmax(input: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "LogSoftmax" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member MatMul(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member MatMul(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member MatMul(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member BitShift(X: Tensor<uint8>, Y: Tensor<uint8>, direction: string) =
        MV() |> fun mv -> execNode<uint8> "BitShift" [|mv.c(X); mv.c(Y)|] ([|Attr.string("direction", direction)|] |> Array.choose id)
    static member Sinh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sinh" [|mv.c(input)|] [||]
    static member Acos(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Acos" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<string>) =
        MV() |> fun mv -> execNode<string> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Identity" [|mv.c(input)|] [||]
    static member Pow(X: Tensor<float32>, Y: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Mod(A: Tensor<uint8>, B: Tensor<uint8>, ?fmod: int64) =
        MV() |> fun mv -> execNode<uint8> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<int8>, B: Tensor<int8>, ?fmod: int64) =
        MV() |> fun mv -> execNode<int8> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<int>, B: Tensor<int>, ?fmod: int64) =
        MV() |> fun mv -> execNode<int> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<int64>, B: Tensor<int64>, ?fmod: int64) =
        MV() |> fun mv -> execNode<int64> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<float32>, B: Tensor<float32>, ?fmod: int64) =
        MV() |> fun mv -> execNode<float32> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Softplus(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Softplus" [|mv.c(X)|] [||]
    static member Normalizer(X: Tensor<float32>, ?norm: string) =
        MV() |> fun mv -> execNode<float32> "Normalizer" [|mv.c(X)|] ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member Normalizer(X: Tensor<int64>, ?norm: string) =
        MV() |> fun mv -> execNode<int64> "Normalizer" [|mv.c(X)|] ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member Normalizer(X: Tensor<int>, ?norm: string) =
        MV() |> fun mv -> execNode<int> "Normalizer" [|mv.c(X)|] ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member Hardmax(input: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Hardmax" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member HardSigmoid(X: Tensor<float32>, ?alpha: float32, ?beta: float32) =
        MV() |> fun mv -> execNode<float32> "HardSigmoid" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 0.20000000298023224f); Attr.float("beta", beta, 0.5f)|] |> Array.choose id)
    static member LpPool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?p: int64, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "LpPool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("p", p, 2L); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Min([<ParamArray>]data_0: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Min" (mv.c(data_0)) [||]
    static member Sum([<ParamArray>]data_0: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Sum" (mv.c(data_0)) [||]
    static member Transpose(data: Tensor<uint8>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<uint8> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<int8>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<int8> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<int>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<int> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<int64>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<int64> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<float32>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<float32> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<string>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<string> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<bool>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<bool> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member ScatterND(data: Tensor<uint8>, indices: Tensor<int64>, updates: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member ScatterND(data: Tensor<int8>, indices: Tensor<int64>, updates: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member ScatterND(data: Tensor<int>, indices: Tensor<int64>, updates: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member ScatterND(data: Tensor<int64>, indices: Tensor<int64>, updates: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member ScatterND(data: Tensor<float32>, indices: Tensor<int64>, updates: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member ScatterND(data: Tensor<string>, indices: Tensor<int64>, updates: Tensor<string>) =
        MV() |> fun mv -> execNode<string> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member ScatterND(data: Tensor<bool>, indices: Tensor<int64>, updates: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member GlobalLpPool(X: Tensor<float32>, ?p: int64) =
        MV() |> fun mv -> execNode<float32> "GlobalLpPool" [|mv.c(X)|] ([|Attr.int("p", p, 2L)|] |> Array.choose id)
    static member Gemm(A: Tensor<float32>, B: Tensor<float32>, ?C: Tensor<float32>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<float32> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member Gemm(A: Tensor<int>, B: Tensor<int>, ?C: Tensor<int>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<int> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member Gemm(A: Tensor<int64>, B: Tensor<int64>, ?C: Tensor<int64>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<int64> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member InstanceNormalization(input: Tensor<float32>, scale: Tensor<float32>, B: Tensor<float32>, ?epsilon: float32) =
        MV() |> fun mv -> execNode<float32> "InstanceNormalization" [|mv.c(input); mv.c(scale); mv.c(B)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f)|] |> Array.choose id)
    static member AveragePool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?count_include_pad: int64, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "AveragePool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("count_include_pad", count_include_pad, 0L); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Sign(input: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Sign" [|mv.c(input)|] [||]
    static member Sign(input: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Sign" [|mv.c(input)|] [||]
    static member Sign(input: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Sign" [|mv.c(input)|] [||]
    static member Sign(input: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Sign" [|mv.c(input)|] [||]
    static member Sign(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sign" [|mv.c(input)|] [||]
    static member Clip(input: Tensor<float32>, ?min: Tensor<float32>, ?max: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Clip" ([|Some(mv.c(input)); mv.c(min); mv.c(max)|] |> Array.choose id) [||]
    static member DequantizeLinear(x: Tensor<int8>, x_scale: Tensor<float32>, ?x_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "DequantizeLinear" ([|Some(mv.c(x)); Some(mv.c(x_scale)); mv.c(x_zero_point)|] |> Array.choose id) [||]
    static member DequantizeLinear(x: Tensor<uint8>, x_scale: Tensor<float32>, ?x_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "DequantizeLinear" ([|Some(mv.c(x)); Some(mv.c(x_scale)); mv.c(x_zero_point)|] |> Array.choose id) [||]
    static member DequantizeLinear(x: Tensor<int>, x_scale: Tensor<float32>, ?x_zero_point: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "DequantizeLinear" ([|Some(mv.c(x)); Some(mv.c(x_scale)); mv.c(x_zero_point)|] |> Array.choose id) [||]
    static member LRN(X: Tensor<float32>, size: int64, ?alpha: float32, ?beta: float32, ?bias: float32) =
        MV() |> fun mv -> execNode<float32> "LRN" [|mv.c(X)|] ([|Attr.int("size", size); Attr.float("alpha", alpha, 9.999999747378752e-05f); Attr.float("beta", beta, 0.75f); Attr.float("bias", bias, 1.0f)|] |> Array.choose id)
    static member Elu(X: Tensor<float32>, ?alpha: float32) =
        MV() |> fun mv -> execNode<float32> "Elu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.0f)|] |> Array.choose id)
    static member Sin(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sin" [|mv.c(input)|] [||]
    static member Pad(data: Tensor<uint8>, pads: Tensor<int64>, ?constant_value: Tensor<uint8>, ?mode: string) =
        MV() |> fun mv -> execNode<uint8> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<int8>, pads: Tensor<int64>, ?constant_value: Tensor<int8>, ?mode: string) =
        MV() |> fun mv -> execNode<int8> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<int>, pads: Tensor<int64>, ?constant_value: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<int> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<int64>, pads: Tensor<int64>, ?constant_value: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<int64> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<float32>, pads: Tensor<int64>, ?constant_value: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<float32> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member GatherND(data: Tensor<uint8>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member GatherND(data: Tensor<int8>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member GatherND(data: Tensor<int>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member GatherND(data: Tensor<int64>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member GatherND(data: Tensor<float32>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member GatherND(data: Tensor<string>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member GatherND(data: Tensor<bool>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member Relu(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Relu" [|mv.c(X)|] [||]
    static member Conv(X: Tensor<float32>, W: Tensor<float32>, ?B: Tensor<float32>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "Conv" ([|Some(mv.c(X)); Some(mv.c(W)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member ArgMax(data: Tensor<uint8>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint8> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<int8>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int8> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<int>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<int64>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<float32>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Div(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member Div(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member Div(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member MaxRoiPool(X: Tensor<float32>, rois: Tensor<float32>, pooled_shape: int64[], ?spatial_scale: float32) =
        MV() |> fun mv -> execNode<float32> "MaxRoiPool" [|mv.c(X); mv.c(rois)|] ([|Attr.ints("pooled_shape", pooled_shape); Attr.float("spatial_scale", spatial_scale, 1.0f)|] |> Array.choose id)
    static member Add(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member Add(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member Add(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member LeakyRelu(X: Tensor<float32>, ?alpha: float32) =
        MV() |> fun mv -> execNode<float32> "LeakyRelu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 0.009999999776482582f)|] |> Array.choose id)
    static member ReduceLogSum(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceLogSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceLogSum(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceLogSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceLogSum(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceLogSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Floor(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Floor" [|mv.c(X)|] [||]
    static member ArgMin(data: Tensor<uint8>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint8> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<int8>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int8> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<int>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<int64>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<float32>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<uint8>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<uint8> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<int8>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<int8> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<int>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<int> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<int64>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<int64> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<float32>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<float32> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<string>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<string> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<bool>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<bool> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member Tan(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Tan" [|mv.c(input)|] [||]
    static member ReduceSum(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceSum(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceSum(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<uint8>[]) =
        MV() |> fun mv -> execNode<uint8> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<int8>[]) =
        MV() |> fun mv -> execNode<int8> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<int>[]) =
        MV() |> fun mv -> execNode<int> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<int64>[]) =
        MV() |> fun mv -> execNode<int64> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<string>[]) =
        MV() |> fun mv -> execNode<string> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<bool>[]) =
        MV() |> fun mv -> execNode<bool> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member OneHotEncoder(X: Tensor<string>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<string> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member OneHotEncoder(X: Tensor<int64>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<int64> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member OneHotEncoder(X: Tensor<int>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<int> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member OneHotEncoder(X: Tensor<float32>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<float32> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member ConvTranspose(X: Tensor<float32>, W: Tensor<float32>, ?B: Tensor<float32>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?output_padding: int64[], ?output_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "ConvTranspose" ([|Some(mv.c(X)); Some(mv.c(W)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("output_padding", output_padding); Attr.ints("output_shape", output_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<uint8>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<uint8> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<int8>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<int8> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<int>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<int> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<int64>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<int64> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<float32>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<float32> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<string>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<string> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<bool>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<bool> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member Max([<ParamArray>]data_0: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Max" (mv.c(data_0)) [||]
    static member GlobalMaxPool(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "GlobalMaxPool" [|mv.c(X)|] [||]
    static member Exp(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Exp" [|mv.c(input)|] [||]
    static member Reshape(data: Tensor<uint8>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member Reshape(data: Tensor<int8>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member Reshape(data: Tensor<int>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member Reshape(data: Tensor<int64>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member Reshape(data: Tensor<float32>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member Reshape(data: Tensor<string>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member Reshape(data: Tensor<bool>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member GlobalAveragePool(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "GlobalAveragePool" [|mv.c(X)|] [||]
    static member Mean([<ParamArray>]data_0: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Mean" (mv.c(data_0)) [||]
    static member Mul(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member Mul(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member Mul(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member Neg(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Neg" [|mv.c(X)|] [||]
    static member Neg(X: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Neg" [|mv.c(X)|] [||]
    static member Neg(X: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Neg" [|mv.c(X)|] [||]
    static member Neg(X: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Neg" [|mv.c(X)|] [||]
    static member Not(X: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Not" [|mv.c(X)|] [||]
    static member ReduceL1(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceL1" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceL1(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceL1" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceL1(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceL1" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member PRelu(X: Tensor<float32>, slope: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member PRelu(X: Tensor<int>, slope: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member PRelu(X: Tensor<int64>, slope: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member Unsqueeze(data: Tensor<uint8>, axes: int64[]) =
        MV() |> fun mv -> execNode<uint8> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<int8>, axes: int64[]) =
        MV() |> fun mv -> execNode<int8> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<int>, axes: int64[]) =
        MV() |> fun mv -> execNode<int> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<int64>, axes: int64[]) =
        MV() |> fun mv -> execNode<int64> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<float32>, axes: int64[]) =
        MV() |> fun mv -> execNode<float32> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<string>, axes: int64[]) =
        MV() |> fun mv -> execNode<string> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<bool>, axes: int64[]) =
        MV() |> fun mv -> execNode<bool> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Tanh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Tanh" [|mv.c(input)|] [||]
    static member Abs(X: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Abs" [|mv.c(X)|] [||]
    static member Abs(X: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Abs" [|mv.c(X)|] [||]
    static member Abs(X: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Abs" [|mv.c(X)|] [||]
    static member Abs(X: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Abs" [|mv.c(X)|] [||]
    static member Abs(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Abs" [|mv.c(X)|] [||]
    static member Reciprocal(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Reciprocal" [|mv.c(X)|] [||]
    static member ReduceLogSumExp(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceLogSumExp" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceLogSumExp(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceLogSumExp" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceLogSumExp(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceLogSumExp" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceMax" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceMax" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceMax" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMean(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceMean" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMean(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceMean" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMean(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceMean" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Cosh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Cosh" [|mv.c(input)|] [||]
    static member ReduceMin(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceMin" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMin(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceMin" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceMin(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceMin" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceProd(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceProd" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceProd(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceProd" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceProd(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceProd" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Squeeze(data: Tensor<uint8>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<uint8> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Squeeze(data: Tensor<int8>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int8> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Squeeze(data: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Squeeze(data: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int64> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Squeeze(data: Tensor<float32>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<float32> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Squeeze(data: Tensor<string>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<string> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Squeeze(data: Tensor<bool>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<bool> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Selu(X: Tensor<float32>, ?alpha: float32, ?gamma: float32) =
        MV() |> fun mv -> execNode<float32> "Selu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.6732631921768188f); Attr.float("gamma", gamma, 1.0507010221481323f)|] |> Array.choose id)
    static member Sigmoid(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sigmoid" [|mv.c(X)|] [||]
    static member ReduceSumSquare(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceSumSquare" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceSumSquare(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceSumSquare" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceSumSquare(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceSumSquare" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Softmax(input: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Softmax" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Softsign(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Softsign" [|mv.c(input)|] [||]
    static member Cos(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Cos" [|mv.c(input)|] [||]
    static member SpaceToDepth(input: Tensor<uint8>, blocksize: int64) =
        MV() |> fun mv -> execNode<uint8> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<int8>, blocksize: int64) =
        MV() |> fun mv -> execNode<int8> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<int>, blocksize: int64) =
        MV() |> fun mv -> execNode<int> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<int64>, blocksize: int64) =
        MV() |> fun mv -> execNode<int64> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<float32>, blocksize: int64) =
        MV() |> fun mv -> execNode<float32> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<string>, blocksize: int64) =
        MV() |> fun mv -> execNode<string> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<bool>, blocksize: int64) =
        MV() |> fun mv -> execNode<bool> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member Asinh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Asinh" [|mv.c(input)|] [||]
    static member ReduceL2(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceL2" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceL2(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceL2" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member ReduceL2(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceL2" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member Sqrt(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sqrt" [|mv.c(X)|] [||]
    static member Log(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Log" [|mv.c(input)|] [||]
    static member Sub(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member Sub(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member Sub(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member Scaler(X: Tensor<float32>, ?offset: float32[], ?scale: float32[]) =
        MV() |> fun mv -> execNode<float32> "Scaler" [|mv.c(X)|] ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member Scaler(X: Tensor<int64>, ?offset: float32[], ?scale: float32[]) =
        MV() |> fun mv -> execNode<int64> "Scaler" [|mv.c(X)|] ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member Scaler(X: Tensor<int>, ?offset: float32[], ?scale: float32[]) =
        MV() |> fun mv -> execNode<int> "Scaler" [|mv.c(X)|] ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member Upsample(X: Tensor<uint8>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<uint8> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<int8>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<int8> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<int>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<int> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<int64>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<int64> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<float32>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<float32> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<string>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<string> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<bool>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<bool> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member IsInf(X: Tensor<float32>, ?detect_negative: int64, ?detect_positive: int64) =
        MV() |> fun mv -> execNode<bool> "IsInf" [|mv.c(X)|] ([|Attr.int("detect_negative", detect_negative, 1L); Attr.int("detect_positive", detect_positive, 1L)|] |> Array.choose id)
    static member TfIdfVectorizer(X: Tensor<string>, max_gram_length: int64, max_skip_count: int64, min_gram_length: int64, mode: string, ngram_counts: int64[], ngram_indexes: int64[], ?pool_int64s: int64[], ?pool_strings: string[], ?weights: float32[]) =
        MV() |> fun mv -> execNode<float32> "TfIdfVectorizer" [|mv.c(X)|] ([|Attr.int("max_gram_length", max_gram_length); Attr.int("max_skip_count", max_skip_count); Attr.int("min_gram_length", min_gram_length); Attr.string("mode", mode); Attr.ints("ngram_counts", ngram_counts); Attr.ints("ngram_indexes", ngram_indexes); Attr.ints("pool_int64s", pool_int64s); Attr.strings("pool_strings", pool_strings); Attr.floats("weights", weights)|] |> Array.choose id)
    static member TfIdfVectorizer(X: Tensor<int>, max_gram_length: int64, max_skip_count: int64, min_gram_length: int64, mode: string, ngram_counts: int64[], ngram_indexes: int64[], ?pool_int64s: int64[], ?pool_strings: string[], ?weights: float32[]) =
        MV() |> fun mv -> execNode<float32> "TfIdfVectorizer" [|mv.c(X)|] ([|Attr.int("max_gram_length", max_gram_length); Attr.int("max_skip_count", max_skip_count); Attr.int("min_gram_length", min_gram_length); Attr.string("mode", mode); Attr.ints("ngram_counts", ngram_counts); Attr.ints("ngram_indexes", ngram_indexes); Attr.ints("pool_int64s", pool_int64s); Attr.strings("pool_strings", pool_strings); Attr.floats("weights", weights)|] |> Array.choose id)
    static member TfIdfVectorizer(X: Tensor<int64>, max_gram_length: int64, max_skip_count: int64, min_gram_length: int64, mode: string, ngram_counts: int64[], ngram_indexes: int64[], ?pool_int64s: int64[], ?pool_strings: string[], ?weights: float32[]) =
        MV() |> fun mv -> execNode<float32> "TfIdfVectorizer" [|mv.c(X)|] ([|Attr.int("max_gram_length", max_gram_length); Attr.int("max_skip_count", max_skip_count); Attr.int("min_gram_length", min_gram_length); Attr.string("mode", mode); Attr.ints("ngram_counts", ngram_counts); Attr.ints("ngram_indexes", ngram_indexes); Attr.ints("pool_int64s", pool_int64s); Attr.strings("pool_strings", pool_strings); Attr.floats("weights", weights)|] |> Array.choose id)
    static member Shape(data: Tensor<uint8>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member Shape(data: Tensor<int8>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member Shape(data: Tensor<int>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member Shape(data: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member Shape(data: Tensor<float32>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member Shape(data: Tensor<string>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member Shape(data: Tensor<bool>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member Greater(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<bool>, B: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member And(A: Tensor<bool>, B: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "And" [|mv.c(A); mv.c(B)|] [||]
    static member Size(data: Tensor<uint8>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<int8>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<int>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<float32>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<string>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<bool>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member IsNaN(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "IsNaN" [|mv.c(X)|] [||]
    static member Less(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Xor(A: Tensor<bool>, B: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Xor" [|mv.c(A); mv.c(B)|] [||]
    static member Or(A: Tensor<bool>, B: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Or" [|mv.c(A); mv.c(B)|] [||]
    static member CumSum(x: Tensor<int>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<int> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member CumSum(x: Tensor<int>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<int> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member CumSum(x: Tensor<int64>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<int64> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member CumSum(x: Tensor<int64>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<int64> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member CumSum(x: Tensor<float32>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<float32> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member CumSum(x: Tensor<float32>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<float32> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member RoiAlign(X: Tensor<float32>, rois: Tensor<float32>, batch_indices: Tensor<int64>, ?mode: string, ?output_height: int64, ?output_width: int64, ?sampling_ratio: int64, ?spatial_scale: float32) =
        MV() |> fun mv -> execNode<float32> "RoiAlign" [|mv.c(X); mv.c(rois); mv.c(batch_indices)|] ([|Attr.string("mode", mode, "avg"); Attr.int("output_height", output_height, 1L); Attr.int("output_width", output_width, 1L); Attr.int("sampling_ratio", sampling_ratio, 0L); Attr.float("spatial_scale", spatial_scale, 1.0f)|] |> Array.choose id)
    static member QLinearConv(x: Tensor<int8>, x_scale: Tensor<float32>, x_zero_point: Tensor<int8>, w: Tensor<int8>, w_scale: Tensor<float32>, w_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member QLinearConv(x: Tensor<int8>, x_scale: Tensor<float32>, x_zero_point: Tensor<int8>, w: Tensor<int8>, w_scale: Tensor<float32>, w_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<uint8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member QLinearConv(x: Tensor<int8>, x_scale: Tensor<float32>, x_zero_point: Tensor<int8>, w: Tensor<uint8>, w_scale: Tensor<float32>, w_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member QLinearConv(x: Tensor<int8>, x_scale: Tensor<float32>, x_zero_point: Tensor<int8>, w: Tensor<uint8>, w_scale: Tensor<float32>, w_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<uint8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member QLinearConv(x: Tensor<uint8>, x_scale: Tensor<float32>, x_zero_point: Tensor<uint8>, w: Tensor<int8>, w_scale: Tensor<float32>, w_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member QLinearConv(x: Tensor<uint8>, x_scale: Tensor<float32>, x_zero_point: Tensor<uint8>, w: Tensor<int8>, w_scale: Tensor<float32>, w_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<uint8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member QLinearConv(x: Tensor<uint8>, x_scale: Tensor<float32>, x_zero_point: Tensor<uint8>, w: Tensor<uint8>, w_scale: Tensor<float32>, w_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member QLinearConv(x: Tensor<uint8>, x_scale: Tensor<float32>, x_zero_point: Tensor<uint8>, w: Tensor<uint8>, w_scale: Tensor<float32>, w_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<uint8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member ConvInteger(x: Tensor<int8>, w: Tensor<int8>, ?x_zero_point: Tensor<int8>, ?w_zero_point: Tensor<int8>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int> "ConvInteger" ([|Some(mv.c(x)); Some(mv.c(w)); mv.c(x_zero_point); mv.c(w_zero_point)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member ConvInteger(x: Tensor<int8>, w: Tensor<uint8>, ?x_zero_point: Tensor<int8>, ?w_zero_point: Tensor<uint8>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int> "ConvInteger" ([|Some(mv.c(x)); Some(mv.c(w)); mv.c(x_zero_point); mv.c(w_zero_point)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member ConvInteger(x: Tensor<uint8>, w: Tensor<int8>, ?x_zero_point: Tensor<uint8>, ?w_zero_point: Tensor<int8>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int> "ConvInteger" ([|Some(mv.c(x)); Some(mv.c(w)); mv.c(x_zero_point); mv.c(w_zero_point)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member ConvInteger(x: Tensor<uint8>, w: Tensor<uint8>, ?x_zero_point: Tensor<uint8>, ?w_zero_point: Tensor<uint8>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int> "ConvInteger" ([|Some(mv.c(x)); Some(mv.c(w)); mv.c(x_zero_point); mv.c(w_zero_point)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member QLinearMatMul(a: Tensor<int8>, a_scale: Tensor<float32>, a_zero_point: Tensor<int8>, b: Tensor<int8>, b_scale: Tensor<float32>, b_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member QLinearMatMul(a: Tensor<int8>, a_scale: Tensor<float32>, a_zero_point: Tensor<int8>, b: Tensor<int8>, b_scale: Tensor<float32>, b_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member QLinearMatMul(a: Tensor<int8>, a_scale: Tensor<float32>, a_zero_point: Tensor<int8>, b: Tensor<uint8>, b_scale: Tensor<float32>, b_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member QLinearMatMul(a: Tensor<int8>, a_scale: Tensor<float32>, a_zero_point: Tensor<int8>, b: Tensor<uint8>, b_scale: Tensor<float32>, b_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member QLinearMatMul(a: Tensor<uint8>, a_scale: Tensor<float32>, a_zero_point: Tensor<uint8>, b: Tensor<int8>, b_scale: Tensor<float32>, b_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member QLinearMatMul(a: Tensor<uint8>, a_scale: Tensor<float32>, a_zero_point: Tensor<uint8>, b: Tensor<int8>, b_scale: Tensor<float32>, b_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member QLinearMatMul(a: Tensor<uint8>, a_scale: Tensor<float32>, a_zero_point: Tensor<uint8>, b: Tensor<uint8>, b_scale: Tensor<float32>, b_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member QLinearMatMul(a: Tensor<uint8>, a_scale: Tensor<float32>, a_zero_point: Tensor<uint8>, b: Tensor<uint8>, b_scale: Tensor<float32>, b_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<uint8>, Y: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<int8>, Y: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<int>, Y: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<int64>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<float32>, Y: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<string>, Y: Tensor<string>) =
        MV() |> fun mv -> execNode<string> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<bool>, Y: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<uint8>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<uint8>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<uint8>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<uint8>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<uint8>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<uint8>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<uint8>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int8>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int8>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int8>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int8>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int8>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int8>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int8>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int64>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int64>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int64>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int64>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int64>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int64>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<int64>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<float32>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<float32>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<float32>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<float32>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<float32>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<float32>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<uint8>, depth: Tensor<float32>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<uint8>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<uint8>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<uint8>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<uint8>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<uint8>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<uint8>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<uint8>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int8>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int8>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int8>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int8>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int8>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int8>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int8>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int64>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int64>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int64>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int64>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int64>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int64>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<int64>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<float32>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<float32>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<float32>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<float32>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<float32>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<float32>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int8>, depth: Tensor<float32>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<uint8>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<uint8>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<uint8>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<uint8>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<uint8>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<uint8>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<uint8>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int8>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int8>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int8>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int8>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int8>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int8>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int8>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int64>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int64>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int64>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int64>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int64>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int64>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<int64>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<float32>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<float32>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<float32>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<float32>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<float32>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<float32>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int>, depth: Tensor<float32>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<uint8>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<uint8>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<uint8>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<uint8>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<uint8>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<uint8>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<uint8>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int8>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int8>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int8>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int8>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int8>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int8>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int8>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int64>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int64>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int64>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int64>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int64>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int64>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<int64>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<float32>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<float32>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<float32>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<float32>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<float32>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<float32>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<int64>, depth: Tensor<float32>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<uint8>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<uint8>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<uint8>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<uint8>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<uint8>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<uint8>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<uint8>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int8>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int8>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int8>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int8>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int8>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int8>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int8>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int64>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int64>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int64>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int64>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int64>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int64>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<int64>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<float32>, values: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<float32>, values: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<float32>, values: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<float32>, values: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<float32>, values: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<float32>, values: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member OneHot(indices: Tensor<float32>, depth: Tensor<float32>, values: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "OneHot" [|mv.c(indices); mv.c(depth); mv.c(values)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member MaxUnpool(X: Tensor<float32>, I: Tensor<int64>, kernel_shape: int64[], ?output_shape: Tensor<int64>, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "MaxUnpool" ([|Some(mv.c(X)); Some(mv.c(I)); mv.c(output_shape)|] |> Array.choose id) ([|Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member GatherElements(data: Tensor<uint8>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<uint8>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<int8>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<int8>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<int>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<int>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<int64>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<int64>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<float32>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<float32>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<string>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<string>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<bool>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<bool>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member QuantizeLinear(x: Tensor<float32>, y_scale: Tensor<float32>, ?y_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "QuantizeLinear" ([|Some(mv.c(x)); Some(mv.c(y_scale)); mv.c(y_zero_point)|] |> Array.choose id) [||]
    static member QuantizeLinear(x: Tensor<float32>, y_scale: Tensor<float32>, ?y_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "QuantizeLinear" ([|Some(mv.c(x)); Some(mv.c(y_scale)); mv.c(y_zero_point)|] |> Array.choose id) [||]
    static member QuantizeLinear(x: Tensor<int>, y_scale: Tensor<float32>, ?y_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "QuantizeLinear" ([|Some(mv.c(x)); Some(mv.c(y_scale)); mv.c(y_zero_point)|] |> Array.choose id) [||]
    static member QuantizeLinear(x: Tensor<int>, y_scale: Tensor<float32>, ?y_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "QuantizeLinear" ([|Some(mv.c(x)); Some(mv.c(y_scale)); mv.c(y_zero_point)|] |> Array.choose id) [||]
    static member Resize(X: Tensor<uint8>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint8> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<int8>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int8> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<int>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<int64>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int64> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<float32>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<float32> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<string>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<string> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<bool>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<bool> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member MatMulInteger(A: Tensor<int8>, B: Tensor<int8>, ?a_zero_point: Tensor<int8>, ?b_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int> "MatMulInteger" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(a_zero_point); mv.c(b_zero_point)|] |> Array.choose id) [||]
    static member MatMulInteger(A: Tensor<int8>, B: Tensor<uint8>, ?a_zero_point: Tensor<int8>, ?b_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<int> "MatMulInteger" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(a_zero_point); mv.c(b_zero_point)|] |> Array.choose id) [||]
    static member MatMulInteger(A: Tensor<uint8>, B: Tensor<int8>, ?a_zero_point: Tensor<uint8>, ?b_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int> "MatMulInteger" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(a_zero_point); mv.c(b_zero_point)|] |> Array.choose id) [||]
    static member MatMulInteger(A: Tensor<uint8>, B: Tensor<uint8>, ?a_zero_point: Tensor<uint8>, ?b_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<int> "MatMulInteger" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(a_zero_point); mv.c(b_zero_point)|] |> Array.choose id) [||]
    static member Compress(input: Tensor<uint8>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<int8>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<int>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<int64>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<float32>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<string>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<bool>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Gather(data: Tensor<uint8>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<uint8>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<int8>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<int8>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<int>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<int>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<int64>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<int64>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<float32>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<float32>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<string>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<string>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<bool>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<bool>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<uint8>, indices: Tensor<int>, updates: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<uint8>, indices: Tensor<int64>, updates: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int8>, indices: Tensor<int>, updates: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int8>, indices: Tensor<int64>, updates: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int>, indices: Tensor<int>, updates: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int>, indices: Tensor<int64>, updates: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int64>, indices: Tensor<int>, updates: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int64>, indices: Tensor<int64>, updates: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<float32>, indices: Tensor<int>, updates: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<float32>, indices: Tensor<int64>, updates: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<string>, indices: Tensor<int>, updates: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<string>, indices: Tensor<int64>, updates: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<bool>, indices: Tensor<int>, updates: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<bool>, indices: Tensor<int64>, updates: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Slice(data: Tensor<uint8>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<uint8> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<uint8>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<int8>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<int8> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<int8>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<int>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<int>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<int64>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<int64> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<int64>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<float32>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<float32> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<float32>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<string>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<string> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<string>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<bool>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<bool>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Tile(input: Tensor<uint8>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<int8>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<int>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<int64>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<float32>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<string>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<bool>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Scatter(data: Tensor<uint8>, indices: Tensor<int>, updates: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<uint8>, indices: Tensor<int64>, updates: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<int8>, indices: Tensor<int>, updates: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<int8>, indices: Tensor<int64>, updates: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<int>, indices: Tensor<int>, updates: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<int>, indices: Tensor<int64>, updates: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<int64>, indices: Tensor<int>, updates: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<int64>, indices: Tensor<int64>, updates: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<float32>, indices: Tensor<int>, updates: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<float32>, indices: Tensor<int64>, updates: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<string>, indices: Tensor<int>, updates: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<string>, indices: Tensor<int64>, updates: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<bool>, indices: Tensor<int>, updates: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<bool>, indices: Tensor<int64>, updates: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member NonMaxSuppression(boxes: Tensor<float32>, scores: Tensor<float32>, ?max_output_boxes_per_class: Tensor<int64>, ?iou_threshold: Tensor<float32>, ?score_threshold: Tensor<float32>, ?center_point_box: int64) =
        MV() |> fun mv -> execNode<int64> "NonMaxSuppression" ([|Some(mv.c(boxes)); Some(mv.c(scores)); mv.c(max_output_boxes_per_class); mv.c(iou_threshold); mv.c(score_threshold)|] |> Array.choose id) ([|Attr.int("center_point_box", center_point_box, 0L)|] |> Array.choose id)
    static member StringNormalizer(X: Tensor<string>, ?case_change_action: string, ?is_case_sensitive: int64, ?locale: string, ?stopwords: string[]) =
        MV() |> fun mv -> execNode<string> "StringNormalizer" [|mv.c(X)|] ([|Attr.string("case_change_action", case_change_action, "NONE"); Attr.int("is_case_sensitive", is_case_sensitive, 0L); Attr.string("locale", locale); Attr.strings("stopwords", stopwords)|] |> Array.choose id)
    static member LabelEncoder(X: Tensor<string>, ?default_float: float32, ?default_int64: int64, ?default_string: string, ?keys_floats: float32[], ?keys_int64s: int64[], ?keys_strings: string[], ?values_floats: float32[], ?values_int64s: int64[], ?values_strings: string[]) =
        MV() |> fun mv -> execNode<string> "LabelEncoder" [|mv.c(X)|] ([|Attr.float("default_float", default_float, -0.0f); Attr.int("default_int64", default_int64, -1L); Attr.string("default_string", default_string, "_Unused"); Attr.floats("keys_floats", keys_floats); Attr.ints("keys_int64s", keys_int64s); Attr.strings("keys_strings", keys_strings); Attr.floats("values_floats", values_floats); Attr.ints("values_int64s", values_int64s); Attr.strings("values_strings", values_strings)|] |> Array.choose id)
    static member LabelEncoder(X: Tensor<int64>, ?default_float: float32, ?default_int64: int64, ?default_string: string, ?keys_floats: float32[], ?keys_int64s: int64[], ?keys_strings: string[], ?values_floats: float32[], ?values_int64s: int64[], ?values_strings: string[]) =
        MV() |> fun mv -> execNode<int64> "LabelEncoder" [|mv.c(X)|] ([|Attr.float("default_float", default_float, -0.0f); Attr.int("default_int64", default_int64, -1L); Attr.string("default_string", default_string, "_Unused"); Attr.floats("keys_floats", keys_floats); Attr.ints("keys_int64s", keys_int64s); Attr.strings("keys_strings", keys_strings); Attr.floats("values_floats", values_floats); Attr.ints("values_int64s", values_int64s); Attr.strings("values_strings", values_strings)|] |> Array.choose id)
    static member LabelEncoder(X: Tensor<float32>, ?default_float: float32, ?default_int64: int64, ?default_string: string, ?keys_floats: float32[], ?keys_int64s: int64[], ?keys_strings: string[], ?values_floats: float32[], ?values_int64s: int64[], ?values_strings: string[]) =
        MV() |> fun mv -> execNode<float32> "LabelEncoder" [|mv.c(X)|] ([|Attr.float("default_float", default_float, -0.0f); Attr.int("default_int64", default_int64, -1L); Attr.string("default_string", default_string, "_Unused"); Attr.floats("keys_floats", keys_floats); Attr.ints("keys_int64s", keys_int64s); Attr.strings("keys_strings", keys_strings); Attr.floats("values_floats", values_floats); Attr.ints("values_int64s", values_int64s); Attr.strings("values_strings", values_strings)|] |> Array.choose id)
    static member CategoryMapper(X: Tensor<string>, ?cats_int64s: int64[], ?cats_strings: string[], ?default_int64: int64, ?default_string: string) =
        MV() |> fun mv -> execNode<string> "CategoryMapper" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("default_int64", default_int64, -1L); Attr.string("default_string", default_string, "_Unused")|] |> Array.choose id)
    static member CategoryMapper(X: Tensor<int64>, ?cats_int64s: int64[], ?cats_strings: string[], ?default_int64: int64, ?default_string: string) =
        MV() |> fun mv -> execNode<int64> "CategoryMapper" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("default_int64", default_int64, -1L); Attr.string("default_string", default_string, "_Unused")|] |> Array.choose id)
    static member SequenceEmpty<'a>() =
        execNodeCheck<'a> "SequenceEmpty" [||] [||] [||]
    static member EyeLike<'a>(input: Tensor<float32>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|1L; 6L; 9L; 7L; 2L; 3L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<float32>, ?k: int64) =
        execNodeCheck<float32> "EyeLike" [|MV.mv(1,input)|] [|1L; 6L; 9L; 7L; 2L; 3L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<int>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|1L; 6L; 9L; 7L; 2L; 3L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<int>, ?k: int64) =
        execNodeCheck<int> "EyeLike" [|MV.mv(1,input)|] [|1L; 6L; 9L; 7L; 2L; 3L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<bool>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|1L; 6L; 9L; 7L; 2L; 3L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<bool>, ?k: int64) =
        execNodeCheck<bool> "EyeLike" [|MV.mv(1,input)|] [|1L; 6L; 9L; 7L; 2L; 3L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<int64>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|1L; 6L; 9L; 7L; 2L; 3L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<int64>, ?k: int64) =
        execNodeCheck<int64> "EyeLike" [|MV.mv(1,input)|] [|1L; 6L; 9L; 7L; 2L; 3L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<uint8>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|1L; 6L; 9L; 7L; 2L; 3L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<uint8>, ?k: int64) =
        execNodeCheck<uint8> "EyeLike" [|MV.mv(1,input)|] [|1L; 6L; 9L; 7L; 2L; 3L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<int8>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|1L; 6L; 9L; 7L; 2L; 3L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<int8>, ?k: int64) =
        execNodeCheck<int8> "EyeLike" [|MV.mv(1,input)|] [|1L; 6L; 9L; 7L; 2L; 3L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member Multinomial<'a>(input: Tensor<float32>, ?sample_size: int64, ?seed: float32) =
        execNodeCheck<'a> "Multinomial" [|MV.mv(1,input)|] [|6L; 7L|] ([|Attr.int("sample_size", sample_size, 1L); Attr.float("seed", seed)|] |> Array.choose id)
    static member Multinomial(input: Tensor<float32>, ?sample_size: int64, ?seed: float32) =
        execNodeCheck<float32> "Multinomial" [|MV.mv(1,input)|] [|6L; 7L|] ([|Attr.int("sample_size", sample_size, 1L); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<float32>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<float32>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<float32> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<string>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<string>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<string> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<int>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<int>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<int> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<bool>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<bool>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<bool> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<int64>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<int64>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<int64> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<uint8>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<uint8>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<uint8> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<int8>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<int8>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<int8> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<float32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<float32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<float32> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<string>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<string>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<string> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<int>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<int>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<int> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<bool>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<bool>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<bool> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<int64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<int64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<int64> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<uint8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<uint8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<uint8> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<int8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<int8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<int8> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormal<'a>(shape: int64[], ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormal" [||] [|1L|] ([|Attr.ints("shape", shape); Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniform<'a>(shape: int64[], ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniform" [||] [|1L|] ([|Attr.ints("shape", shape); Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member Cast<'a>(input: Tensor<float32>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|1L; 8L; 6L; 9L; 7L; 2L; 3L|] [||]
    static member Cast<'a>(input: Tensor<string>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|1L; 8L; 6L; 9L; 7L; 2L; 3L|] [||]
    static member Cast<'a>(input: Tensor<int>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|1L; 8L; 6L; 9L; 7L; 2L; 3L|] [||]
    static member Cast<'a>(input: Tensor<bool>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|1L; 8L; 6L; 9L; 7L; 2L; 3L|] [||]
    static member Cast<'a>(input: Tensor<int64>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|1L; 8L; 6L; 9L; 7L; 2L; 3L|] [||]
    static member Cast<'a>(input: Tensor<uint8>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|1L; 8L; 6L; 9L; 7L; 2L; 3L|] [||]
    static member Cast<'a>(input: Tensor<int8>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|1L; 8L; 6L; 9L; 7L; 2L; 3L|] [||]
    static member TreeEnsembleClassifier(X: Tensor<float32>, ?base_values: float32[], ?class_ids: int64[], ?class_nodeids: int64[], ?class_treeids: int64[], ?class_weights: float32[], ?classlabels_int64s: int64[], ?classlabels_strings: string[], ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "TreeEnsembleClassifier" [|mv.c(X)|] ([|Attr.floats("base_values", base_values); Attr.ints("class_ids", class_ids); Attr.ints("class_nodeids", class_nodeids); Attr.ints("class_treeids", class_treeids); Attr.floats("class_weights", class_weights); Attr.ints("classlabels_int64s", classlabels_int64s); Attr.strings("classlabels_strings", classlabels_strings); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member TreeEnsembleClassifier(X: Tensor<int64>, ?base_values: float32[], ?class_ids: int64[], ?class_nodeids: int64[], ?class_treeids: int64[], ?class_weights: float32[], ?classlabels_int64s: int64[], ?classlabels_strings: string[], ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "TreeEnsembleClassifier" [|mv.c(X)|] ([|Attr.floats("base_values", base_values); Attr.ints("class_ids", class_ids); Attr.ints("class_nodeids", class_nodeids); Attr.ints("class_treeids", class_treeids); Attr.floats("class_weights", class_weights); Attr.ints("classlabels_int64s", classlabels_int64s); Attr.strings("classlabels_strings", classlabels_strings); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member TreeEnsembleClassifier(X: Tensor<int>, ?base_values: float32[], ?class_ids: int64[], ?class_nodeids: int64[], ?class_treeids: int64[], ?class_weights: float32[], ?classlabels_int64s: int64[], ?classlabels_strings: string[], ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "TreeEnsembleClassifier" [|mv.c(X)|] ([|Attr.floats("base_values", base_values); Attr.ints("class_ids", class_ids); Attr.ints("class_nodeids", class_nodeids); Attr.ints("class_treeids", class_treeids); Attr.floats("class_weights", class_weights); Attr.ints("classlabels_int64s", classlabels_int64s); Attr.strings("classlabels_strings", classlabels_strings); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member LSTM(X: Tensor<float32>, W: Tensor<float32>, R: Tensor<float32>, ?B: Tensor<float32>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<float32>, ?initial_c: Tensor<float32>, ?P: Tensor<float32>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?input_forget: int64) =
        MV() |> fun mv -> execNodeTuple3<float32, float32, float32> "LSTM" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h); mv.c(initial_c); mv.c(P)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("input_forget", input_forget, 0L)|] |> Array.choose id)
    static member LinearClassifier(X: Tensor<float32>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "LinearClassifier" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("intercepts", intercepts); Attr.int("multi_class", multi_class, 0L); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member LinearClassifier(X: Tensor<int64>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "LinearClassifier" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("intercepts", intercepts); Attr.int("multi_class", multi_class, 0L); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member LinearClassifier(X: Tensor<int>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "LinearClassifier" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("intercepts", intercepts); Attr.int("multi_class", multi_class, 0L); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member SVMClassifier(X: Tensor<float32>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "SVMClassifier" [|mv.c(X)|] ([|Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("prob_a", prob_a); Attr.floats("prob_b", prob_b); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors); Attr.ints("vectors_per_class", vectors_per_class)|] |> Array.choose id)
    static member SVMClassifier(X: Tensor<int64>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "SVMClassifier" [|mv.c(X)|] ([|Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("prob_a", prob_a); Attr.floats("prob_b", prob_b); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors); Attr.ints("vectors_per_class", vectors_per_class)|] |> Array.choose id)
    static member SVMClassifier(X: Tensor<int>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "SVMClassifier" [|mv.c(X)|] ([|Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("prob_a", prob_a); Attr.floats("prob_b", prob_b); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors); Attr.ints("vectors_per_class", vectors_per_class)|] |> Array.choose id)
    static member MaxPool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?pads: int64[], ?storage_order: int64, ?strides: int64[]) =
        MV() |> fun mv -> execNodeTuple2<float32, int64> "MaxPool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.int("storage_order", storage_order, 0L); Attr.ints("strides", strides)|] |> Array.choose id)
    static member GRU(X: Tensor<float32>, W: Tensor<float32>, R: Tensor<float32>, ?B: Tensor<float32>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<float32>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?linear_before_reset: int64) =
        MV() |> fun mv -> execNodeTuple2<float32, float32> "GRU" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("linear_before_reset", linear_before_reset, 0L)|] |> Array.choose id)
    static member TopK(X: Tensor<uint8>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<uint8, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member TopK(X: Tensor<int8>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<int8, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member TopK(X: Tensor<int>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<int, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member TopK(X: Tensor<int64>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<int64, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member TopK(X: Tensor<float32>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<float32, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Dropout(data: Tensor<float32>, ?ratio: float32) =
        MV() |> fun mv -> execNodeTuple2<float32, bool> "Dropout" [|mv.c(data)|] ([|Attr.float("ratio", ratio, 0.5f)|] |> Array.choose id)
    static member Unique(X: Tensor<uint8>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<uint8, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<int8>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<int8, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<int>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<int, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<int64>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<int64, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<float32>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<float32, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<string>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<string, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<bool>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<bool, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member DynamicQuantizeLinear(x: Tensor<float32>) =
        MV() |> fun mv -> execNodeTuple3<uint8, float32, uint8> "DynamicQuantizeLinear" [|mv.c(x)|] [||]
    static member RNN(X: Tensor<float32>, W: Tensor<float32>, R: Tensor<float32>, ?B: Tensor<float32>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<float32>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64) =
        MV() |> fun mv -> execNodeTuple2<float32, float32> "RNN" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations, [|"Tanh";"Tanh"|]); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size)|] |> Array.choose id)
    static member BatchNormalization(X: Tensor<float32>, scale: Tensor<float32>, B: Tensor<float32>, mean: Tensor<float32>, var: Tensor<float32>, ?epsilon: float32, ?momentum: float32) =
        MV() |> fun mv -> execNodeTuple5<float32, float32, float32, float32, float32> "BatchNormalization" [|mv.c(X); mv.c(scale); mv.c(B); mv.c(mean); mv.c(var)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f); Attr.float("momentum", momentum, 0.8999999761581421f)|] |> Array.choose id)
