module ONNXAPI

open System
open System.Numerics
open System.IO
open System.Text
open Onnx
open Google.Protobuf.Collections
open Microsoft.ML.OnnxRuntime.Tensors
open Microsoft.ML.OnnxRuntime
open ProtoBuf

type ONNX() =
    static member linear_regressor(X: Tensor<float32>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        MV() |> fun mv -> execNode<float32> "LinearRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member linear_regressor(X: Tensor<double>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        MV() |> fun mv -> execNode<double> "LinearRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member linear_regressor(X: Tensor<int64>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        MV() |> fun mv -> execNode<int64> "LinearRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member linear_regressor(X: Tensor<int>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        MV() |> fun mv -> execNode<int> "LinearRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member imputer(X: Tensor<float32>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        MV() |> fun mv -> execNode<float32> "Imputer" [|mv.c(X)|] ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member imputer(X: Tensor<double>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        MV() |> fun mv -> execNode<double> "Imputer" [|mv.c(X)|] ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member imputer(X: Tensor<int64>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        MV() |> fun mv -> execNode<int64> "Imputer" [|mv.c(X)|] ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member imputer(X: Tensor<int>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        MV() |> fun mv -> execNode<int> "Imputer" [|mv.c(X)|] ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member feature_vectorizer([<ParamArray>]X: Tensor<int>[], ?inputdimensions: int64[]) =
        MV() |> fun mv -> execNode<int> "FeatureVectorizer" (mv.c(X)) ([|Attr.ints("inputdimensions", inputdimensions)|] |> Array.choose id)
    static member feature_vectorizer([<ParamArray>]X: Tensor<int64>[], ?inputdimensions: int64[]) =
        MV() |> fun mv -> execNode<int64> "FeatureVectorizer" (mv.c(X)) ([|Attr.ints("inputdimensions", inputdimensions)|] |> Array.choose id)
    static member feature_vectorizer([<ParamArray>]X: Tensor<float32>[], ?inputdimensions: int64[]) =
        MV() |> fun mv -> execNode<float32> "FeatureVectorizer" (mv.c(X)) ([|Attr.ints("inputdimensions", inputdimensions)|] |> Array.choose id)
    static member feature_vectorizer([<ParamArray>]X: Tensor<double>[], ?inputdimensions: int64[]) =
        MV() |> fun mv -> execNode<double> "FeatureVectorizer" (mv.c(X)) ([|Attr.ints("inputdimensions", inputdimensions)|] |> Array.choose id)
    static member binarizer(X: Tensor<float32>, ?threshold: float32) =
        MV() |> fun mv -> execNode<float32> "Binarizer" [|mv.c(X)|] ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member binarizer(X: Tensor<double>, ?threshold: float32) =
        MV() |> fun mv -> execNode<double> "Binarizer" [|mv.c(X)|] ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member binarizer(X: Tensor<int64>, ?threshold: float32) =
        MV() |> fun mv -> execNode<int64> "Binarizer" [|mv.c(X)|] ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member binarizer(X: Tensor<int>, ?threshold: float32) =
        MV() |> fun mv -> execNode<int> "Binarizer" [|mv.c(X)|] ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member array_feature_extractor(X: Tensor<float32>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member array_feature_extractor(X: Tensor<double>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member array_feature_extractor(X: Tensor<int64>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member array_feature_extractor(X: Tensor<int>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member array_feature_extractor(X: Tensor<string>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member svm_regressor(X: Tensor<float32>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        MV() |> fun mv -> execNode<float32> "SVMRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member svm_regressor(X: Tensor<double>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        MV() |> fun mv -> execNode<double> "SVMRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member svm_regressor(X: Tensor<int64>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        MV() |> fun mv -> execNode<int64> "SVMRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member svm_regressor(X: Tensor<int>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        MV() |> fun mv -> execNode<int> "SVMRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member det(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Det" [|mv.c(X)|] [||]
    static member det(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Det" [|mv.c(X)|] [||]
    static member tree_ensemble_regressor(X: Tensor<float32>, ?aggregate_function: string, ?base_values: float32[], ?n_targets: int64, ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string, ?target_ids: int64[], ?target_nodeids: int64[], ?target_treeids: int64[], ?target_weights: float32[]) =
        MV() |> fun mv -> execNode<float32> "TreeEnsembleRegressor" [|mv.c(X)|] ([|Attr.string("aggregate_function", aggregate_function, "SUM"); Attr.floats("base_values", base_values); Attr.int("n_targets", n_targets); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE"); Attr.ints("target_ids", target_ids); Attr.ints("target_nodeids", target_nodeids); Attr.ints("target_treeids", target_treeids); Attr.floats("target_weights", target_weights)|] |> Array.choose id)
    static member tree_ensemble_regressor(X: Tensor<double>, ?aggregate_function: string, ?base_values: float32[], ?n_targets: int64, ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string, ?target_ids: int64[], ?target_nodeids: int64[], ?target_treeids: int64[], ?target_weights: float32[]) =
        MV() |> fun mv -> execNode<double> "TreeEnsembleRegressor" [|mv.c(X)|] ([|Attr.string("aggregate_function", aggregate_function, "SUM"); Attr.floats("base_values", base_values); Attr.int("n_targets", n_targets); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE"); Attr.ints("target_ids", target_ids); Attr.ints("target_nodeids", target_nodeids); Attr.ints("target_treeids", target_treeids); Attr.floats("target_weights", target_weights)|] |> Array.choose id)
    static member tree_ensemble_regressor(X: Tensor<int64>, ?aggregate_function: string, ?base_values: float32[], ?n_targets: int64, ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string, ?target_ids: int64[], ?target_nodeids: int64[], ?target_treeids: int64[], ?target_weights: float32[]) =
        MV() |> fun mv -> execNode<int64> "TreeEnsembleRegressor" [|mv.c(X)|] ([|Attr.string("aggregate_function", aggregate_function, "SUM"); Attr.floats("base_values", base_values); Attr.int("n_targets", n_targets); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE"); Attr.ints("target_ids", target_ids); Attr.ints("target_nodeids", target_nodeids); Attr.ints("target_treeids", target_treeids); Attr.floats("target_weights", target_weights)|] |> Array.choose id)
    static member tree_ensemble_regressor(X: Tensor<int>, ?aggregate_function: string, ?base_values: float32[], ?n_targets: int64, ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string, ?target_ids: int64[], ?target_nodeids: int64[], ?target_treeids: int64[], ?target_weights: float32[]) =
        MV() |> fun mv -> execNode<int> "TreeEnsembleRegressor" [|mv.c(X)|] ([|Attr.string("aggregate_function", aggregate_function, "SUM"); Attr.floats("base_values", base_values); Attr.int("n_targets", n_targets); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE"); Attr.ints("target_ids", target_ids); Attr.ints("target_nodeids", target_nodeids); Attr.ints("target_treeids", target_treeids); Attr.floats("target_weights", target_weights)|] |> Array.choose id)
    static member round(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Round" [|mv.c(X)|] [||]
    static member round(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Round" [|mv.c(X)|] [||]
    static member range(start: Tensor<float32>, limit: Tensor<float32>, delta: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Range" [|mv.c(start); mv.c(limit); mv.c(delta)|] [||]
    static member range(start: Tensor<double>, limit: Tensor<double>, delta: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Range" [|mv.c(start); mv.c(limit); mv.c(delta)|] [||]
    static member range(start: Tensor<int16>, limit: Tensor<int16>, delta: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Range" [|mv.c(start); mv.c(limit); mv.c(delta)|] [||]
    static member range(start: Tensor<int>, limit: Tensor<int>, delta: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Range" [|mv.c(start); mv.c(limit); mv.c(delta)|] [||]
    static member range(start: Tensor<int64>, limit: Tensor<int64>, delta: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Range" [|mv.c(start); mv.c(limit); mv.c(delta)|] [||]
    static member thresholded_relu(X: Tensor<float32>, ?alpha: float32) =
        MV() |> fun mv -> execNode<float32> "ThresholdedRelu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.0f)|] |> Array.choose id)
    static member thresholded_relu(X: Tensor<double>, ?alpha: float32) =
        MV() |> fun mv -> execNode<double> "ThresholdedRelu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.0f)|] |> Array.choose id)
    static member mean_variance_normalization(X: Tensor<float32>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<float32> "MeanVarianceNormalization" [|mv.c(X)|] ([|Attr.ints("axes", axes, [|0L;2L;3L|])|] |> Array.choose id)
    static member mean_variance_normalization(X: Tensor<double>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<double> "MeanVarianceNormalization" [|mv.c(X)|] ([|Attr.ints("axes", axes, [|0L;2L;3L|])|] |> Array.choose id)
    static member non_zero(X: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "NonZero" [|mv.c(X)|] [||]
    static member non_zero(X: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "NonZero" [|mv.c(X)|] [||]
    static member non_zero(X: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "NonZero" [|mv.c(X)|] [||]
    static member non_zero(X: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "NonZero" [|mv.c(X)|] [||]
    static member non_zero(X: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "NonZero" [|mv.c(X)|] [||]
    static member non_zero(X: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "NonZero" [|mv.c(X)|] [||]
    static member non_zero(X: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "NonZero" [|mv.c(X)|] [||]
    static member non_zero(X: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "NonZero" [|mv.c(X)|] [||]
    static member non_zero(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "NonZero" [|mv.c(X)|] [||]
    static member non_zero(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "NonZero" [|mv.c(X)|] [||]
    static member non_zero(X: Tensor<string>) =
        MV() |> fun mv -> execNode<string> "NonZero" [|mv.c(X)|] [||]
    static member non_zero(X: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "NonZero" [|mv.c(X)|] [||]
    static member non_zero(X: Tensor<Complex>) =
        MV() |> fun mv -> execNode<Complex> "NonZero" [|mv.c(X)|] [||]
    static member shrink(input: Tensor<uint8>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<uint8> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member shrink(input: Tensor<uint16>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<uint16> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member shrink(input: Tensor<uint32>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<uint32> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member shrink(input: Tensor<uint64>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<uint64> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member shrink(input: Tensor<int8>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<int8> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member shrink(input: Tensor<int16>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<int16> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member shrink(input: Tensor<int>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<int> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member shrink(input: Tensor<int64>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<int64> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member shrink(input: Tensor<float32>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<float32> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member shrink(input: Tensor<double>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<double> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member erf(input: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Erf" [|mv.c(input)|] [||]
    static member erf(input: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Erf" [|mv.c(input)|] [||]
    static member erf(input: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Erf" [|mv.c(input)|] [||]
    static member erf(input: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Erf" [|mv.c(input)|] [||]
    static member erf(input: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Erf" [|mv.c(input)|] [||]
    static member erf(input: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Erf" [|mv.c(input)|] [||]
    static member erf(input: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Erf" [|mv.c(input)|] [||]
    static member erf(input: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Erf" [|mv.c(input)|] [||]
    static member erf(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Erf" [|mv.c(input)|] [||]
    static member erf(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Erf" [|mv.c(input)|] [||]
    static member atanh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Atanh" [|mv.c(input)|] [||]
    static member atanh(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Atanh" [|mv.c(input)|] [||]
    static member acosh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Acosh" [|mv.c(input)|] [||]
    static member acosh(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Acosh" [|mv.c(input)|] [||]
    static member expand(input: Tensor<uint8>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member expand(input: Tensor<uint16>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint16> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member expand(input: Tensor<uint32>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint32> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member expand(input: Tensor<uint64>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint64> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member expand(input: Tensor<int8>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member expand(input: Tensor<int16>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int16> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member expand(input: Tensor<int>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member expand(input: Tensor<int64>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member expand(input: Tensor<float32>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member expand(input: Tensor<double>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member expand(input: Tensor<string>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member expand(input: Tensor<bool>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member expand(input: Tensor<Complex>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<Complex> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member atan(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Atan" [|mv.c(input)|] [||]
    static member atan(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Atan" [|mv.c(input)|] [||]
    static member asin(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Asin" [|mv.c(input)|] [||]
    static member asin(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Asin" [|mv.c(input)|] [||]
    static member lp_normalization(input: Tensor<float32>, ?axis: int64, ?p: int64) =
        MV() |> fun mv -> execNode<float32> "LpNormalization" [|mv.c(input)|] ([|Attr.int("axis", axis, -1L); Attr.int("p", p, 2L)|] |> Array.choose id)
    static member lp_normalization(input: Tensor<double>, ?axis: int64, ?p: int64) =
        MV() |> fun mv -> execNode<double> "LpNormalization" [|mv.c(input)|] ([|Attr.int("axis", axis, -1L); Attr.int("p", p, 2L)|] |> Array.choose id)
    static member ceil(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Ceil" [|mv.c(X)|] [||]
    static member ceil(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Ceil" [|mv.c(X)|] [||]
    static member log_softmax(input: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "LogSoftmax" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member log_softmax(input: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "LogSoftmax" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member mat_mul(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member mat_mul(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member mat_mul(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member mat_mul(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member mat_mul(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member mat_mul(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member bit_shift(X: Tensor<uint8>, Y: Tensor<uint8>, direction: string) =
        MV() |> fun mv -> execNode<uint8> "BitShift" [|mv.c(X); mv.c(Y)|] ([|Attr.string("direction", direction)|] |> Array.choose id)
    static member bit_shift(X: Tensor<uint16>, Y: Tensor<uint16>, direction: string) =
        MV() |> fun mv -> execNode<uint16> "BitShift" [|mv.c(X); mv.c(Y)|] ([|Attr.string("direction", direction)|] |> Array.choose id)
    static member bit_shift(X: Tensor<uint32>, Y: Tensor<uint32>, direction: string) =
        MV() |> fun mv -> execNode<uint32> "BitShift" [|mv.c(X); mv.c(Y)|] ([|Attr.string("direction", direction)|] |> Array.choose id)
    static member bit_shift(X: Tensor<uint64>, Y: Tensor<uint64>, direction: string) =
        MV() |> fun mv -> execNode<uint64> "BitShift" [|mv.c(X); mv.c(Y)|] ([|Attr.string("direction", direction)|] |> Array.choose id)
    static member sinh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sinh" [|mv.c(input)|] [||]
    static member sinh(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Sinh" [|mv.c(input)|] [||]
    static member acos(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Acos" [|mv.c(input)|] [||]
    static member acos(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Acos" [|mv.c(input)|] [||]
    static member identity(input: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Identity" [|mv.c(input)|] [||]
    static member identity(input: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Identity" [|mv.c(input)|] [||]
    static member identity(input: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Identity" [|mv.c(input)|] [||]
    static member identity(input: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Identity" [|mv.c(input)|] [||]
    static member identity(input: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Identity" [|mv.c(input)|] [||]
    static member identity(input: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Identity" [|mv.c(input)|] [||]
    static member identity(input: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Identity" [|mv.c(input)|] [||]
    static member identity(input: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Identity" [|mv.c(input)|] [||]
    static member identity(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Identity" [|mv.c(input)|] [||]
    static member identity(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Identity" [|mv.c(input)|] [||]
    static member identity(input: Tensor<string>) =
        MV() |> fun mv -> execNode<string> "Identity" [|mv.c(input)|] [||]
    static member identity(input: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Identity" [|mv.c(input)|] [||]
    static member identity(input: Tensor<Complex>) =
        MV() |> fun mv -> execNode<Complex> "Identity" [|mv.c(input)|] [||]
    static member pow(X: Tensor<float32>, Y: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member pow(X: Tensor<double>, Y: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member mod_(A: Tensor<uint8>, B: Tensor<uint8>, ?fmod: int64) =
        MV() |> fun mv -> execNode<uint8> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member mod_(A: Tensor<uint16>, B: Tensor<uint16>, ?fmod: int64) =
        MV() |> fun mv -> execNode<uint16> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member mod_(A: Tensor<uint32>, B: Tensor<uint32>, ?fmod: int64) =
        MV() |> fun mv -> execNode<uint32> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member mod_(A: Tensor<uint64>, B: Tensor<uint64>, ?fmod: int64) =
        MV() |> fun mv -> execNode<uint64> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member mod_(A: Tensor<int8>, B: Tensor<int8>, ?fmod: int64) =
        MV() |> fun mv -> execNode<int8> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member mod_(A: Tensor<int16>, B: Tensor<int16>, ?fmod: int64) =
        MV() |> fun mv -> execNode<int16> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member mod_(A: Tensor<int>, B: Tensor<int>, ?fmod: int64) =
        MV() |> fun mv -> execNode<int> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member mod_(A: Tensor<int64>, B: Tensor<int64>, ?fmod: int64) =
        MV() |> fun mv -> execNode<int64> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member mod_(A: Tensor<float32>, B: Tensor<float32>, ?fmod: int64) =
        MV() |> fun mv -> execNode<float32> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member mod_(A: Tensor<double>, B: Tensor<double>, ?fmod: int64) =
        MV() |> fun mv -> execNode<double> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member softplus(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Softplus" [|mv.c(X)|] [||]
    static member softplus(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Softplus" [|mv.c(X)|] [||]
    static member normalizer(X: Tensor<float32>, ?norm: string) =
        MV() |> fun mv -> execNode<float32> "Normalizer" [|mv.c(X)|] ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member normalizer(X: Tensor<double>, ?norm: string) =
        MV() |> fun mv -> execNode<double> "Normalizer" [|mv.c(X)|] ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member normalizer(X: Tensor<int64>, ?norm: string) =
        MV() |> fun mv -> execNode<int64> "Normalizer" [|mv.c(X)|] ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member normalizer(X: Tensor<int>, ?norm: string) =
        MV() |> fun mv -> execNode<int> "Normalizer" [|mv.c(X)|] ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member hardmax(input: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Hardmax" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member hardmax(input: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Hardmax" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member hard_sigmoid(X: Tensor<float32>, ?alpha: float32, ?beta: float32) =
        MV() |> fun mv -> execNode<float32> "HardSigmoid" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 0.20000000298023224f); Attr.float("beta", beta, 0.5f)|] |> Array.choose id)
    static member hard_sigmoid(X: Tensor<double>, ?alpha: float32, ?beta: float32) =
        MV() |> fun mv -> execNode<double> "HardSigmoid" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 0.20000000298023224f); Attr.float("beta", beta, 0.5f)|] |> Array.choose id)
    static member lp_pool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?p: int64, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "LpPool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("p", p, 2L); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member lp_pool(X: Tensor<double>, kernel_shape: int64[], ?auto_pad: string, ?p: int64, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<double> "LpPool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("p", p, 2L); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member min([<ParamArray>]data_0: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Min" (mv.c(data_0)) [||]
    static member min([<ParamArray>]data_0: Tensor<double>[]) =
        MV() |> fun mv -> execNode<double> "Min" (mv.c(data_0)) [||]
    static member sum([<ParamArray>]data_0: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Sum" (mv.c(data_0)) [||]
    static member sum([<ParamArray>]data_0: Tensor<double>[]) =
        MV() |> fun mv -> execNode<double> "Sum" (mv.c(data_0)) [||]
    static member transpose(data: Tensor<uint8>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<uint8> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member transpose(data: Tensor<uint16>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<uint16> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member transpose(data: Tensor<uint32>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<uint32> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member transpose(data: Tensor<uint64>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<uint64> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member transpose(data: Tensor<int8>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<int8> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member transpose(data: Tensor<int16>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<int16> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member transpose(data: Tensor<int>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<int> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member transpose(data: Tensor<int64>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<int64> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member transpose(data: Tensor<float32>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<float32> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member transpose(data: Tensor<double>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<double> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member transpose(data: Tensor<string>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<string> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member transpose(data: Tensor<bool>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<bool> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member transpose(data: Tensor<Complex>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<Complex> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member scatternd(data: Tensor<uint8>, indices: Tensor<int64>, updates: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member scatternd(data: Tensor<uint16>, indices: Tensor<int64>, updates: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member scatternd(data: Tensor<uint32>, indices: Tensor<int64>, updates: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member scatternd(data: Tensor<uint64>, indices: Tensor<int64>, updates: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member scatternd(data: Tensor<int8>, indices: Tensor<int64>, updates: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member scatternd(data: Tensor<int16>, indices: Tensor<int64>, updates: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member scatternd(data: Tensor<int>, indices: Tensor<int64>, updates: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member scatternd(data: Tensor<int64>, indices: Tensor<int64>, updates: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member scatternd(data: Tensor<float32>, indices: Tensor<int64>, updates: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member scatternd(data: Tensor<double>, indices: Tensor<int64>, updates: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member scatternd(data: Tensor<string>, indices: Tensor<int64>, updates: Tensor<string>) =
        MV() |> fun mv -> execNode<string> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member scatternd(data: Tensor<bool>, indices: Tensor<int64>, updates: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member scatternd(data: Tensor<Complex>, indices: Tensor<int64>, updates: Tensor<Complex>) =
        MV() |> fun mv -> execNode<Complex> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] [||]
    static member global_lp_pool(X: Tensor<float32>, ?p: int64) =
        MV() |> fun mv -> execNode<float32> "GlobalLpPool" [|mv.c(X)|] ([|Attr.int("p", p, 2L)|] |> Array.choose id)
    static member global_lp_pool(X: Tensor<double>, ?p: int64) =
        MV() |> fun mv -> execNode<double> "GlobalLpPool" [|mv.c(X)|] ([|Attr.int("p", p, 2L)|] |> Array.choose id)
    static member gemm(A: Tensor<float32>, B: Tensor<float32>, ?C: Tensor<float32>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<float32> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member gemm(A: Tensor<double>, B: Tensor<double>, ?C: Tensor<double>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<double> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member gemm(A: Tensor<uint32>, B: Tensor<uint32>, ?C: Tensor<uint32>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<uint32> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member gemm(A: Tensor<uint64>, B: Tensor<uint64>, ?C: Tensor<uint64>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<uint64> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member gemm(A: Tensor<int>, B: Tensor<int>, ?C: Tensor<int>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<int> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member gemm(A: Tensor<int64>, B: Tensor<int64>, ?C: Tensor<int64>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<int64> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member instance_normalization(input: Tensor<float32>, scale: Tensor<float32>, B: Tensor<float32>, ?epsilon: float32) =
        MV() |> fun mv -> execNode<float32> "InstanceNormalization" [|mv.c(input); mv.c(scale); mv.c(B)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f)|] |> Array.choose id)
    static member instance_normalization(input: Tensor<double>, scale: Tensor<double>, B: Tensor<double>, ?epsilon: float32) =
        MV() |> fun mv -> execNode<double> "InstanceNormalization" [|mv.c(input); mv.c(scale); mv.c(B)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f)|] |> Array.choose id)
    static member average_pool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?count_include_pad: int64, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "AveragePool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("count_include_pad", count_include_pad, 0L); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member average_pool(X: Tensor<double>, kernel_shape: int64[], ?auto_pad: string, ?count_include_pad: int64, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<double> "AveragePool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("count_include_pad", count_include_pad, 0L); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member sign(input: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Sign" [|mv.c(input)|] [||]
    static member sign(input: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Sign" [|mv.c(input)|] [||]
    static member sign(input: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Sign" [|mv.c(input)|] [||]
    static member sign(input: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Sign" [|mv.c(input)|] [||]
    static member sign(input: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Sign" [|mv.c(input)|] [||]
    static member sign(input: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Sign" [|mv.c(input)|] [||]
    static member sign(input: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Sign" [|mv.c(input)|] [||]
    static member sign(input: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Sign" [|mv.c(input)|] [||]
    static member sign(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sign" [|mv.c(input)|] [||]
    static member sign(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Sign" [|mv.c(input)|] [||]
    static member clip(input: Tensor<float32>, ?min: Tensor<float32>, ?max: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Clip" ([|Some(mv.c(input)); mv.c(min); mv.c(max)|] |> Array.choose id) [||]
    static member clip(input: Tensor<double>, ?min: Tensor<double>, ?max: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Clip" ([|Some(mv.c(input)); mv.c(min); mv.c(max)|] |> Array.choose id) [||]
    static member dequantize_linear(x: Tensor<int8>, x_scale: Tensor<float32>, ?x_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "DequantizeLinear" ([|Some(mv.c(x)); Some(mv.c(x_scale)); mv.c(x_zero_point)|] |> Array.choose id) [||]
    static member dequantize_linear(x: Tensor<uint8>, x_scale: Tensor<float32>, ?x_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "DequantizeLinear" ([|Some(mv.c(x)); Some(mv.c(x_scale)); mv.c(x_zero_point)|] |> Array.choose id) [||]
    static member dequantize_linear(x: Tensor<int>, x_scale: Tensor<float32>, ?x_zero_point: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "DequantizeLinear" ([|Some(mv.c(x)); Some(mv.c(x_scale)); mv.c(x_zero_point)|] |> Array.choose id) [||]
    static member lrn(X: Tensor<float32>, size: int64, ?alpha: float32, ?beta: float32, ?bias: float32) =
        MV() |> fun mv -> execNode<float32> "LRN" [|mv.c(X)|] ([|Attr.int("size", size); Attr.float("alpha", alpha, 9.999999747378752e-05f); Attr.float("beta", beta, 0.75f); Attr.float("bias", bias, 1.0f)|] |> Array.choose id)
    static member lrn(X: Tensor<double>, size: int64, ?alpha: float32, ?beta: float32, ?bias: float32) =
        MV() |> fun mv -> execNode<double> "LRN" [|mv.c(X)|] ([|Attr.int("size", size); Attr.float("alpha", alpha, 9.999999747378752e-05f); Attr.float("beta", beta, 0.75f); Attr.float("bias", bias, 1.0f)|] |> Array.choose id)
    static member elu(X: Tensor<float32>, ?alpha: float32) =
        MV() |> fun mv -> execNode<float32> "Elu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.0f)|] |> Array.choose id)
    static member elu(X: Tensor<double>, ?alpha: float32) =
        MV() |> fun mv -> execNode<double> "Elu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.0f)|] |> Array.choose id)
    static member sin(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sin" [|mv.c(input)|] [||]
    static member sin(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Sin" [|mv.c(input)|] [||]
    static member pad(data: Tensor<uint8>, pads: Tensor<int64>, ?constant_value: Tensor<uint8>, ?mode: string) =
        MV() |> fun mv -> execNode<uint8> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member pad(data: Tensor<uint16>, pads: Tensor<int64>, ?constant_value: Tensor<uint16>, ?mode: string) =
        MV() |> fun mv -> execNode<uint16> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member pad(data: Tensor<uint32>, pads: Tensor<int64>, ?constant_value: Tensor<uint32>, ?mode: string) =
        MV() |> fun mv -> execNode<uint32> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member pad(data: Tensor<uint64>, pads: Tensor<int64>, ?constant_value: Tensor<uint64>, ?mode: string) =
        MV() |> fun mv -> execNode<uint64> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member pad(data: Tensor<int8>, pads: Tensor<int64>, ?constant_value: Tensor<int8>, ?mode: string) =
        MV() |> fun mv -> execNode<int8> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member pad(data: Tensor<int16>, pads: Tensor<int64>, ?constant_value: Tensor<int16>, ?mode: string) =
        MV() |> fun mv -> execNode<int16> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member pad(data: Tensor<int>, pads: Tensor<int64>, ?constant_value: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<int> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member pad(data: Tensor<int64>, pads: Tensor<int64>, ?constant_value: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<int64> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member pad(data: Tensor<float32>, pads: Tensor<int64>, ?constant_value: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<float32> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member pad(data: Tensor<double>, pads: Tensor<int64>, ?constant_value: Tensor<double>, ?mode: string) =
        MV() |> fun mv -> execNode<double> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member gathernd(data: Tensor<uint8>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member gathernd(data: Tensor<uint16>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint16> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member gathernd(data: Tensor<uint32>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint32> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member gathernd(data: Tensor<uint64>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint64> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member gathernd(data: Tensor<int8>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member gathernd(data: Tensor<int16>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<int16> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member gathernd(data: Tensor<int>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member gathernd(data: Tensor<int64>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member gathernd(data: Tensor<float32>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member gathernd(data: Tensor<double>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member gathernd(data: Tensor<string>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member gathernd(data: Tensor<bool>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member gathernd(data: Tensor<Complex>, indices: Tensor<int64>) =
        MV() |> fun mv -> execNode<Complex> "GatherND" [|mv.c(data); mv.c(indices)|] [||]
    static member relu(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Relu" [|mv.c(X)|] [||]
    static member relu(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Relu" [|mv.c(X)|] [||]
    static member conv(X: Tensor<float32>, W: Tensor<float32>, ?B: Tensor<float32>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "Conv" ([|Some(mv.c(X)); Some(mv.c(W)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member conv(X: Tensor<double>, W: Tensor<double>, ?B: Tensor<double>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<double> "Conv" ([|Some(mv.c(X)); Some(mv.c(W)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member arg_max(data: Tensor<uint8>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint8> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_max(data: Tensor<uint16>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint16> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_max(data: Tensor<uint32>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint32> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_max(data: Tensor<uint64>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint64> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_max(data: Tensor<int8>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int8> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_max(data: Tensor<int16>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int16> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_max(data: Tensor<int>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_max(data: Tensor<int64>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_max(data: Tensor<float32>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_max(data: Tensor<double>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<double> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member div(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member div(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member div(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member div(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member div(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member div(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member max_roi_pool(X: Tensor<float32>, rois: Tensor<float32>, pooled_shape: int64[], ?spatial_scale: float32) =
        MV() |> fun mv -> execNode<float32> "MaxRoiPool" [|mv.c(X); mv.c(rois)|] ([|Attr.ints("pooled_shape", pooled_shape); Attr.float("spatial_scale", spatial_scale, 1.0f)|] |> Array.choose id)
    static member max_roi_pool(X: Tensor<double>, rois: Tensor<double>, pooled_shape: int64[], ?spatial_scale: float32) =
        MV() |> fun mv -> execNode<double> "MaxRoiPool" [|mv.c(X); mv.c(rois)|] ([|Attr.ints("pooled_shape", pooled_shape); Attr.float("spatial_scale", spatial_scale, 1.0f)|] |> Array.choose id)
    static member add(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member add(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member add(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member add(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member add(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member add(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member leaky_relu(X: Tensor<float32>, ?alpha: float32) =
        MV() |> fun mv -> execNode<float32> "LeakyRelu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 0.009999999776482582f)|] |> Array.choose id)
    static member leaky_relu(X: Tensor<double>, ?alpha: float32) =
        MV() |> fun mv -> execNode<double> "LeakyRelu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 0.009999999776482582f)|] |> Array.choose id)
    static member reduce_log_sum(data: Tensor<uint32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceLogSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_log_sum(data: Tensor<uint64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceLogSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_log_sum(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceLogSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_log_sum(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceLogSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_log_sum(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceLogSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_log_sum(data: Tensor<double>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<double> "ReduceLogSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member floor(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Floor" [|mv.c(X)|] [||]
    static member floor(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Floor" [|mv.c(X)|] [||]
    static member arg_min(data: Tensor<uint8>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint8> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_min(data: Tensor<uint16>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint16> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_min(data: Tensor<uint32>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint32> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_min(data: Tensor<uint64>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint64> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_min(data: Tensor<int8>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int8> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_min(data: Tensor<int16>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int16> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_min(data: Tensor<int>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_min(data: Tensor<int64>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_min(data: Tensor<float32>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member arg_min(data: Tensor<double>, ?axis: int64, ?keepdims: int64) =
        MV() |> fun mv -> execNode<double> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member depth_to_space(input: Tensor<uint8>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<uint8> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member depth_to_space(input: Tensor<uint16>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<uint16> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member depth_to_space(input: Tensor<uint32>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<uint32> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member depth_to_space(input: Tensor<uint64>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<uint64> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member depth_to_space(input: Tensor<int8>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<int8> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member depth_to_space(input: Tensor<int16>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<int16> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member depth_to_space(input: Tensor<int>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<int> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member depth_to_space(input: Tensor<int64>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<int64> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member depth_to_space(input: Tensor<float32>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<float32> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member depth_to_space(input: Tensor<double>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<double> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member depth_to_space(input: Tensor<string>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<string> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member depth_to_space(input: Tensor<bool>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<bool> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member depth_to_space(input: Tensor<Complex>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<Complex> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member tan(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Tan" [|mv.c(input)|] [||]
    static member tan(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Tan" [|mv.c(input)|] [||]
    static member reduce_sum(data: Tensor<uint32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_sum(data: Tensor<uint64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_sum(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_sum(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_sum(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_sum(data: Tensor<double>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<double> "ReduceSum" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<uint8>[]) =
        MV() |> fun mv -> execNode<uint8> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<uint16>[]) =
        MV() |> fun mv -> execNode<uint16> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<uint32>[]) =
        MV() |> fun mv -> execNode<uint32> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<uint64>[]) =
        MV() |> fun mv -> execNode<uint64> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<int8>[]) =
        MV() |> fun mv -> execNode<int8> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<int16>[]) =
        MV() |> fun mv -> execNode<int16> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<int>[]) =
        MV() |> fun mv -> execNode<int> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<int64>[]) =
        MV() |> fun mv -> execNode<int64> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<double>[]) =
        MV() |> fun mv -> execNode<double> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<string>[]) =
        MV() |> fun mv -> execNode<string> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<bool>[]) =
        MV() |> fun mv -> execNode<bool> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<Complex>[]) =
        MV() |> fun mv -> execNode<Complex> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member one_hot_encoder(X: Tensor<string>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<string> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member one_hot_encoder(X: Tensor<int64>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<int64> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member one_hot_encoder(X: Tensor<int>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<int> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member one_hot_encoder(X: Tensor<float32>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<float32> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member one_hot_encoder(X: Tensor<double>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<double> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member conv_transpose(X: Tensor<float32>, W: Tensor<float32>, ?B: Tensor<float32>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?output_padding: int64[], ?output_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "ConvTranspose" ([|Some(mv.c(X)); Some(mv.c(W)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("output_padding", output_padding); Attr.ints("output_shape", output_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member conv_transpose(X: Tensor<double>, W: Tensor<double>, ?B: Tensor<double>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?output_padding: int64[], ?output_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<double> "ConvTranspose" ([|Some(mv.c(X)); Some(mv.c(W)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("output_padding", output_padding); Attr.ints("output_shape", output_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member reverse_sequence(input: Tensor<uint8>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<uint8> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member reverse_sequence(input: Tensor<uint16>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<uint16> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member reverse_sequence(input: Tensor<uint32>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<uint32> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member reverse_sequence(input: Tensor<uint64>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<uint64> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member reverse_sequence(input: Tensor<int8>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<int8> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member reverse_sequence(input: Tensor<int16>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<int16> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member reverse_sequence(input: Tensor<int>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<int> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member reverse_sequence(input: Tensor<int64>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<int64> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member reverse_sequence(input: Tensor<float32>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<float32> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member reverse_sequence(input: Tensor<double>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<double> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member reverse_sequence(input: Tensor<string>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<string> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member reverse_sequence(input: Tensor<bool>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<bool> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member reverse_sequence(input: Tensor<Complex>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<Complex> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member max([<ParamArray>]data_0: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Max" (mv.c(data_0)) [||]
    static member max([<ParamArray>]data_0: Tensor<double>[]) =
        MV() |> fun mv -> execNode<double> "Max" (mv.c(data_0)) [||]
    static member global_max_pool(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "GlobalMaxPool" [|mv.c(X)|] [||]
    static member global_max_pool(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "GlobalMaxPool" [|mv.c(X)|] [||]
    static member exp(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Exp" [|mv.c(input)|] [||]
    static member exp(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Exp" [|mv.c(input)|] [||]
    static member reshape(data: Tensor<uint8>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member reshape(data: Tensor<uint16>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint16> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member reshape(data: Tensor<uint32>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint32> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member reshape(data: Tensor<uint64>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint64> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member reshape(data: Tensor<int8>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member reshape(data: Tensor<int16>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int16> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member reshape(data: Tensor<int>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member reshape(data: Tensor<int64>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member reshape(data: Tensor<float32>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member reshape(data: Tensor<double>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member reshape(data: Tensor<string>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member reshape(data: Tensor<bool>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member reshape(data: Tensor<Complex>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<Complex> "Reshape" [|mv.c(data); mv.c(shape)|] [||]
    static member global_average_pool(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "GlobalAveragePool" [|mv.c(X)|] [||]
    static member global_average_pool(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "GlobalAveragePool" [|mv.c(X)|] [||]
    static member mean([<ParamArray>]data_0: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Mean" (mv.c(data_0)) [||]
    static member mean([<ParamArray>]data_0: Tensor<double>[]) =
        MV() |> fun mv -> execNode<double> "Mean" (mv.c(data_0)) [||]
    static member mul(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member mul(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member mul(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member mul(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member mul(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member mul(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member neg(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Neg" [|mv.c(X)|] [||]
    static member neg(X: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Neg" [|mv.c(X)|] [||]
    static member neg(X: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Neg" [|mv.c(X)|] [||]
    static member neg(X: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Neg" [|mv.c(X)|] [||]
    static member neg(X: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Neg" [|mv.c(X)|] [||]
    static member neg(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Neg" [|mv.c(X)|] [||]
    static member not_(X: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Not" [|mv.c(X)|] [||]
    static member reducel1(data: Tensor<uint32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceL1" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reducel1(data: Tensor<uint64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceL1" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reducel1(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceL1" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reducel1(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceL1" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reducel1(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceL1" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reducel1(data: Tensor<double>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<double> "ReduceL1" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member flatten(input: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member flatten(input: Tensor<uint16>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member flatten(input: Tensor<uint32>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member flatten(input: Tensor<uint64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member flatten(input: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member flatten(input: Tensor<int16>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member flatten(input: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member flatten(input: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member flatten(input: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member flatten(input: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member flatten(input: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member flatten(input: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member flatten(input: Tensor<Complex>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member p_relu(X: Tensor<float32>, slope: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member p_relu(X: Tensor<double>, slope: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member p_relu(X: Tensor<uint32>, slope: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member p_relu(X: Tensor<uint64>, slope: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member p_relu(X: Tensor<int>, slope: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member p_relu(X: Tensor<int64>, slope: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member unsqueeze(data: Tensor<uint8>, axes: int64[]) =
        MV() |> fun mv -> execNode<uint8> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member unsqueeze(data: Tensor<uint16>, axes: int64[]) =
        MV() |> fun mv -> execNode<uint16> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member unsqueeze(data: Tensor<uint32>, axes: int64[]) =
        MV() |> fun mv -> execNode<uint32> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member unsqueeze(data: Tensor<uint64>, axes: int64[]) =
        MV() |> fun mv -> execNode<uint64> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member unsqueeze(data: Tensor<int8>, axes: int64[]) =
        MV() |> fun mv -> execNode<int8> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member unsqueeze(data: Tensor<int16>, axes: int64[]) =
        MV() |> fun mv -> execNode<int16> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member unsqueeze(data: Tensor<int>, axes: int64[]) =
        MV() |> fun mv -> execNode<int> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member unsqueeze(data: Tensor<int64>, axes: int64[]) =
        MV() |> fun mv -> execNode<int64> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member unsqueeze(data: Tensor<float32>, axes: int64[]) =
        MV() |> fun mv -> execNode<float32> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member unsqueeze(data: Tensor<double>, axes: int64[]) =
        MV() |> fun mv -> execNode<double> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member unsqueeze(data: Tensor<string>, axes: int64[]) =
        MV() |> fun mv -> execNode<string> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member unsqueeze(data: Tensor<bool>, axes: int64[]) =
        MV() |> fun mv -> execNode<bool> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member unsqueeze(data: Tensor<Complex>, axes: int64[]) =
        MV() |> fun mv -> execNode<Complex> "Unsqueeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member tanh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Tanh" [|mv.c(input)|] [||]
    static member tanh(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Tanh" [|mv.c(input)|] [||]
    static member abs(X: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Abs" [|mv.c(X)|] [||]
    static member abs(X: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Abs" [|mv.c(X)|] [||]
    static member abs(X: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Abs" [|mv.c(X)|] [||]
    static member abs(X: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Abs" [|mv.c(X)|] [||]
    static member abs(X: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Abs" [|mv.c(X)|] [||]
    static member abs(X: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Abs" [|mv.c(X)|] [||]
    static member abs(X: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Abs" [|mv.c(X)|] [||]
    static member abs(X: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Abs" [|mv.c(X)|] [||]
    static member abs(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Abs" [|mv.c(X)|] [||]
    static member abs(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Abs" [|mv.c(X)|] [||]
    static member reciprocal(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Reciprocal" [|mv.c(X)|] [||]
    static member reciprocal(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Reciprocal" [|mv.c(X)|] [||]
    static member reduce_log_sum_exp(data: Tensor<uint32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceLogSumExp" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_log_sum_exp(data: Tensor<uint64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceLogSumExp" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_log_sum_exp(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceLogSumExp" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_log_sum_exp(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceLogSumExp" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_log_sum_exp(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceLogSumExp" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_log_sum_exp(data: Tensor<double>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<double> "ReduceLogSumExp" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_max(data: Tensor<uint32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceMax" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_max(data: Tensor<uint64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceMax" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_max(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceMax" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_max(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceMax" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_max(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceMax" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_max(data: Tensor<double>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<double> "ReduceMax" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_mean(data: Tensor<uint32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceMean" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_mean(data: Tensor<uint64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceMean" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_mean(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceMean" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_mean(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceMean" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_mean(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceMean" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_mean(data: Tensor<double>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<double> "ReduceMean" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member cosh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Cosh" [|mv.c(input)|] [||]
    static member cosh(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Cosh" [|mv.c(input)|] [||]
    static member reduce_min(data: Tensor<uint32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceMin" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_min(data: Tensor<uint64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceMin" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_min(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceMin" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_min(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceMin" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_min(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceMin" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_min(data: Tensor<double>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<double> "ReduceMin" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_prod(data: Tensor<uint32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceProd" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_prod(data: Tensor<uint64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceProd" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_prod(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceProd" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_prod(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceProd" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_prod(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceProd" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_prod(data: Tensor<double>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<double> "ReduceProd" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member squeeze(data: Tensor<uint8>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<uint8> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member squeeze(data: Tensor<uint16>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<uint16> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member squeeze(data: Tensor<uint32>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<uint32> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member squeeze(data: Tensor<uint64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<uint64> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member squeeze(data: Tensor<int8>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int8> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member squeeze(data: Tensor<int16>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int16> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member squeeze(data: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member squeeze(data: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int64> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member squeeze(data: Tensor<float32>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<float32> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member squeeze(data: Tensor<double>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<double> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member squeeze(data: Tensor<string>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<string> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member squeeze(data: Tensor<bool>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<bool> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member squeeze(data: Tensor<Complex>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<Complex> "Squeeze" [|mv.c(data)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member selu(X: Tensor<float32>, ?alpha: float32, ?gamma: float32) =
        MV() |> fun mv -> execNode<float32> "Selu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.6732631921768188f); Attr.float("gamma", gamma, 1.0507010221481323f)|] |> Array.choose id)
    static member selu(X: Tensor<double>, ?alpha: float32, ?gamma: float32) =
        MV() |> fun mv -> execNode<double> "Selu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.6732631921768188f); Attr.float("gamma", gamma, 1.0507010221481323f)|] |> Array.choose id)
    static member sigmoid(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sigmoid" [|mv.c(X)|] [||]
    static member sigmoid(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Sigmoid" [|mv.c(X)|] [||]
    static member reduce_sum_square(data: Tensor<uint32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceSumSquare" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_sum_square(data: Tensor<uint64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceSumSquare" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_sum_square(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceSumSquare" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_sum_square(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceSumSquare" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_sum_square(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceSumSquare" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reduce_sum_square(data: Tensor<double>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<double> "ReduceSumSquare" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member softmax(input: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Softmax" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member softmax(input: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Softmax" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member softsign(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Softsign" [|mv.c(input)|] [||]
    static member softsign(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Softsign" [|mv.c(input)|] [||]
    static member cos(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Cos" [|mv.c(input)|] [||]
    static member cos(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Cos" [|mv.c(input)|] [||]
    static member space_to_depth(input: Tensor<uint8>, blocksize: int64) =
        MV() |> fun mv -> execNode<uint8> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member space_to_depth(input: Tensor<uint16>, blocksize: int64) =
        MV() |> fun mv -> execNode<uint16> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member space_to_depth(input: Tensor<uint32>, blocksize: int64) =
        MV() |> fun mv -> execNode<uint32> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member space_to_depth(input: Tensor<uint64>, blocksize: int64) =
        MV() |> fun mv -> execNode<uint64> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member space_to_depth(input: Tensor<int8>, blocksize: int64) =
        MV() |> fun mv -> execNode<int8> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member space_to_depth(input: Tensor<int16>, blocksize: int64) =
        MV() |> fun mv -> execNode<int16> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member space_to_depth(input: Tensor<int>, blocksize: int64) =
        MV() |> fun mv -> execNode<int> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member space_to_depth(input: Tensor<int64>, blocksize: int64) =
        MV() |> fun mv -> execNode<int64> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member space_to_depth(input: Tensor<float32>, blocksize: int64) =
        MV() |> fun mv -> execNode<float32> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member space_to_depth(input: Tensor<double>, blocksize: int64) =
        MV() |> fun mv -> execNode<double> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member space_to_depth(input: Tensor<string>, blocksize: int64) =
        MV() |> fun mv -> execNode<string> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member space_to_depth(input: Tensor<bool>, blocksize: int64) =
        MV() |> fun mv -> execNode<bool> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member space_to_depth(input: Tensor<Complex>, blocksize: int64) =
        MV() |> fun mv -> execNode<Complex> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member asinh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Asinh" [|mv.c(input)|] [||]
    static member asinh(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Asinh" [|mv.c(input)|] [||]
    static member reducel2(data: Tensor<uint32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceL2" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reducel2(data: Tensor<uint64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceL2" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reducel2(data: Tensor<int>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int> "ReduceL2" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reducel2(data: Tensor<int64>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceL2" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reducel2(data: Tensor<float32>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceL2" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member reducel2(data: Tensor<double>, ?axes: int64[], ?keepdims: int64) =
        MV() |> fun mv -> execNode<double> "ReduceL2" [|mv.c(data)|] ([|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|] |> Array.choose id)
    static member sqrt(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sqrt" [|mv.c(X)|] [||]
    static member sqrt(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Sqrt" [|mv.c(X)|] [||]
    static member log(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Log" [|mv.c(input)|] [||]
    static member log(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Log" [|mv.c(input)|] [||]
    static member sub(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member sub(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member sub(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member sub(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member sub(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member sub(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member scaler(X: Tensor<float32>, ?offset: float32[], ?scale: float32[]) =
        MV() |> fun mv -> execNode<float32> "Scaler" [|mv.c(X)|] ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member scaler(X: Tensor<double>, ?offset: float32[], ?scale: float32[]) =
        MV() |> fun mv -> execNode<double> "Scaler" [|mv.c(X)|] ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member scaler(X: Tensor<int64>, ?offset: float32[], ?scale: float32[]) =
        MV() |> fun mv -> execNode<int64> "Scaler" [|mv.c(X)|] ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member scaler(X: Tensor<int>, ?offset: float32[], ?scale: float32[]) =
        MV() |> fun mv -> execNode<int> "Scaler" [|mv.c(X)|] ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member upsample(X: Tensor<uint8>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<uint8> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member upsample(X: Tensor<uint16>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<uint16> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member upsample(X: Tensor<uint32>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<uint32> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member upsample(X: Tensor<uint64>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<uint64> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member upsample(X: Tensor<int8>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<int8> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member upsample(X: Tensor<int16>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<int16> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member upsample(X: Tensor<int>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<int> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member upsample(X: Tensor<int64>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<int64> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member upsample(X: Tensor<float32>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<float32> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member upsample(X: Tensor<double>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<double> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member upsample(X: Tensor<string>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<string> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member upsample(X: Tensor<bool>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<bool> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member upsample(X: Tensor<Complex>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<Complex> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member is_inf(X: Tensor<float32>, ?detect_negative: int64, ?detect_positive: int64) =
        MV() |> fun mv -> execNode<bool> "IsInf" [|mv.c(X)|] ([|Attr.int("detect_negative", detect_negative, 1L); Attr.int("detect_positive", detect_positive, 1L)|] |> Array.choose id)
    static member is_inf(X: Tensor<double>, ?detect_negative: int64, ?detect_positive: int64) =
        MV() |> fun mv -> execNode<bool> "IsInf" [|mv.c(X)|] ([|Attr.int("detect_negative", detect_negative, 1L); Attr.int("detect_positive", detect_positive, 1L)|] |> Array.choose id)
    static member tf_idf_vectorizer(X: Tensor<string>, max_gram_length: int64, max_skip_count: int64, min_gram_length: int64, mode: string, ngram_counts: int64[], ngram_indexes: int64[], ?pool_int64s: int64[], ?pool_strings: string[], ?weights: float32[]) =
        MV() |> fun mv -> execNode<float32> "TfIdfVectorizer" [|mv.c(X)|] ([|Attr.int("max_gram_length", max_gram_length); Attr.int("max_skip_count", max_skip_count); Attr.int("min_gram_length", min_gram_length); Attr.string("mode", mode); Attr.ints("ngram_counts", ngram_counts); Attr.ints("ngram_indexes", ngram_indexes); Attr.ints("pool_int64s", pool_int64s); Attr.strings("pool_strings", pool_strings); Attr.floats("weights", weights)|] |> Array.choose id)
    static member tf_idf_vectorizer(X: Tensor<int>, max_gram_length: int64, max_skip_count: int64, min_gram_length: int64, mode: string, ngram_counts: int64[], ngram_indexes: int64[], ?pool_int64s: int64[], ?pool_strings: string[], ?weights: float32[]) =
        MV() |> fun mv -> execNode<float32> "TfIdfVectorizer" [|mv.c(X)|] ([|Attr.int("max_gram_length", max_gram_length); Attr.int("max_skip_count", max_skip_count); Attr.int("min_gram_length", min_gram_length); Attr.string("mode", mode); Attr.ints("ngram_counts", ngram_counts); Attr.ints("ngram_indexes", ngram_indexes); Attr.ints("pool_int64s", pool_int64s); Attr.strings("pool_strings", pool_strings); Attr.floats("weights", weights)|] |> Array.choose id)
    static member tf_idf_vectorizer(X: Tensor<int64>, max_gram_length: int64, max_skip_count: int64, min_gram_length: int64, mode: string, ngram_counts: int64[], ngram_indexes: int64[], ?pool_int64s: int64[], ?pool_strings: string[], ?weights: float32[]) =
        MV() |> fun mv -> execNode<float32> "TfIdfVectorizer" [|mv.c(X)|] ([|Attr.int("max_gram_length", max_gram_length); Attr.int("max_skip_count", max_skip_count); Attr.int("min_gram_length", min_gram_length); Attr.string("mode", mode); Attr.ints("ngram_counts", ngram_counts); Attr.ints("ngram_indexes", ngram_indexes); Attr.ints("pool_int64s", pool_int64s); Attr.strings("pool_strings", pool_strings); Attr.floats("weights", weights)|] |> Array.choose id)
    static member shape(data: Tensor<uint8>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member shape(data: Tensor<uint16>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member shape(data: Tensor<uint32>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member shape(data: Tensor<uint64>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member shape(data: Tensor<int8>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member shape(data: Tensor<int16>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member shape(data: Tensor<int>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member shape(data: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member shape(data: Tensor<float32>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member shape(data: Tensor<double>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member shape(data: Tensor<string>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member shape(data: Tensor<bool>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member shape(data: Tensor<Complex>) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] [||]
    static member greater(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member greater(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member greater(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member greater(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member greater(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member greater(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member greater(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member greater(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member greater(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member greater(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member equal(A: Tensor<bool>, B: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member equal(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member equal(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member equal(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member equal(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member equal(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member equal(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member equal(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member equal(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member equal(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member equal(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member and_(A: Tensor<bool>, B: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "And" [|mv.c(A); mv.c(B)|] [||]
    static member size(data: Tensor<uint8>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member size(data: Tensor<uint16>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member size(data: Tensor<uint32>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member size(data: Tensor<uint64>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member size(data: Tensor<int8>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member size(data: Tensor<int16>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member size(data: Tensor<int>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member size(data: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member size(data: Tensor<float32>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member size(data: Tensor<double>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member size(data: Tensor<string>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member size(data: Tensor<bool>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member size(data: Tensor<Complex>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member is_nan(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "IsNaN" [|mv.c(X)|] [||]
    static member is_nan(X: Tensor<double>) =
        MV() |> fun mv -> execNode<bool> "IsNaN" [|mv.c(X)|] [||]
    static member less(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member less(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member less(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member less(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member less(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member less(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member less(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member less(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member less(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member less(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member xor(A: Tensor<bool>, B: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Xor" [|mv.c(A); mv.c(B)|] [||]
    static member or_(A: Tensor<bool>, B: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Or" [|mv.c(A); mv.c(B)|] [||]
    static member cum_sum(x: Tensor<uint32>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<uint32> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member cum_sum(x: Tensor<uint32>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<uint32> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member cum_sum(x: Tensor<uint64>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<uint64> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member cum_sum(x: Tensor<uint64>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<uint64> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member cum_sum(x: Tensor<int>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<int> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member cum_sum(x: Tensor<int>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<int> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member cum_sum(x: Tensor<int64>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<int64> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member cum_sum(x: Tensor<int64>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<int64> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member cum_sum(x: Tensor<float32>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<float32> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member cum_sum(x: Tensor<float32>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<float32> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member cum_sum(x: Tensor<double>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<double> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member cum_sum(x: Tensor<double>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<double> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member roi_align(X: Tensor<float32>, rois: Tensor<float32>, batch_indices: Tensor<int64>, ?mode: string, ?output_height: int64, ?output_width: int64, ?sampling_ratio: int64, ?spatial_scale: float32) =
        MV() |> fun mv -> execNode<float32> "RoiAlign" [|mv.c(X); mv.c(rois); mv.c(batch_indices)|] ([|Attr.string("mode", mode, "avg"); Attr.int("output_height", output_height, 1L); Attr.int("output_width", output_width, 1L); Attr.int("sampling_ratio", sampling_ratio, 0L); Attr.float("spatial_scale", spatial_scale, 1.0f)|] |> Array.choose id)
    static member roi_align(X: Tensor<double>, rois: Tensor<double>, batch_indices: Tensor<int64>, ?mode: string, ?output_height: int64, ?output_width: int64, ?sampling_ratio: int64, ?spatial_scale: float32) =
        MV() |> fun mv -> execNode<double> "RoiAlign" [|mv.c(X); mv.c(rois); mv.c(batch_indices)|] ([|Attr.string("mode", mode, "avg"); Attr.int("output_height", output_height, 1L); Attr.int("output_width", output_width, 1L); Attr.int("sampling_ratio", sampling_ratio, 0L); Attr.float("spatial_scale", spatial_scale, 1.0f)|] |> Array.choose id)
    static member q_linear_conv(x: Tensor<int8>, x_scale: Tensor<float32>, x_zero_point: Tensor<int8>, w: Tensor<int8>, w_scale: Tensor<float32>, w_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member q_linear_conv(x: Tensor<int8>, x_scale: Tensor<float32>, x_zero_point: Tensor<int8>, w: Tensor<int8>, w_scale: Tensor<float32>, w_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<uint8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member q_linear_conv(x: Tensor<int8>, x_scale: Tensor<float32>, x_zero_point: Tensor<int8>, w: Tensor<uint8>, w_scale: Tensor<float32>, w_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member q_linear_conv(x: Tensor<int8>, x_scale: Tensor<float32>, x_zero_point: Tensor<int8>, w: Tensor<uint8>, w_scale: Tensor<float32>, w_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<uint8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member q_linear_conv(x: Tensor<uint8>, x_scale: Tensor<float32>, x_zero_point: Tensor<uint8>, w: Tensor<int8>, w_scale: Tensor<float32>, w_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member q_linear_conv(x: Tensor<uint8>, x_scale: Tensor<float32>, x_zero_point: Tensor<uint8>, w: Tensor<int8>, w_scale: Tensor<float32>, w_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<uint8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member q_linear_conv(x: Tensor<uint8>, x_scale: Tensor<float32>, x_zero_point: Tensor<uint8>, w: Tensor<uint8>, w_scale: Tensor<float32>, w_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member q_linear_conv(x: Tensor<uint8>, x_scale: Tensor<float32>, x_zero_point: Tensor<uint8>, w: Tensor<uint8>, w_scale: Tensor<float32>, w_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<uint8> "QLinearConv" ([|Some(mv.c(x)); Some(mv.c(x_scale)); Some(mv.c(x_zero_point)); Some(mv.c(w)); Some(mv.c(w_scale)); Some(mv.c(w_zero_point)); Some(mv.c(y_scale)); Some(mv.c(y_zero_point)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member conv_integer(x: Tensor<int8>, w: Tensor<int8>, ?x_zero_point: Tensor<int8>, ?w_zero_point: Tensor<int8>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int> "ConvInteger" ([|Some(mv.c(x)); Some(mv.c(w)); mv.c(x_zero_point); mv.c(w_zero_point)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member conv_integer(x: Tensor<int8>, w: Tensor<uint8>, ?x_zero_point: Tensor<int8>, ?w_zero_point: Tensor<uint8>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int> "ConvInteger" ([|Some(mv.c(x)); Some(mv.c(w)); mv.c(x_zero_point); mv.c(w_zero_point)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member conv_integer(x: Tensor<uint8>, w: Tensor<int8>, ?x_zero_point: Tensor<uint8>, ?w_zero_point: Tensor<int8>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int> "ConvInteger" ([|Some(mv.c(x)); Some(mv.c(w)); mv.c(x_zero_point); mv.c(w_zero_point)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member conv_integer(x: Tensor<uint8>, w: Tensor<uint8>, ?x_zero_point: Tensor<uint8>, ?w_zero_point: Tensor<uint8>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int> "ConvInteger" ([|Some(mv.c(x)); Some(mv.c(w)); mv.c(x_zero_point); mv.c(w_zero_point)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member q_linear_mat_mul(a: Tensor<int8>, a_scale: Tensor<float32>, a_zero_point: Tensor<int8>, b: Tensor<int8>, b_scale: Tensor<float32>, b_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member q_linear_mat_mul(a: Tensor<int8>, a_scale: Tensor<float32>, a_zero_point: Tensor<int8>, b: Tensor<int8>, b_scale: Tensor<float32>, b_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member q_linear_mat_mul(a: Tensor<int8>, a_scale: Tensor<float32>, a_zero_point: Tensor<int8>, b: Tensor<uint8>, b_scale: Tensor<float32>, b_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member q_linear_mat_mul(a: Tensor<int8>, a_scale: Tensor<float32>, a_zero_point: Tensor<int8>, b: Tensor<uint8>, b_scale: Tensor<float32>, b_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member q_linear_mat_mul(a: Tensor<uint8>, a_scale: Tensor<float32>, a_zero_point: Tensor<uint8>, b: Tensor<int8>, b_scale: Tensor<float32>, b_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member q_linear_mat_mul(a: Tensor<uint8>, a_scale: Tensor<float32>, a_zero_point: Tensor<uint8>, b: Tensor<int8>, b_scale: Tensor<float32>, b_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member q_linear_mat_mul(a: Tensor<uint8>, a_scale: Tensor<float32>, a_zero_point: Tensor<uint8>, b: Tensor<uint8>, b_scale: Tensor<float32>, b_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member q_linear_mat_mul(a: Tensor<uint8>, a_scale: Tensor<float32>, a_zero_point: Tensor<uint8>, b: Tensor<uint8>, b_scale: Tensor<float32>, b_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "QLinearMatMul" [|mv.c(a); mv.c(a_scale); mv.c(a_zero_point); mv.c(b); mv.c(b_scale); mv.c(b_zero_point); mv.c(y_scale); mv.c(y_zero_point)|] [||]
    static member where(condition: Tensor<bool>, X: Tensor<uint8>, Y: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member where(condition: Tensor<bool>, X: Tensor<uint16>, Y: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member where(condition: Tensor<bool>, X: Tensor<uint32>, Y: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member where(condition: Tensor<bool>, X: Tensor<uint64>, Y: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member where(condition: Tensor<bool>, X: Tensor<int8>, Y: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member where(condition: Tensor<bool>, X: Tensor<int16>, Y: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member where(condition: Tensor<bool>, X: Tensor<int>, Y: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member where(condition: Tensor<bool>, X: Tensor<int64>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member where(condition: Tensor<bool>, X: Tensor<float32>, Y: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member where(condition: Tensor<bool>, X: Tensor<double>, Y: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member where(condition: Tensor<bool>, X: Tensor<string>, Y: Tensor<string>) =
        MV() |> fun mv -> execNode<string> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member where(condition: Tensor<bool>, X: Tensor<bool>, Y: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member where(condition: Tensor<bool>, X: Tensor<Complex>, Y: Tensor<Complex>) =
        MV() |> fun mv -> execNode<Complex> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member max_unpool(X: Tensor<float32>, I: Tensor<int64>, kernel_shape: int64[], ?output_shape: Tensor<int64>, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "MaxUnpool" ([|Some(mv.c(X)); Some(mv.c(I)); mv.c(output_shape)|] |> Array.choose id) ([|Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member max_unpool(X: Tensor<double>, I: Tensor<int64>, kernel_shape: int64[], ?output_shape: Tensor<int64>, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<double> "MaxUnpool" ([|Some(mv.c(X)); Some(mv.c(I)); mv.c(output_shape)|] |> Array.choose id) ([|Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member gather_elements(data: Tensor<uint8>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<uint8>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<uint16>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<uint16>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<uint32>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<uint32>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<uint64>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<uint64>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<int8>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<int8>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<int16>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<int16>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<int>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<int>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<int64>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<int64>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<float32>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<float32>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<double>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<double>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<string>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<string>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<bool>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<bool>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<Complex>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather_elements(data: Tensor<Complex>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member quantize_linear(x: Tensor<float32>, y_scale: Tensor<float32>, ?y_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "QuantizeLinear" ([|Some(mv.c(x)); Some(mv.c(y_scale)); mv.c(y_zero_point)|] |> Array.choose id) [||]
    static member quantize_linear(x: Tensor<float32>, y_scale: Tensor<float32>, ?y_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "QuantizeLinear" ([|Some(mv.c(x)); Some(mv.c(y_scale)); mv.c(y_zero_point)|] |> Array.choose id) [||]
    static member quantize_linear(x: Tensor<int>, y_scale: Tensor<float32>, ?y_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "QuantizeLinear" ([|Some(mv.c(x)); Some(mv.c(y_scale)); mv.c(y_zero_point)|] |> Array.choose id) [||]
    static member quantize_linear(x: Tensor<int>, y_scale: Tensor<float32>, ?y_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "QuantizeLinear" ([|Some(mv.c(x)); Some(mv.c(y_scale)); mv.c(y_zero_point)|] |> Array.choose id) [||]
    static member resize(X: Tensor<uint8>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint8> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<uint8>, roi: Tensor<double>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint8> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<uint16>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint16> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<uint16>, roi: Tensor<double>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint16> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<uint32>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint32> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<uint32>, roi: Tensor<double>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint32> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<uint64>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint64> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<uint64>, roi: Tensor<double>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint64> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<int8>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int8> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<int8>, roi: Tensor<double>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int8> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<int16>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int16> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<int16>, roi: Tensor<double>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int16> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<int>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<int>, roi: Tensor<double>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<int64>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int64> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<int64>, roi: Tensor<double>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int64> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<float32>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<float32> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<float32>, roi: Tensor<double>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<float32> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<double>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<double> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<double>, roi: Tensor<double>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<double> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<string>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<string> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<string>, roi: Tensor<double>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<string> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<bool>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<bool> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<bool>, roi: Tensor<double>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<bool> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<Complex>, roi: Tensor<float32>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<Complex> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member resize(X: Tensor<Complex>, roi: Tensor<double>, scales: Tensor<float32>, ?sizes: Tensor<int64>, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<Complex> "Resize" ([|Some(mv.c(X)); Some(mv.c(roi)); Some(mv.c(scales)); mv.c(sizes)|] |> Array.choose id) ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member mat_mul_integer(A: Tensor<int8>, B: Tensor<int8>, ?a_zero_point: Tensor<int8>, ?b_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int> "MatMulInteger" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(a_zero_point); mv.c(b_zero_point)|] |> Array.choose id) [||]
    static member mat_mul_integer(A: Tensor<int8>, B: Tensor<uint8>, ?a_zero_point: Tensor<int8>, ?b_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<int> "MatMulInteger" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(a_zero_point); mv.c(b_zero_point)|] |> Array.choose id) [||]
    static member mat_mul_integer(A: Tensor<uint8>, B: Tensor<int8>, ?a_zero_point: Tensor<uint8>, ?b_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int> "MatMulInteger" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(a_zero_point); mv.c(b_zero_point)|] |> Array.choose id) [||]
    static member mat_mul_integer(A: Tensor<uint8>, B: Tensor<uint8>, ?a_zero_point: Tensor<uint8>, ?b_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<int> "MatMulInteger" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(a_zero_point); mv.c(b_zero_point)|] |> Array.choose id) [||]
    static member compress(input: Tensor<uint8>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member compress(input: Tensor<uint16>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member compress(input: Tensor<uint32>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member compress(input: Tensor<uint64>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member compress(input: Tensor<int8>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member compress(input: Tensor<int16>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member compress(input: Tensor<int>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member compress(input: Tensor<int64>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member compress(input: Tensor<float32>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member compress(input: Tensor<double>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member compress(input: Tensor<string>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member compress(input: Tensor<bool>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member compress(input: Tensor<Complex>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member gather(data: Tensor<uint8>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<uint8>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<uint16>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<uint16>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<uint32>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<uint32>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<uint64>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<uint64>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<int8>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<int8>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<int16>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<int16>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<int>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<int>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<int64>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<int64>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<float32>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<float32>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<double>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<double>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<string>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<string>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<bool>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<bool>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<Complex>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member gather(data: Tensor<Complex>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<uint8>, indices: Tensor<int>, updates: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<uint8>, indices: Tensor<int64>, updates: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<uint16>, indices: Tensor<int>, updates: Tensor<uint16>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<uint16>, indices: Tensor<int64>, updates: Tensor<uint16>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<uint32>, indices: Tensor<int>, updates: Tensor<uint32>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<uint32>, indices: Tensor<int64>, updates: Tensor<uint32>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<uint64>, indices: Tensor<int>, updates: Tensor<uint64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<uint64>, indices: Tensor<int64>, updates: Tensor<uint64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<int8>, indices: Tensor<int>, updates: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<int8>, indices: Tensor<int64>, updates: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<int16>, indices: Tensor<int>, updates: Tensor<int16>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<int16>, indices: Tensor<int64>, updates: Tensor<int16>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<int>, indices: Tensor<int>, updates: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<int>, indices: Tensor<int64>, updates: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<int64>, indices: Tensor<int>, updates: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<int64>, indices: Tensor<int64>, updates: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<float32>, indices: Tensor<int>, updates: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<float32>, indices: Tensor<int64>, updates: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<double>, indices: Tensor<int>, updates: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<double>, indices: Tensor<int64>, updates: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<string>, indices: Tensor<int>, updates: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<string>, indices: Tensor<int64>, updates: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<bool>, indices: Tensor<int>, updates: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<bool>, indices: Tensor<int64>, updates: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<Complex>, indices: Tensor<int>, updates: Tensor<Complex>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter_elements(data: Tensor<Complex>, indices: Tensor<int64>, updates: Tensor<Complex>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member slice(data: Tensor<uint8>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<uint8> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<uint8>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<uint16>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<uint16> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<uint16>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint16> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<uint32>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<uint32> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<uint32>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint32> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<uint64>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<uint64> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<uint64>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint64> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<int8>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<int8> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<int8>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<int16>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<int16> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<int16>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<int16> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<int>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<int>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<int64>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<int64> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<int64>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<float32>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<float32> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<float32>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<double>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<double> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<double>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<string>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<string> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<string>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<bool>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<bool>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<Complex>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<Complex> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member slice(data: Tensor<Complex>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<Complex> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member tile(input: Tensor<uint8>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member tile(input: Tensor<uint16>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint16> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member tile(input: Tensor<uint32>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint32> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member tile(input: Tensor<uint64>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint64> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member tile(input: Tensor<int8>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member tile(input: Tensor<int16>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<int16> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member tile(input: Tensor<int>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member tile(input: Tensor<int64>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member tile(input: Tensor<float32>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member tile(input: Tensor<double>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member tile(input: Tensor<string>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member tile(input: Tensor<bool>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member tile(input: Tensor<Complex>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<Complex> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member scatter(data: Tensor<uint8>, indices: Tensor<int>, updates: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<uint8>, indices: Tensor<int64>, updates: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<uint16>, indices: Tensor<int>, updates: Tensor<uint16>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<uint16>, indices: Tensor<int64>, updates: Tensor<uint16>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<uint32>, indices: Tensor<int>, updates: Tensor<uint32>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<uint32>, indices: Tensor<int64>, updates: Tensor<uint32>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<uint64>, indices: Tensor<int>, updates: Tensor<uint64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<uint64>, indices: Tensor<int64>, updates: Tensor<uint64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<int8>, indices: Tensor<int>, updates: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<int8>, indices: Tensor<int64>, updates: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<int16>, indices: Tensor<int>, updates: Tensor<int16>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<int16>, indices: Tensor<int64>, updates: Tensor<int16>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<int>, indices: Tensor<int>, updates: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<int>, indices: Tensor<int64>, updates: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<int64>, indices: Tensor<int>, updates: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<int64>, indices: Tensor<int64>, updates: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<float32>, indices: Tensor<int>, updates: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<float32>, indices: Tensor<int64>, updates: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<double>, indices: Tensor<int>, updates: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<double>, indices: Tensor<int64>, updates: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<string>, indices: Tensor<int>, updates: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<string>, indices: Tensor<int64>, updates: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<bool>, indices: Tensor<int>, updates: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<bool>, indices: Tensor<int64>, updates: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<Complex>, indices: Tensor<int>, updates: Tensor<Complex>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member scatter(data: Tensor<Complex>, indices: Tensor<int64>, updates: Tensor<Complex>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member non_max_suppression(boxes: Tensor<float32>, scores: Tensor<float32>, ?max_output_boxes_per_class: Tensor<int64>, ?iou_threshold: Tensor<float32>, ?score_threshold: Tensor<float32>, ?center_point_box: int64) =
        MV() |> fun mv -> execNode<int64> "NonMaxSuppression" ([|Some(mv.c(boxes)); Some(mv.c(scores)); mv.c(max_output_boxes_per_class); mv.c(iou_threshold); mv.c(score_threshold)|] |> Array.choose id) ([|Attr.int("center_point_box", center_point_box, 0L)|] |> Array.choose id)
    static member string_normalizer(X: Tensor<string>, ?case_change_action: string, ?is_case_sensitive: int64, ?locale: string, ?stopwords: string[]) =
        MV() |> fun mv -> execNode<string> "StringNormalizer" [|mv.c(X)|] ([|Attr.string("case_change_action", case_change_action, "NONE"); Attr.int("is_case_sensitive", is_case_sensitive, 0L); Attr.string("locale", locale); Attr.strings("stopwords", stopwords)|] |> Array.choose id)
    static member label_encoder(X: Tensor<string>, ?default_float: float32, ?default_int64: int64, ?default_string: string, ?keys_floats: float32[], ?keys_int64s: int64[], ?keys_strings: string[], ?values_floats: float32[], ?values_int64s: int64[], ?values_strings: string[]) =
        MV() |> fun mv -> execNode<string> "LabelEncoder" [|mv.c(X)|] ([|Attr.float("default_float", default_float, -0.0f); Attr.int("default_int64", default_int64, -1L); Attr.string("default_string", default_string, "_Unused"); Attr.floats("keys_floats", keys_floats); Attr.ints("keys_int64s", keys_int64s); Attr.strings("keys_strings", keys_strings); Attr.floats("values_floats", values_floats); Attr.ints("values_int64s", values_int64s); Attr.strings("values_strings", values_strings)|] |> Array.choose id)
    static member label_encoder(X: Tensor<int64>, ?default_float: float32, ?default_int64: int64, ?default_string: string, ?keys_floats: float32[], ?keys_int64s: int64[], ?keys_strings: string[], ?values_floats: float32[], ?values_int64s: int64[], ?values_strings: string[]) =
        MV() |> fun mv -> execNode<int64> "LabelEncoder" [|mv.c(X)|] ([|Attr.float("default_float", default_float, -0.0f); Attr.int("default_int64", default_int64, -1L); Attr.string("default_string", default_string, "_Unused"); Attr.floats("keys_floats", keys_floats); Attr.ints("keys_int64s", keys_int64s); Attr.strings("keys_strings", keys_strings); Attr.floats("values_floats", values_floats); Attr.ints("values_int64s", values_int64s); Attr.strings("values_strings", values_strings)|] |> Array.choose id)
    static member label_encoder(X: Tensor<float32>, ?default_float: float32, ?default_int64: int64, ?default_string: string, ?keys_floats: float32[], ?keys_int64s: int64[], ?keys_strings: string[], ?values_floats: float32[], ?values_int64s: int64[], ?values_strings: string[]) =
        MV() |> fun mv -> execNode<float32> "LabelEncoder" [|mv.c(X)|] ([|Attr.float("default_float", default_float, -0.0f); Attr.int("default_int64", default_int64, -1L); Attr.string("default_string", default_string, "_Unused"); Attr.floats("keys_floats", keys_floats); Attr.ints("keys_int64s", keys_int64s); Attr.strings("keys_strings", keys_strings); Attr.floats("values_floats", values_floats); Attr.ints("values_int64s", values_int64s); Attr.strings("values_strings", values_strings)|] |> Array.choose id)
    static member category_mapper(X: Tensor<string>, ?cats_int64s: int64[], ?cats_strings: string[], ?default_int64: int64, ?default_string: string) =
        MV() |> fun mv -> execNode<string> "CategoryMapper" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("default_int64", default_int64, -1L); Attr.string("default_string", default_string, "_Unused")|] |> Array.choose id)
    static member category_mapper(X: Tensor<int64>, ?cats_int64s: int64[], ?cats_strings: string[], ?default_int64: int64, ?default_string: string) =
        MV() |> fun mv -> execNode<int64> "CategoryMapper" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("default_int64", default_int64, -1L); Attr.string("default_string", default_string, "_Unused")|] |> Array.choose id)
    static member sequence_empty<'a>() =
        execNodeCheck<'a> "SequenceEmpty" [||] [||] [||]
    static member eye_like<'a>(input: Tensor<bool>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like(input: Tensor<bool>, ?k: int64) =
        execNodeCheck<bool> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like<'a>(input: Tensor<uint8>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like(input: Tensor<uint8>, ?k: int64) =
        execNodeCheck<uint8> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like<'a>(input: Tensor<int8>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like(input: Tensor<int8>, ?k: int64) =
        execNodeCheck<int8> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like<'a>(input: Tensor<uint16>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like(input: Tensor<uint16>, ?k: int64) =
        execNodeCheck<uint16> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like<'a>(input: Tensor<int>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like(input: Tensor<int>, ?k: int64) =
        execNodeCheck<int> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like<'a>(input: Tensor<int16>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like(input: Tensor<int16>, ?k: int64) =
        execNodeCheck<int16> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like<'a>(input: Tensor<float32>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like(input: Tensor<float32>, ?k: int64) =
        execNodeCheck<float32> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like<'a>(input: Tensor<int64>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like(input: Tensor<int64>, ?k: int64) =
        execNodeCheck<int64> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like<'a>(input: Tensor<double>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like(input: Tensor<double>, ?k: int64) =
        execNodeCheck<double> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like<'a>(input: Tensor<uint64>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like(input: Tensor<uint64>, ?k: int64) =
        execNodeCheck<uint64> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like<'a>(input: Tensor<uint32>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member eye_like(input: Tensor<uint32>, ?k: int64) =
        execNodeCheck<uint32> "EyeLike" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 1L; 7L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member multinomial<'a>(input: Tensor<double>, ?sample_size: int64, ?seed: float32) =
        execNodeCheck<'a> "Multinomial" [|MV.mv(1,input)|] [|7L; 6L|] ([|Attr.int("sample_size", sample_size, 1L); Attr.float("seed", seed)|] |> Array.choose id)
    static member multinomial(input: Tensor<double>, ?sample_size: int64, ?seed: float32) =
        execNodeCheck<double> "Multinomial" [|MV.mv(1,input)|] [|7L; 6L|] ([|Attr.int("sample_size", sample_size, 1L); Attr.float("seed", seed)|] |> Array.choose id)
    static member multinomial<'a>(input: Tensor<float32>, ?sample_size: int64, ?seed: float32) =
        execNodeCheck<'a> "Multinomial" [|MV.mv(1,input)|] [|7L; 6L|] ([|Attr.int("sample_size", sample_size, 1L); Attr.float("seed", seed)|] |> Array.choose id)
    static member multinomial(input: Tensor<float32>, ?sample_size: int64, ?seed: float32) =
        execNodeCheck<float32> "Multinomial" [|MV.mv(1,input)|] [|7L; 6L|] ([|Attr.int("sample_size", sample_size, 1L); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like<'a>(input: Tensor<bool>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like(input: Tensor<bool>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<bool> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like<'a>(input: Tensor<uint8>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like(input: Tensor<uint8>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<uint8> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like<'a>(input: Tensor<int8>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like(input: Tensor<int8>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<int8> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like<'a>(input: Tensor<uint16>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like(input: Tensor<uint16>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<uint16> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like<'a>(input: Tensor<int>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like(input: Tensor<int>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<int> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like<'a>(input: Tensor<int16>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like(input: Tensor<int16>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<int16> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like<'a>(input: Tensor<double>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like(input: Tensor<double>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<double> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like<'a>(input: Tensor<string>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like(input: Tensor<string>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<string> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like<'a>(input: Tensor<float32>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like(input: Tensor<float32>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<float32> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like<'a>(input: Tensor<int64>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like(input: Tensor<int64>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<int64> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like<'a>(input: Tensor<Complex>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like(input: Tensor<Complex>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<Complex> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like<'a>(input: Tensor<uint64>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like(input: Tensor<uint64>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<uint64> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like<'a>(input: Tensor<uint32>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform_like(input: Tensor<uint32>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<uint32> "RandomUniformLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like<'a>(input: Tensor<bool>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like(input: Tensor<bool>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<bool> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like<'a>(input: Tensor<uint8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like(input: Tensor<uint8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<uint8> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like<'a>(input: Tensor<int8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like(input: Tensor<int8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<int8> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like<'a>(input: Tensor<uint16>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like(input: Tensor<uint16>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<uint16> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like<'a>(input: Tensor<int>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like(input: Tensor<int>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<int> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like<'a>(input: Tensor<int16>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like(input: Tensor<int16>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<int16> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like<'a>(input: Tensor<double>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like(input: Tensor<double>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<double> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like<'a>(input: Tensor<string>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like(input: Tensor<string>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<string> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like<'a>(input: Tensor<float32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like(input: Tensor<float32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<float32> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like<'a>(input: Tensor<int64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like(input: Tensor<int64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<int64> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like<'a>(input: Tensor<Complex>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like(input: Tensor<Complex>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<Complex> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like<'a>(input: Tensor<uint64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like(input: Tensor<uint64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<uint64> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like<'a>(input: Tensor<uint32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal_like(input: Tensor<uint32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<uint32> "RandomNormalLike" [|MV.mv(1,input)|] [|1L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_normal<'a>(shape: int64[], ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormal" [||] [|1L|] ([|Attr.ints("shape", shape); Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member random_uniform<'a>(shape: int64[], ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniform" [||] [|1L|] ([|Attr.ints("shape", shape); Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member cast<'a>(input: Tensor<bool>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 8L; 1L; 7L|] [||]
    static member cast<'a>(input: Tensor<uint8>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 8L; 1L; 7L|] [||]
    static member cast<'a>(input: Tensor<int8>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 8L; 1L; 7L|] [||]
    static member cast<'a>(input: Tensor<uint16>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 8L; 1L; 7L|] [||]
    static member cast<'a>(input: Tensor<int>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 8L; 1L; 7L|] [||]
    static member cast<'a>(input: Tensor<int16>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 8L; 1L; 7L|] [||]
    static member cast<'a>(input: Tensor<string>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 8L; 1L; 7L|] [||]
    static member cast<'a>(input: Tensor<float32>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 8L; 1L; 7L|] [||]
    static member cast<'a>(input: Tensor<int64>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 8L; 1L; 7L|] [||]
    static member cast<'a>(input: Tensor<double>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 8L; 1L; 7L|] [||]
    static member cast<'a>(input: Tensor<uint64>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 8L; 1L; 7L|] [||]
    static member cast<'a>(input: Tensor<uint32>) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|9L; 2L; 3L; 6L; 8L; 1L; 7L|] [||]
    static member tree_ensemble_classifier(X: Tensor<float32>, ?base_values: float32[], ?class_ids: int64[], ?class_nodeids: int64[], ?class_treeids: int64[], ?class_weights: float32[], ?classlabels_int64s: int64[], ?classlabels_strings: string[], ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "TreeEnsembleClassifier" [|mv.c(X)|] ([|Attr.floats("base_values", base_values); Attr.ints("class_ids", class_ids); Attr.ints("class_nodeids", class_nodeids); Attr.ints("class_treeids", class_treeids); Attr.floats("class_weights", class_weights); Attr.ints("classlabels_int64s", classlabels_int64s); Attr.strings("classlabels_strings", classlabels_strings); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member tree_ensemble_classifier(X: Tensor<double>, ?base_values: float32[], ?class_ids: int64[], ?class_nodeids: int64[], ?class_treeids: int64[], ?class_weights: float32[], ?classlabels_int64s: int64[], ?classlabels_strings: string[], ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "TreeEnsembleClassifier" [|mv.c(X)|] ([|Attr.floats("base_values", base_values); Attr.ints("class_ids", class_ids); Attr.ints("class_nodeids", class_nodeids); Attr.ints("class_treeids", class_treeids); Attr.floats("class_weights", class_weights); Attr.ints("classlabels_int64s", classlabels_int64s); Attr.strings("classlabels_strings", classlabels_strings); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member tree_ensemble_classifier(X: Tensor<int64>, ?base_values: float32[], ?class_ids: int64[], ?class_nodeids: int64[], ?class_treeids: int64[], ?class_weights: float32[], ?classlabels_int64s: int64[], ?classlabels_strings: string[], ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "TreeEnsembleClassifier" [|mv.c(X)|] ([|Attr.floats("base_values", base_values); Attr.ints("class_ids", class_ids); Attr.ints("class_nodeids", class_nodeids); Attr.ints("class_treeids", class_treeids); Attr.floats("class_weights", class_weights); Attr.ints("classlabels_int64s", classlabels_int64s); Attr.strings("classlabels_strings", classlabels_strings); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member tree_ensemble_classifier(X: Tensor<int>, ?base_values: float32[], ?class_ids: int64[], ?class_nodeids: int64[], ?class_treeids: int64[], ?class_weights: float32[], ?classlabels_int64s: int64[], ?classlabels_strings: string[], ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "TreeEnsembleClassifier" [|mv.c(X)|] ([|Attr.floats("base_values", base_values); Attr.ints("class_ids", class_ids); Attr.ints("class_nodeids", class_nodeids); Attr.ints("class_treeids", class_treeids); Attr.floats("class_weights", class_weights); Attr.ints("classlabels_int64s", classlabels_int64s); Attr.strings("classlabels_strings", classlabels_strings); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member lstm(X: Tensor<float32>, W: Tensor<float32>, R: Tensor<float32>, ?B: Tensor<float32>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<float32>, ?initial_c: Tensor<float32>, ?P: Tensor<float32>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?input_forget: int64) =
        MV() |> fun mv -> execNodeTuple3<float32, float32, float32> "LSTM" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h); mv.c(initial_c); mv.c(P)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("input_forget", input_forget, 0L)|] |> Array.choose id)
    static member lstm(X: Tensor<double>, W: Tensor<double>, R: Tensor<double>, ?B: Tensor<double>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<double>, ?initial_c: Tensor<double>, ?P: Tensor<double>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?input_forget: int64) =
        MV() |> fun mv -> execNodeTuple3<double, double, double> "LSTM" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h); mv.c(initial_c); mv.c(P)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("input_forget", input_forget, 0L)|] |> Array.choose id)
    static member linear_classifier(X: Tensor<float32>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "LinearClassifier" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("intercepts", intercepts); Attr.int("multi_class", multi_class, 0L); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member linear_classifier(X: Tensor<double>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "LinearClassifier" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("intercepts", intercepts); Attr.int("multi_class", multi_class, 0L); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member linear_classifier(X: Tensor<int64>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "LinearClassifier" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("intercepts", intercepts); Attr.int("multi_class", multi_class, 0L); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member linear_classifier(X: Tensor<int>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "LinearClassifier" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("intercepts", intercepts); Attr.int("multi_class", multi_class, 0L); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member svm_classifier(X: Tensor<float32>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "SVMClassifier" [|mv.c(X)|] ([|Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("prob_a", prob_a); Attr.floats("prob_b", prob_b); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors); Attr.ints("vectors_per_class", vectors_per_class)|] |> Array.choose id)
    static member svm_classifier(X: Tensor<double>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "SVMClassifier" [|mv.c(X)|] ([|Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("prob_a", prob_a); Attr.floats("prob_b", prob_b); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors); Attr.ints("vectors_per_class", vectors_per_class)|] |> Array.choose id)
    static member svm_classifier(X: Tensor<int64>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "SVMClassifier" [|mv.c(X)|] ([|Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("prob_a", prob_a); Attr.floats("prob_b", prob_b); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors); Attr.ints("vectors_per_class", vectors_per_class)|] |> Array.choose id)
    static member svm_classifier(X: Tensor<int>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "SVMClassifier" [|mv.c(X)|] ([|Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("prob_a", prob_a); Attr.floats("prob_b", prob_b); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors); Attr.ints("vectors_per_class", vectors_per_class)|] |> Array.choose id)
    static member max_pool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?pads: int64[], ?storage_order: int64, ?strides: int64[]) =
        MV() |> fun mv -> execNodeTuple2<float32, int64> "MaxPool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.int("storage_order", storage_order, 0L); Attr.ints("strides", strides)|] |> Array.choose id)
    static member max_pool(X: Tensor<double>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?pads: int64[], ?storage_order: int64, ?strides: int64[]) =
        MV() |> fun mv -> execNodeTuple2<double, int64> "MaxPool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.int("storage_order", storage_order, 0L); Attr.ints("strides", strides)|] |> Array.choose id)
    static member gru(X: Tensor<float32>, W: Tensor<float32>, R: Tensor<float32>, ?B: Tensor<float32>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<float32>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?linear_before_reset: int64) =
        MV() |> fun mv -> execNodeTuple2<float32, float32> "GRU" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("linear_before_reset", linear_before_reset, 0L)|] |> Array.choose id)
    static member gru(X: Tensor<double>, W: Tensor<double>, R: Tensor<double>, ?B: Tensor<double>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<double>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?linear_before_reset: int64) =
        MV() |> fun mv -> execNodeTuple2<double, double> "GRU" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("linear_before_reset", linear_before_reset, 0L)|] |> Array.choose id)
    static member topk(X: Tensor<uint8>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<uint8, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member topk(X: Tensor<uint16>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<uint16, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member topk(X: Tensor<uint32>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<uint32, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member topk(X: Tensor<uint64>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<uint64, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member topk(X: Tensor<int8>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<int8, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member topk(X: Tensor<int16>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<int16, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member topk(X: Tensor<int>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<int, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member topk(X: Tensor<int64>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<int64, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member topk(X: Tensor<float32>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<float32, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member topk(X: Tensor<double>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<double, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member dropout(data: Tensor<float32>, ?ratio: float32) =
        MV() |> fun mv -> execNodeTuple2<float32, bool> "Dropout" [|mv.c(data)|] ([|Attr.float("ratio", ratio, 0.5f)|] |> Array.choose id)
    static member dropout(data: Tensor<double>, ?ratio: float32) =
        MV() |> fun mv -> execNodeTuple2<double, bool> "Dropout" [|mv.c(data)|] ([|Attr.float("ratio", ratio, 0.5f)|] |> Array.choose id)
    static member unique(X: Tensor<uint8>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<uint8, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member unique(X: Tensor<uint16>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<uint16, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member unique(X: Tensor<uint32>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<uint32, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member unique(X: Tensor<uint64>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<uint64, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member unique(X: Tensor<int8>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<int8, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member unique(X: Tensor<int16>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<int16, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member unique(X: Tensor<int>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<int, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member unique(X: Tensor<int64>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<int64, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member unique(X: Tensor<float32>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<float32, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member unique(X: Tensor<double>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<double, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member unique(X: Tensor<string>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<string, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member unique(X: Tensor<bool>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<bool, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member unique(X: Tensor<Complex>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<Complex, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member dynamic_quantize_linear(x: Tensor<float32>) =
        MV() |> fun mv -> execNodeTuple3<uint8, float32, uint8> "DynamicQuantizeLinear" [|mv.c(x)|] [||]
    static member rnn(X: Tensor<float32>, W: Tensor<float32>, R: Tensor<float32>, ?B: Tensor<float32>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<float32>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64) =
        MV() |> fun mv -> execNodeTuple2<float32, float32> "RNN" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations, [|"Tanh";"Tanh"|]); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size)|] |> Array.choose id)
    static member rnn(X: Tensor<double>, W: Tensor<double>, R: Tensor<double>, ?B: Tensor<double>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<double>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64) =
        MV() |> fun mv -> execNodeTuple2<double, double> "RNN" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations, [|"Tanh";"Tanh"|]); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size)|] |> Array.choose id)
    static member batch_normalization(X: Tensor<float32>, scale: Tensor<float32>, B: Tensor<float32>, mean: Tensor<float32>, var: Tensor<float32>, ?epsilon: float32, ?momentum: float32) =
        MV() |> fun mv -> execNodeTuple5<float32, float32, float32, float32, float32> "BatchNormalization" [|mv.c(X); mv.c(scale); mv.c(B); mv.c(mean); mv.c(var)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f); Attr.float("momentum", momentum, 0.8999999761581421f)|] |> Array.choose id)
    static member batch_normalization(X: Tensor<double>, scale: Tensor<double>, B: Tensor<double>, mean: Tensor<double>, var: Tensor<double>, ?epsilon: float32, ?momentum: float32) =
        MV() |> fun mv -> execNodeTuple5<double, double, double, double, double> "BatchNormalization" [|mv.c(X); mv.c(scale); mv.c(B); mv.c(mean); mv.c(var)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f); Attr.float("momentum", momentum, 0.8999999761581421f)|] |> Array.choose id)
