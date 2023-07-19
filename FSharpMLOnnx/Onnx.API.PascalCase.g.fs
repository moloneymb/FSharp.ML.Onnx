module FSharp.ML.Onnx.API.PascalCase

open System
open System.Numerics
open Microsoft.ML.OnnxRuntime.Tensors
open FSharp.ML.Onnx.Protobuf
type Onnx() =
    static member Scaler(X: Tensor<float32>, ?offset: float32[], ?scale: float32[]) =
        MV() |> fun mv -> execNode<float32> "Scaler" [|mv.c(X)|] ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member Scaler(X: Tensor<double>, ?offset: float32[], ?scale: float32[]) =
        MV() |> fun mv -> execNode<double> "Scaler" [|mv.c(X)|] ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member Scaler(X: Tensor<int64>, ?offset: float32[], ?scale: float32[]) =
        MV() |> fun mv -> execNode<int64> "Scaler" [|mv.c(X)|] ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member Scaler(X: Tensor<int>, ?offset: float32[], ?scale: float32[]) =
        MV() |> fun mv -> execNode<int> "Scaler" [|mv.c(X)|] ([|Attr.floats("offset", offset); Attr.floats("scale", scale)|] |> Array.choose id)
    static member SVMRegressor(X: Tensor<float32>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        MV() |> fun mv -> execNode<float32> "SVMRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member SVMRegressor(X: Tensor<double>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        MV() |> fun mv -> execNode<double> "SVMRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member SVMRegressor(X: Tensor<int64>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        MV() |> fun mv -> execNode<int64> "SVMRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member SVMRegressor(X: Tensor<int>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        MV() |> fun mv -> execNode<int> "SVMRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|] |> Array.choose id)
    static member BitwiseOr(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "BitwiseOr" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseOr(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "BitwiseOr" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseOr(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "BitwiseOr" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseOr(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "BitwiseOr" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseOr(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "BitwiseOr" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseOr(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "BitwiseOr" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseOr(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "BitwiseOr" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseOr(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "BitwiseOr" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseAnd(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "BitwiseAnd" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseAnd(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "BitwiseAnd" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseAnd(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "BitwiseAnd" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseAnd(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "BitwiseAnd" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseAnd(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "BitwiseAnd" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseAnd(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "BitwiseAnd" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseAnd(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "BitwiseAnd" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseAnd(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "BitwiseAnd" [|mv.c(A); mv.c(B)|] [||]
    static member Mish(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Mish" [|mv.c(X)|] [||]
    static member Mish(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Mish" [|mv.c(X)|] [||]
    static member Celu(X: Tensor<float32>, ?alpha: float32) =
        MV() |> fun mv -> execNode<float32> "Celu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.0f)|] |> Array.choose id)
    static member Imputer(X: Tensor<float32>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        MV() |> fun mv -> execNode<float32> "Imputer" [|mv.c(X)|] ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member Imputer(X: Tensor<double>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        MV() |> fun mv -> execNode<double> "Imputer" [|mv.c(X)|] ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member Imputer(X: Tensor<int64>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        MV() |> fun mv -> execNode<int64> "Imputer" [|mv.c(X)|] ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member Imputer(X: Tensor<int>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        MV() |> fun mv -> execNode<int> "Imputer" [|mv.c(X)|] ([|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<uint8>, indices: Tensor<int64>, ?batch_dims: int64) =
        MV() |> fun mv -> execNode<uint8> "GatherND" [|mv.c(data); mv.c(indices)|] ([|Attr.int("batch_dims", batch_dims, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<uint16>, indices: Tensor<int64>, ?batch_dims: int64) =
        MV() |> fun mv -> execNode<uint16> "GatherND" [|mv.c(data); mv.c(indices)|] ([|Attr.int("batch_dims", batch_dims, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<uint32>, indices: Tensor<int64>, ?batch_dims: int64) =
        MV() |> fun mv -> execNode<uint32> "GatherND" [|mv.c(data); mv.c(indices)|] ([|Attr.int("batch_dims", batch_dims, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<uint64>, indices: Tensor<int64>, ?batch_dims: int64) =
        MV() |> fun mv -> execNode<uint64> "GatherND" [|mv.c(data); mv.c(indices)|] ([|Attr.int("batch_dims", batch_dims, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<int8>, indices: Tensor<int64>, ?batch_dims: int64) =
        MV() |> fun mv -> execNode<int8> "GatherND" [|mv.c(data); mv.c(indices)|] ([|Attr.int("batch_dims", batch_dims, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<int16>, indices: Tensor<int64>, ?batch_dims: int64) =
        MV() |> fun mv -> execNode<int16> "GatherND" [|mv.c(data); mv.c(indices)|] ([|Attr.int("batch_dims", batch_dims, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<int>, indices: Tensor<int64>, ?batch_dims: int64) =
        MV() |> fun mv -> execNode<int> "GatherND" [|mv.c(data); mv.c(indices)|] ([|Attr.int("batch_dims", batch_dims, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<int64>, indices: Tensor<int64>, ?batch_dims: int64) =
        MV() |> fun mv -> execNode<int64> "GatherND" [|mv.c(data); mv.c(indices)|] ([|Attr.int("batch_dims", batch_dims, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<float32>, indices: Tensor<int64>, ?batch_dims: int64) =
        MV() |> fun mv -> execNode<float32> "GatherND" [|mv.c(data); mv.c(indices)|] ([|Attr.int("batch_dims", batch_dims, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<double>, indices: Tensor<int64>, ?batch_dims: int64) =
        MV() |> fun mv -> execNode<double> "GatherND" [|mv.c(data); mv.c(indices)|] ([|Attr.int("batch_dims", batch_dims, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<string>, indices: Tensor<int64>, ?batch_dims: int64) =
        MV() |> fun mv -> execNode<string> "GatherND" [|mv.c(data); mv.c(indices)|] ([|Attr.int("batch_dims", batch_dims, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<bool>, indices: Tensor<int64>, ?batch_dims: int64) =
        MV() |> fun mv -> execNode<bool> "GatherND" [|mv.c(data); mv.c(indices)|] ([|Attr.int("batch_dims", batch_dims, 0L)|] |> Array.choose id)
    static member GatherND(data: Tensor<Complex>, indices: Tensor<int64>, ?batch_dims: int64) =
        MV() |> fun mv -> execNode<Complex> "GatherND" [|mv.c(data); mv.c(indices)|] ([|Attr.int("batch_dims", batch_dims, 0L)|] |> Array.choose id)
    static member ScatterND(data: Tensor<uint8>, indices: Tensor<int64>, updates: Tensor<uint8>, ?reduction: string) =
        MV() |> fun mv -> execNode<uint8> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterND(data: Tensor<uint16>, indices: Tensor<int64>, updates: Tensor<uint16>, ?reduction: string) =
        MV() |> fun mv -> execNode<uint16> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterND(data: Tensor<uint32>, indices: Tensor<int64>, updates: Tensor<uint32>, ?reduction: string) =
        MV() |> fun mv -> execNode<uint32> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterND(data: Tensor<uint64>, indices: Tensor<int64>, updates: Tensor<uint64>, ?reduction: string) =
        MV() |> fun mv -> execNode<uint64> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterND(data: Tensor<int8>, indices: Tensor<int64>, updates: Tensor<int8>, ?reduction: string) =
        MV() |> fun mv -> execNode<int8> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterND(data: Tensor<int16>, indices: Tensor<int64>, updates: Tensor<int16>, ?reduction: string) =
        MV() |> fun mv -> execNode<int16> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterND(data: Tensor<int>, indices: Tensor<int64>, updates: Tensor<int>, ?reduction: string) =
        MV() |> fun mv -> execNode<int> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterND(data: Tensor<int64>, indices: Tensor<int64>, updates: Tensor<int64>, ?reduction: string) =
        MV() |> fun mv -> execNode<int64> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterND(data: Tensor<float32>, indices: Tensor<int64>, updates: Tensor<float32>, ?reduction: string) =
        MV() |> fun mv -> execNode<float32> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterND(data: Tensor<double>, indices: Tensor<int64>, updates: Tensor<double>, ?reduction: string) =
        MV() |> fun mv -> execNode<double> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterND(data: Tensor<string>, indices: Tensor<int64>, updates: Tensor<string>, ?reduction: string) =
        MV() |> fun mv -> execNode<string> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterND(data: Tensor<bool>, indices: Tensor<int64>, updates: Tensor<bool>, ?reduction: string) =
        MV() |> fun mv -> execNode<bool> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterND(data: Tensor<Complex>, indices: Tensor<int64>, updates: Tensor<Complex>, ?reduction: string) =
        MV() |> fun mv -> execNode<Complex> "ScatterND" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member Det(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Det" [|mv.c(X)|] [||]
    static member Det(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Det" [|mv.c(X)|] [||]
    static member Normalizer(X: Tensor<float32>, ?norm: string) =
        MV() |> fun mv -> execNode<float32> "Normalizer" [|mv.c(X)|] ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member Normalizer(X: Tensor<double>, ?norm: string) =
        MV() |> fun mv -> execNode<double> "Normalizer" [|mv.c(X)|] ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member Normalizer(X: Tensor<int64>, ?norm: string) =
        MV() |> fun mv -> execNode<int64> "Normalizer" [|mv.c(X)|] ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member Normalizer(X: Tensor<int>, ?norm: string) =
        MV() |> fun mv -> execNode<int> "Normalizer" [|mv.c(X)|] ([|Attr.string("norm", norm, "MAX")|] |> Array.choose id)
    static member FeatureVectorizer([<ParamArray>]X: Tensor<int>[], ?inputdimensions: int64[]) =
        MV() |> fun mv -> execNode<int> "FeatureVectorizer" (mv.c(X)) ([|Attr.ints("inputdimensions", inputdimensions)|] |> Array.choose id)
    static member FeatureVectorizer([<ParamArray>]X: Tensor<int64>[], ?inputdimensions: int64[]) =
        MV() |> fun mv -> execNode<int64> "FeatureVectorizer" (mv.c(X)) ([|Attr.ints("inputdimensions", inputdimensions)|] |> Array.choose id)
    static member FeatureVectorizer([<ParamArray>]X: Tensor<float32>[], ?inputdimensions: int64[]) =
        MV() |> fun mv -> execNode<float32> "FeatureVectorizer" (mv.c(X)) ([|Attr.ints("inputdimensions", inputdimensions)|] |> Array.choose id)
    static member FeatureVectorizer([<ParamArray>]X: Tensor<double>[], ?inputdimensions: int64[]) =
        MV() |> fun mv -> execNode<double> "FeatureVectorizer" (mv.c(X)) ([|Attr.ints("inputdimensions", inputdimensions)|] |> Array.choose id)
    static member Mul(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member Mul(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member Mul(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member Mul(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member Mul(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member Mul(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member Mul(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member Mul(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member Mul(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member Mul(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Mul" [|mv.c(A); mv.c(B)|] [||]
    static member Max([<ParamArray>]data_0: Tensor<uint8>[]) =
        MV() |> fun mv -> execNode<uint8> "Max" (mv.c(data_0)) [||]
    static member Max([<ParamArray>]data_0: Tensor<uint16>[]) =
        MV() |> fun mv -> execNode<uint16> "Max" (mv.c(data_0)) [||]
    static member Max([<ParamArray>]data_0: Tensor<uint32>[]) =
        MV() |> fun mv -> execNode<uint32> "Max" (mv.c(data_0)) [||]
    static member Max([<ParamArray>]data_0: Tensor<uint64>[]) =
        MV() |> fun mv -> execNode<uint64> "Max" (mv.c(data_0)) [||]
    static member Max([<ParamArray>]data_0: Tensor<int8>[]) =
        MV() |> fun mv -> execNode<int8> "Max" (mv.c(data_0)) [||]
    static member Max([<ParamArray>]data_0: Tensor<int16>[]) =
        MV() |> fun mv -> execNode<int16> "Max" (mv.c(data_0)) [||]
    static member Max([<ParamArray>]data_0: Tensor<int>[]) =
        MV() |> fun mv -> execNode<int> "Max" (mv.c(data_0)) [||]
    static member Max([<ParamArray>]data_0: Tensor<int64>[]) =
        MV() |> fun mv -> execNode<int64> "Max" (mv.c(data_0)) [||]
    static member Max([<ParamArray>]data_0: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Max" (mv.c(data_0)) [||]
    static member Max([<ParamArray>]data_0: Tensor<double>[]) =
        MV() |> fun mv -> execNode<double> "Max" (mv.c(data_0)) [||]
    static member GroupNormalization(X: Tensor<float32>, scale: Tensor<float32>, bias: Tensor<float32>, num_groups: int64, ?epsilon: float32) =
        MV() |> fun mv -> execNode<float32> "GroupNormalization" [|mv.c(X); mv.c(scale); mv.c(bias)|] ([|Attr.int("num_groups", num_groups); Attr.float("epsilon", epsilon, 9.999999747378752e-06f)|] |> Array.choose id)
    static member GroupNormalization(X: Tensor<double>, scale: Tensor<double>, bias: Tensor<double>, num_groups: int64, ?epsilon: float32) =
        MV() |> fun mv -> execNode<double> "GroupNormalization" [|mv.c(X); mv.c(scale); mv.c(bias)|] ([|Attr.int("num_groups", num_groups); Attr.float("epsilon", epsilon, 9.999999747378752e-06f)|] |> Array.choose id)
    static member Mod(A: Tensor<uint8>, B: Tensor<uint8>, ?fmod: int64) =
        MV() |> fun mv -> execNode<uint8> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<uint16>, B: Tensor<uint16>, ?fmod: int64) =
        MV() |> fun mv -> execNode<uint16> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<uint32>, B: Tensor<uint32>, ?fmod: int64) =
        MV() |> fun mv -> execNode<uint32> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<uint64>, B: Tensor<uint64>, ?fmod: int64) =
        MV() |> fun mv -> execNode<uint64> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<int8>, B: Tensor<int8>, ?fmod: int64) =
        MV() |> fun mv -> execNode<int8> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<int16>, B: Tensor<int16>, ?fmod: int64) =
        MV() |> fun mv -> execNode<int16> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<int>, B: Tensor<int>, ?fmod: int64) =
        MV() |> fun mv -> execNode<int> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<int64>, B: Tensor<int64>, ?fmod: int64) =
        MV() |> fun mv -> execNode<int64> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<float32>, B: Tensor<float32>, ?fmod: int64) =
        MV() |> fun mv -> execNode<float32> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Mod(A: Tensor<double>, B: Tensor<double>, ?fmod: int64) =
        MV() |> fun mv -> execNode<double> "Mod" [|mv.c(A); mv.c(B)|] ([|Attr.int("fmod", fmod, 0L)|] |> Array.choose id)
    static member Log(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Log" [|mv.c(input)|] [||]
    static member Log(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Log" [|mv.c(input)|] [||]
    static member ArgMax(data: Tensor<uint8>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<uint8> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<uint16>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<uint16> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<uint32>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<uint32> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<uint64>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<uint64> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<int8>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<int8> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<int16>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<int16> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<int>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<int> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<int64>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<int64> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<float32>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<float32> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMax(data: Tensor<double>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<double> "ArgMax" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceMax" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceMax" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int> "ReduceMax" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceMax" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceMax" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<double> "ReduceMax" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<uint8>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint8> "ReduceMax" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMax(data: Tensor<int8>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int8> "ReduceMax" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMin(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceMin" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMin(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceMin" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMin(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int> "ReduceMin" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMin(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceMin" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMin(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceMin" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMin(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<double> "ReduceMin" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMin(data: Tensor<uint8>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint8> "ReduceMin" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMin(data: Tensor<int8>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int8> "ReduceMin" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member DeformConv(X: Tensor<float32>, W: Tensor<float32>, offset: Tensor<float32>, ?B: Tensor<float32>, ?mask: Tensor<float32>, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?offset_group: int64, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "DeformConv" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(offset)); mv.c(B); mv.c(mask)|] |> Array.choose id) ([|Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.int("offset_group", offset_group, 1L); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member DeformConv(X: Tensor<double>, W: Tensor<double>, offset: Tensor<double>, ?B: Tensor<double>, ?mask: Tensor<double>, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?offset_group: int64, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<double> "DeformConv" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(offset)); mv.c(B); mv.c(mask)|] |> Array.choose id) ([|Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.int("offset_group", offset_group, 1L); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Sign(input: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Sign" [|mv.c(input)|] [||]
    static member Sign(input: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Sign" [|mv.c(input)|] [||]
    static member Sign(input: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Sign" [|mv.c(input)|] [||]
    static member Sign(input: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Sign" [|mv.c(input)|] [||]
    static member Sign(input: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Sign" [|mv.c(input)|] [||]
    static member Sign(input: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Sign" [|mv.c(input)|] [||]
    static member Sign(input: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Sign" [|mv.c(input)|] [||]
    static member Sign(input: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Sign" [|mv.c(input)|] [||]
    static member Sign(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sign" [|mv.c(input)|] [||]
    static member Sign(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Sign" [|mv.c(input)|] [||]
    static member Min([<ParamArray>]data_0: Tensor<uint8>[]) =
        MV() |> fun mv -> execNode<uint8> "Min" (mv.c(data_0)) [||]
    static member Min([<ParamArray>]data_0: Tensor<uint16>[]) =
        MV() |> fun mv -> execNode<uint16> "Min" (mv.c(data_0)) [||]
    static member Min([<ParamArray>]data_0: Tensor<uint32>[]) =
        MV() |> fun mv -> execNode<uint32> "Min" (mv.c(data_0)) [||]
    static member Min([<ParamArray>]data_0: Tensor<uint64>[]) =
        MV() |> fun mv -> execNode<uint64> "Min" (mv.c(data_0)) [||]
    static member Min([<ParamArray>]data_0: Tensor<int8>[]) =
        MV() |> fun mv -> execNode<int8> "Min" (mv.c(data_0)) [||]
    static member Min([<ParamArray>]data_0: Tensor<int16>[]) =
        MV() |> fun mv -> execNode<int16> "Min" (mv.c(data_0)) [||]
    static member Min([<ParamArray>]data_0: Tensor<int>[]) =
        MV() |> fun mv -> execNode<int> "Min" (mv.c(data_0)) [||]
    static member Min([<ParamArray>]data_0: Tensor<int64>[]) =
        MV() |> fun mv -> execNode<int64> "Min" (mv.c(data_0)) [||]
    static member Min([<ParamArray>]data_0: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Min" (mv.c(data_0)) [||]
    static member Min([<ParamArray>]data_0: Tensor<double>[]) =
        MV() |> fun mv -> execNode<double> "Min" (mv.c(data_0)) [||]
    static member Range(start: Tensor<float32>, limit: Tensor<float32>, delta: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Range" [|mv.c(start); mv.c(limit); mv.c(delta)|] [||]
    static member Range(start: Tensor<double>, limit: Tensor<double>, delta: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Range" [|mv.c(start); mv.c(limit); mv.c(delta)|] [||]
    static member Range(start: Tensor<int16>, limit: Tensor<int16>, delta: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Range" [|mv.c(start); mv.c(limit); mv.c(delta)|] [||]
    static member Range(start: Tensor<int>, limit: Tensor<int>, delta: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Range" [|mv.c(start); mv.c(limit); mv.c(delta)|] [||]
    static member Range(start: Tensor<int64>, limit: Tensor<int64>, delta: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Range" [|mv.c(start); mv.c(limit); mv.c(delta)|] [||]
    static member PRelu(X: Tensor<float32>, slope: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member PRelu(X: Tensor<double>, slope: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member PRelu(X: Tensor<uint32>, slope: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member PRelu(X: Tensor<uint64>, slope: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member PRelu(X: Tensor<int>, slope: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member PRelu(X: Tensor<int64>, slope: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "PRelu" [|mv.c(X); mv.c(slope)|] [||]
    static member NonZero(X: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<string>) =
        MV() |> fun mv -> execNode<string> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "NonZero" [|mv.c(X)|] [||]
    static member NonZero(X: Tensor<Complex>) =
        MV() |> fun mv -> execNode<Complex> "NonZero" [|mv.c(X)|] [||]
    static member Ceil(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Ceil" [|mv.c(X)|] [||]
    static member Ceil(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Ceil" [|mv.c(X)|] [||]
    static member Tan(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Tan" [|mv.c(input)|] [||]
    static member Tan(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Tan" [|mv.c(input)|] [||]
    static member Not(X: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Not" [|mv.c(X)|] [||]
    static member Clip(input: Tensor<uint8>, ?min: Tensor<uint8>, ?max: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Clip" ([|Some(mv.c(input)); mv.c(min); mv.c(max)|] |> Array.choose id) [||]
    static member Clip(input: Tensor<uint16>, ?min: Tensor<uint16>, ?max: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Clip" ([|Some(mv.c(input)); mv.c(min); mv.c(max)|] |> Array.choose id) [||]
    static member Clip(input: Tensor<uint32>, ?min: Tensor<uint32>, ?max: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Clip" ([|Some(mv.c(input)); mv.c(min); mv.c(max)|] |> Array.choose id) [||]
    static member Clip(input: Tensor<uint64>, ?min: Tensor<uint64>, ?max: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Clip" ([|Some(mv.c(input)); mv.c(min); mv.c(max)|] |> Array.choose id) [||]
    static member Clip(input: Tensor<int8>, ?min: Tensor<int8>, ?max: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Clip" ([|Some(mv.c(input)); mv.c(min); mv.c(max)|] |> Array.choose id) [||]
    static member Clip(input: Tensor<int16>, ?min: Tensor<int16>, ?max: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Clip" ([|Some(mv.c(input)); mv.c(min); mv.c(max)|] |> Array.choose id) [||]
    static member Clip(input: Tensor<int>, ?min: Tensor<int>, ?max: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Clip" ([|Some(mv.c(input)); mv.c(min); mv.c(max)|] |> Array.choose id) [||]
    static member Clip(input: Tensor<int64>, ?min: Tensor<int64>, ?max: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Clip" ([|Some(mv.c(input)); mv.c(min); mv.c(max)|] |> Array.choose id) [||]
    static member Clip(input: Tensor<float32>, ?min: Tensor<float32>, ?max: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Clip" ([|Some(mv.c(input)); mv.c(min); mv.c(max)|] |> Array.choose id) [||]
    static member Clip(input: Tensor<double>, ?min: Tensor<double>, ?max: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Clip" ([|Some(mv.c(input)); mv.c(min); mv.c(max)|] |> Array.choose id) [||]
    static member ReduceL2(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceL2" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceL2(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceL2" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceL2(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int> "ReduceL2" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceL2(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceL2" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceL2(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceL2" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceL2(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<double> "ReduceL2" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member Neg(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Neg" [|mv.c(X)|] [||]
    static member Neg(X: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Neg" [|mv.c(X)|] [||]
    static member Neg(X: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Neg" [|mv.c(X)|] [||]
    static member Neg(X: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Neg" [|mv.c(X)|] [||]
    static member Neg(X: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Neg" [|mv.c(X)|] [||]
    static member Neg(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Neg" [|mv.c(X)|] [||]
    static member LinearRegressor(X: Tensor<float32>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        MV() |> fun mv -> execNode<float32> "LinearRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member LinearRegressor(X: Tensor<double>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        MV() |> fun mv -> execNode<double> "LinearRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member LinearRegressor(X: Tensor<int64>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        MV() |> fun mv -> execNode<int64> "LinearRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member LinearRegressor(X: Tensor<int>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        MV() |> fun mv -> execNode<int> "LinearRegressor" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|] |> Array.choose id)
    static member BitwiseXor(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "BitwiseXor" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseXor(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "BitwiseXor" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseXor(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "BitwiseXor" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseXor(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "BitwiseXor" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseXor(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "BitwiseXor" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseXor(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "BitwiseXor" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseXor(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "BitwiseXor" [|mv.c(A); mv.c(B)|] [||]
    static member BitwiseXor(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "BitwiseXor" [|mv.c(A); mv.c(B)|] [||]
    static member Conv(X: Tensor<float32>, W: Tensor<float32>, ?B: Tensor<float32>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "Conv" ([|Some(mv.c(X)); Some(mv.c(W)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Conv(X: Tensor<double>, W: Tensor<double>, ?B: Tensor<double>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<double> "Conv" ([|Some(mv.c(X)); Some(mv.c(W)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Abs(X: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Abs" [|mv.c(X)|] [||]
    static member Abs(X: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Abs" [|mv.c(X)|] [||]
    static member Abs(X: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Abs" [|mv.c(X)|] [||]
    static member Abs(X: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Abs" [|mv.c(X)|] [||]
    static member Abs(X: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Abs" [|mv.c(X)|] [||]
    static member Abs(X: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Abs" [|mv.c(X)|] [||]
    static member Abs(X: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Abs" [|mv.c(X)|] [||]
    static member Abs(X: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Abs" [|mv.c(X)|] [||]
    static member Abs(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Abs" [|mv.c(X)|] [||]
    static member Abs(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Abs" [|mv.c(X)|] [||]
    static member Softplus(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Softplus" [|mv.c(X)|] [||]
    static member Softplus(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Softplus" [|mv.c(X)|] [||]
    static member ConvTranspose(X: Tensor<float32>, W: Tensor<float32>, ?B: Tensor<float32>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?output_padding: int64[], ?output_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "ConvTranspose" ([|Some(mv.c(X)); Some(mv.c(W)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("output_padding", output_padding); Attr.ints("output_shape", output_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member ConvTranspose(X: Tensor<double>, W: Tensor<double>, ?B: Tensor<double>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?output_padding: int64[], ?output_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<double> "ConvTranspose" ([|Some(mv.c(X)); Some(mv.c(W)); mv.c(B)|] |> Array.choose id) ([|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("output_padding", output_padding); Attr.ints("output_shape", output_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Flatten(input: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<uint16>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<uint32>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<uint64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<int16>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member Flatten(input: Tensor<Complex>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "Flatten" [|mv.c(input)|] ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member ReduceLogSum(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceLogSum" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceLogSum(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceLogSum" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceLogSum(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int> "ReduceLogSum" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceLogSum(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceLogSum" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceLogSum(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceLogSum" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceLogSum(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<double> "ReduceLogSum" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member Einsum(equation: string, [<ParamArray>]inputs: Tensor<uint8>[]) =
        MV() |> fun mv -> execNode<uint8> "Einsum" (mv.c(inputs)) ([|Attr.string("equation", equation)|] |> Array.choose id)
    static member Einsum(equation: string, [<ParamArray>]inputs: Tensor<uint16>[]) =
        MV() |> fun mv -> execNode<uint16> "Einsum" (mv.c(inputs)) ([|Attr.string("equation", equation)|] |> Array.choose id)
    static member Einsum(equation: string, [<ParamArray>]inputs: Tensor<uint32>[]) =
        MV() |> fun mv -> execNode<uint32> "Einsum" (mv.c(inputs)) ([|Attr.string("equation", equation)|] |> Array.choose id)
    static member Einsum(equation: string, [<ParamArray>]inputs: Tensor<uint64>[]) =
        MV() |> fun mv -> execNode<uint64> "Einsum" (mv.c(inputs)) ([|Attr.string("equation", equation)|] |> Array.choose id)
    static member Einsum(equation: string, [<ParamArray>]inputs: Tensor<int8>[]) =
        MV() |> fun mv -> execNode<int8> "Einsum" (mv.c(inputs)) ([|Attr.string("equation", equation)|] |> Array.choose id)
    static member Einsum(equation: string, [<ParamArray>]inputs: Tensor<int16>[]) =
        MV() |> fun mv -> execNode<int16> "Einsum" (mv.c(inputs)) ([|Attr.string("equation", equation)|] |> Array.choose id)
    static member Einsum(equation: string, [<ParamArray>]inputs: Tensor<int>[]) =
        MV() |> fun mv -> execNode<int> "Einsum" (mv.c(inputs)) ([|Attr.string("equation", equation)|] |> Array.choose id)
    static member Einsum(equation: string, [<ParamArray>]inputs: Tensor<int64>[]) =
        MV() |> fun mv -> execNode<int64> "Einsum" (mv.c(inputs)) ([|Attr.string("equation", equation)|] |> Array.choose id)
    static member Einsum(equation: string, [<ParamArray>]inputs: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Einsum" (mv.c(inputs)) ([|Attr.string("equation", equation)|] |> Array.choose id)
    static member Einsum(equation: string, [<ParamArray>]inputs: Tensor<double>[]) =
        MV() |> fun mv -> execNode<double> "Einsum" (mv.c(inputs)) ([|Attr.string("equation", equation)|] |> Array.choose id)
    static member ReduceLogSumExp(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceLogSumExp" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceLogSumExp(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceLogSumExp" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceLogSumExp(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int> "ReduceLogSumExp" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceLogSumExp(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceLogSumExp" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceLogSumExp(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceLogSumExp" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceLogSumExp(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<double> "ReduceLogSumExp" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member Sub(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member Sub(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member Sub(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member Sub(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member Sub(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member Sub(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member Sub(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member Sub(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member Sub(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member Sub(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Sub" [|mv.c(A); mv.c(B)|] [||]
    static member Floor(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Floor" [|mv.c(X)|] [||]
    static member Floor(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Floor" [|mv.c(X)|] [||]
    static member MaxRoiPool(X: Tensor<float32>, rois: Tensor<float32>, pooled_shape: int64[], ?spatial_scale: float32) =
        MV() |> fun mv -> execNode<float32> "MaxRoiPool" [|mv.c(X); mv.c(rois)|] ([|Attr.ints("pooled_shape", pooled_shape); Attr.float("spatial_scale", spatial_scale, 1.0f)|] |> Array.choose id)
    static member MaxRoiPool(X: Tensor<double>, rois: Tensor<double>, pooled_shape: int64[], ?spatial_scale: float32) =
        MV() |> fun mv -> execNode<double> "MaxRoiPool" [|mv.c(X); mv.c(rois)|] ([|Attr.ints("pooled_shape", pooled_shape); Attr.float("spatial_scale", spatial_scale, 1.0f)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<uint8>[]) =
        MV() |> fun mv -> execNode<uint8> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<uint16>[]) =
        MV() |> fun mv -> execNode<uint16> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<uint32>[]) =
        MV() |> fun mv -> execNode<uint32> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<uint64>[]) =
        MV() |> fun mv -> execNode<uint64> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<int8>[]) =
        MV() |> fun mv -> execNode<int8> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<int16>[]) =
        MV() |> fun mv -> execNode<int16> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<int>[]) =
        MV() |> fun mv -> execNode<int> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<int64>[]) =
        MV() |> fun mv -> execNode<int64> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<double>[]) =
        MV() |> fun mv -> execNode<double> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<string>[]) =
        MV() |> fun mv -> execNode<string> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<bool>[]) =
        MV() |> fun mv -> execNode<bool> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Concat(axis: int64, [<ParamArray>]inputs: Tensor<Complex>[]) =
        MV() |> fun mv -> execNode<Complex> "Concat" (mv.c(inputs)) ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Sigmoid(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sigmoid" [|mv.c(X)|] [||]
    static member Sigmoid(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Sigmoid" [|mv.c(X)|] [||]
    static member Softmax(input: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Softmax" [|mv.c(input)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member Softmax(input: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Softmax" [|mv.c(input)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member Add(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member Add(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member Add(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member Add(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member Add(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member Add(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member Add(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member Add(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member Add(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member Add(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Add" [|mv.c(A); mv.c(B)|] [||]
    static member InstanceNormalization(input: Tensor<float32>, scale: Tensor<float32>, B: Tensor<float32>, ?epsilon: float32) =
        MV() |> fun mv -> execNode<float32> "InstanceNormalization" [|mv.c(input); mv.c(scale); mv.c(B)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f)|] |> Array.choose id)
    static member InstanceNormalization(input: Tensor<double>, scale: Tensor<double>, B: Tensor<double>, ?epsilon: float32) =
        MV() |> fun mv -> execNode<double> "InstanceNormalization" [|mv.c(input); mv.c(scale); mv.c(B)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f)|] |> Array.choose id)
    static member LpPool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?p: int64, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "LpPool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("p", p, 2L); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member LpPool(X: Tensor<double>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?p: int64, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<double> "LpPool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("p", p, 2L); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member ArgMin(data: Tensor<uint8>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<uint8> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<uint16>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<uint16> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<uint32>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<uint32> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<uint64>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<uint64> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<int8>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<int8> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<int16>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<int16> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<int>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<int> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<int64>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<int64> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<float32>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<float32> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member ArgMin(data: Tensor<double>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        MV() |> fun mv -> execNode<double> "ArgMin" [|mv.c(data)|] ([|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L); Attr.int("select_last_index", select_last_index, 0L)|] |> Array.choose id)
    static member Round(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Round" [|mv.c(X)|] [||]
    static member Round(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Round" [|mv.c(X)|] [||]
    static member BitShift(X: Tensor<uint8>, Y: Tensor<uint8>, direction: string) =
        MV() |> fun mv -> execNode<uint8> "BitShift" [|mv.c(X); mv.c(Y)|] ([|Attr.string("direction", direction)|] |> Array.choose id)
    static member BitShift(X: Tensor<uint16>, Y: Tensor<uint16>, direction: string) =
        MV() |> fun mv -> execNode<uint16> "BitShift" [|mv.c(X); mv.c(Y)|] ([|Attr.string("direction", direction)|] |> Array.choose id)
    static member BitShift(X: Tensor<uint32>, Y: Tensor<uint32>, direction: string) =
        MV() |> fun mv -> execNode<uint32> "BitShift" [|mv.c(X); mv.c(Y)|] ([|Attr.string("direction", direction)|] |> Array.choose id)
    static member BitShift(X: Tensor<uint64>, Y: Tensor<uint64>, direction: string) =
        MV() |> fun mv -> execNode<uint64> "BitShift" [|mv.c(X); mv.c(Y)|] ([|Attr.string("direction", direction)|] |> Array.choose id)
    static member AveragePool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?count_include_pad: int64, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "AveragePool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("count_include_pad", count_include_pad, 0L); Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member AveragePool(X: Tensor<double>, kernel_shape: int64[], ?auto_pad: string, ?count_include_pad: int64, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<double> "AveragePool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("count_include_pad", count_include_pad, 0L); Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Exp(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Exp" [|mv.c(input)|] [||]
    static member Exp(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Exp" [|mv.c(input)|] [||]
    static member ArrayFeatureExtractor(X: Tensor<float32>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member ArrayFeatureExtractor(X: Tensor<double>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member ArrayFeatureExtractor(X: Tensor<int64>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member ArrayFeatureExtractor(X: Tensor<int>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member ArrayFeatureExtractor(X: Tensor<string>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "ArrayFeatureExtractor" [|mv.c(X); mv.c(Y)|] [||]
    static member MatMul(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member MatMul(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member MatMul(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member MatMul(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member MatMul(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member MatMul(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "MatMul" [|mv.c(A); mv.c(B)|] [||]
    static member LeakyRelu(X: Tensor<float32>, ?alpha: float32) =
        MV() |> fun mv -> execNode<float32> "LeakyRelu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 0.009999999776482582f)|] |> Array.choose id)
    static member LeakyRelu(X: Tensor<double>, ?alpha: float32) =
        MV() |> fun mv -> execNode<double> "LeakyRelu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 0.009999999776482582f)|] |> Array.choose id)
    static member ReduceMean(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceMean" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMean(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceMean" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMean(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int> "ReduceMean" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMean(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceMean" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMean(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceMean" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceMean(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<double> "ReduceMean" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<uint8>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<uint8> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<uint16>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<uint16> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<uint32>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<uint32> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<uint64>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<uint64> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<int8>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<int8> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<int16>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<int16> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<int>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<int> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<int64>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<int64> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<float32>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<float32> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<double>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<double> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<string>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<string> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<bool>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<bool> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member ReverseSequence(input: Tensor<Complex>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        MV() |> fun mv -> execNode<Complex> "ReverseSequence" [|mv.c(input); mv.c(sequence_lens)|] ([|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|] |> Array.choose id)
    static member LpNormalization(input: Tensor<float32>, ?axis: int64, ?p: int64) =
        MV() |> fun mv -> execNode<float32> "LpNormalization" [|mv.c(input)|] ([|Attr.int("axis", axis, -1L); Attr.int("p", p, 2L)|] |> Array.choose id)
    static member LpNormalization(input: Tensor<double>, ?axis: int64, ?p: int64) =
        MV() |> fun mv -> execNode<double> "LpNormalization" [|mv.c(input)|] ([|Attr.int("axis", axis, -1L); Attr.int("p", p, 2L)|] |> Array.choose id)
    static member Gemm(A: Tensor<float32>, B: Tensor<float32>, ?C: Tensor<float32>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<float32> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member Gemm(A: Tensor<double>, B: Tensor<double>, ?C: Tensor<double>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<double> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member Gemm(A: Tensor<uint32>, B: Tensor<uint32>, ?C: Tensor<uint32>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<uint32> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member Gemm(A: Tensor<uint64>, B: Tensor<uint64>, ?C: Tensor<uint64>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<uint64> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member Gemm(A: Tensor<int>, B: Tensor<int>, ?C: Tensor<int>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<int> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member Gemm(A: Tensor<int64>, B: Tensor<int64>, ?C: Tensor<int64>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        MV() |> fun mv -> execNode<int64> "Gemm" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(C)|] |> Array.choose id) ([|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|] |> Array.choose id)
    static member GlobalLpPool(X: Tensor<float32>, ?p: int64) =
        MV() |> fun mv -> execNode<float32> "GlobalLpPool" [|mv.c(X)|] ([|Attr.int("p", p, 2L)|] |> Array.choose id)
    static member GlobalLpPool(X: Tensor<double>, ?p: int64) =
        MV() |> fun mv -> execNode<double> "GlobalLpPool" [|mv.c(X)|] ([|Attr.int("p", p, 2L)|] |> Array.choose id)
    static member HardSwish(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "HardSwish" [|mv.c(X)|] [||]
    static member HardSwish(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "HardSwish" [|mv.c(X)|] [||]
    static member Mean([<ParamArray>]data_0: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Mean" (mv.c(data_0)) [||]
    static member Mean([<ParamArray>]data_0: Tensor<double>[]) =
        MV() |> fun mv -> execNode<double> "Mean" (mv.c(data_0)) [||]
    static member Asin(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Asin" [|mv.c(input)|] [||]
    static member Asin(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Asin" [|mv.c(input)|] [||]
    static member OneHotEncoder(X: Tensor<string>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<string> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member OneHotEncoder(X: Tensor<int64>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<int64> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member OneHotEncoder(X: Tensor<int>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<int> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member OneHotEncoder(X: Tensor<float32>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<float32> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member OneHotEncoder(X: Tensor<double>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        MV() |> fun mv -> execNode<double> "OneHotEncoder" [|mv.c(X)|] ([|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<uint8>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<uint8> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<uint16>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<uint16> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<uint32>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<uint32> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<uint64>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<uint64> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<int8>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<int8> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<int16>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<int16> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<int>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<int> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<int64>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<int64> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<float32>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<float32> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<double>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<double> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<string>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<string> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<bool>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<bool> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member DepthToSpace(input: Tensor<Complex>, blocksize: int64, ?mode: string) =
        MV() |> fun mv -> execNode<Complex> "DepthToSpace" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|] |> Array.choose id)
    static member Div(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member Div(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member Div(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member Div(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member Div(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member Div(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member Div(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member Div(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member Div(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member Div(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Div" [|mv.c(A); mv.c(B)|] [||]
    static member Softsign(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Softsign" [|mv.c(input)|] [||]
    static member Softsign(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Softsign" [|mv.c(input)|] [||]
    static member GlobalMaxPool(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "GlobalMaxPool" [|mv.c(X)|] [||]
    static member GlobalMaxPool(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "GlobalMaxPool" [|mv.c(X)|] [||]
    static member Reciprocal(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Reciprocal" [|mv.c(X)|] [||]
    static member Reciprocal(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Reciprocal" [|mv.c(X)|] [||]
    static member MeanVarianceNormalization(X: Tensor<float32>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<float32> "MeanVarianceNormalization" [|mv.c(X)|] ([|Attr.ints("axes", axes, [|0L;2L;3L|])|] |> Array.choose id)
    static member MeanVarianceNormalization(X: Tensor<double>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<double> "MeanVarianceNormalization" [|mv.c(X)|] ([|Attr.ints("axes", axes, [|0L;2L;3L|])|] |> Array.choose id)
    static member ReduceL1(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceL1" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceL1(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceL1" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceL1(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int> "ReduceL1" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceL1(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceL1" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceL1(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceL1" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceL1(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<double> "ReduceL1" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member Relu(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Relu" [|mv.c(X)|] [||]
    static member Relu(X: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Relu" [|mv.c(X)|] [||]
    static member Relu(X: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Relu" [|mv.c(X)|] [||]
    static member Relu(X: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Relu" [|mv.c(X)|] [||]
    static member Relu(X: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Relu" [|mv.c(X)|] [||]
    static member Relu(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Relu" [|mv.c(X)|] [||]
    static member ReduceSum(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceSum" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceSum(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceSum" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceSum(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int> "ReduceSum" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceSum(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceSum" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceSum(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceSum" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceSum(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<double> "ReduceSum" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member Elu(X: Tensor<float32>, ?alpha: float32) =
        MV() |> fun mv -> execNode<float32> "Elu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.0f)|] |> Array.choose id)
    static member Elu(X: Tensor<double>, ?alpha: float32) =
        MV() |> fun mv -> execNode<double> "Elu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.0f)|] |> Array.choose id)
    static member Reshape(data: Tensor<uint8>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Reshape" [|mv.c(data); mv.c(shape)|] ([||] |> Array.choose id)
    static member Reshape(data: Tensor<uint16>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint16> "Reshape" [|mv.c(data); mv.c(shape)|] ([||] |> Array.choose id)
    static member Reshape(data: Tensor<uint32>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint32> "Reshape" [|mv.c(data); mv.c(shape)|] ([||] |> Array.choose id)
    static member Reshape(data: Tensor<uint64>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint64> "Reshape" [|mv.c(data); mv.c(shape)|] ([||] |> Array.choose id)
    static member Reshape(data: Tensor<int8>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Reshape" [|mv.c(data); mv.c(shape)|] ([||] |> Array.choose id)
    static member Reshape(data: Tensor<int16>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int16> "Reshape" [|mv.c(data); mv.c(shape)|] ([||] |> Array.choose id)
    static member Reshape(data: Tensor<int>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Reshape" [|mv.c(data); mv.c(shape)|] ([||] |> Array.choose id)
    static member Reshape(data: Tensor<int64>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Reshape" [|mv.c(data); mv.c(shape)|] ([||] |> Array.choose id)
    static member Reshape(data: Tensor<float32>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Reshape" [|mv.c(data); mv.c(shape)|] ([||] |> Array.choose id)
    static member Reshape(data: Tensor<double>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "Reshape" [|mv.c(data); mv.c(shape)|] ([||] |> Array.choose id)
    static member Reshape(data: Tensor<string>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Reshape" [|mv.c(data); mv.c(shape)|] ([||] |> Array.choose id)
    static member Reshape(data: Tensor<bool>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Reshape" [|mv.c(data); mv.c(shape)|] ([||] |> Array.choose id)
    static member Reshape(data: Tensor<Complex>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<Complex> "Reshape" [|mv.c(data); mv.c(shape)|] ([||] |> Array.choose id)
    static member Selu(X: Tensor<float32>, ?alpha: float32, ?gamma: float32) =
        MV() |> fun mv -> execNode<float32> "Selu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.6732631921768188f); Attr.float("gamma", gamma, 1.0507010221481323f)|] |> Array.choose id)
    static member Selu(X: Tensor<double>, ?alpha: float32, ?gamma: float32) =
        MV() |> fun mv -> execNode<double> "Selu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.6732631921768188f); Attr.float("gamma", gamma, 1.0507010221481323f)|] |> Array.choose id)
    static member GlobalAveragePool(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "GlobalAveragePool" [|mv.c(X)|] [||]
    static member GlobalAveragePool(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "GlobalAveragePool" [|mv.c(X)|] [||]
    static member HardSigmoid(X: Tensor<float32>, ?alpha: float32, ?beta: float32) =
        MV() |> fun mv -> execNode<float32> "HardSigmoid" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 0.20000000298023224f); Attr.float("beta", beta, 0.5f)|] |> Array.choose id)
    static member HardSigmoid(X: Tensor<double>, ?alpha: float32, ?beta: float32) =
        MV() |> fun mv -> execNode<double> "HardSigmoid" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 0.20000000298023224f); Attr.float("beta", beta, 0.5f)|] |> Array.choose id)
    static member LogSoftmax(input: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "LogSoftmax" [|mv.c(input)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member LogSoftmax(input: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "LogSoftmax" [|mv.c(input)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<uint8>, blocksize: int64) =
        MV() |> fun mv -> execNode<uint8> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<uint16>, blocksize: int64) =
        MV() |> fun mv -> execNode<uint16> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<uint32>, blocksize: int64) =
        MV() |> fun mv -> execNode<uint32> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<uint64>, blocksize: int64) =
        MV() |> fun mv -> execNode<uint64> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<int8>, blocksize: int64) =
        MV() |> fun mv -> execNode<int8> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<int16>, blocksize: int64) =
        MV() |> fun mv -> execNode<int16> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<int>, blocksize: int64) =
        MV() |> fun mv -> execNode<int> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<int64>, blocksize: int64) =
        MV() |> fun mv -> execNode<int64> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<float32>, blocksize: int64) =
        MV() |> fun mv -> execNode<float32> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<double>, blocksize: int64) =
        MV() |> fun mv -> execNode<double> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<string>, blocksize: int64) =
        MV() |> fun mv -> execNode<string> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<bool>, blocksize: int64) =
        MV() |> fun mv -> execNode<bool> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member SpaceToDepth(input: Tensor<Complex>, blocksize: int64) =
        MV() |> fun mv -> execNode<Complex> "SpaceToDepth" [|mv.c(input)|] ([|Attr.int("blocksize", blocksize)|] |> Array.choose id)
    static member BitwiseNot(X: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "BitwiseNot" [|mv.c(X)|] [||]
    static member BitwiseNot(X: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "BitwiseNot" [|mv.c(X)|] [||]
    static member BitwiseNot(X: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "BitwiseNot" [|mv.c(X)|] [||]
    static member BitwiseNot(X: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "BitwiseNot" [|mv.c(X)|] [||]
    static member BitwiseNot(X: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "BitwiseNot" [|mv.c(X)|] [||]
    static member BitwiseNot(X: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "BitwiseNot" [|mv.c(X)|] [||]
    static member BitwiseNot(X: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "BitwiseNot" [|mv.c(X)|] [||]
    static member BitwiseNot(X: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "BitwiseNot" [|mv.c(X)|] [||]
    static member ReduceSumSquare(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceSumSquare" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceSumSquare(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceSumSquare" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceSumSquare(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int> "ReduceSumSquare" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceSumSquare(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceSumSquare" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceSumSquare(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceSumSquare" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceSumSquare(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<double> "ReduceSumSquare" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member Sqrt(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sqrt" [|mv.c(X)|] [||]
    static member Sqrt(X: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Sqrt" [|mv.c(X)|] [||]
    static member Col2Im(input: Tensor<uint8>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<uint8> "Col2Im" [|mv.c(input); mv.c(image_shape); mv.c(block_shape)|] ([|Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Col2Im(input: Tensor<uint16>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<uint16> "Col2Im" [|mv.c(input); mv.c(image_shape); mv.c(block_shape)|] ([|Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Col2Im(input: Tensor<uint32>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<uint32> "Col2Im" [|mv.c(input); mv.c(image_shape); mv.c(block_shape)|] ([|Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Col2Im(input: Tensor<uint64>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<uint64> "Col2Im" [|mv.c(input); mv.c(image_shape); mv.c(block_shape)|] ([|Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Col2Im(input: Tensor<int8>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int8> "Col2Im" [|mv.c(input); mv.c(image_shape); mv.c(block_shape)|] ([|Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Col2Im(input: Tensor<int16>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int16> "Col2Im" [|mv.c(input); mv.c(image_shape); mv.c(block_shape)|] ([|Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Col2Im(input: Tensor<int>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int> "Col2Im" [|mv.c(input); mv.c(image_shape); mv.c(block_shape)|] ([|Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Col2Im(input: Tensor<int64>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<int64> "Col2Im" [|mv.c(input); mv.c(image_shape); mv.c(block_shape)|] ([|Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Col2Im(input: Tensor<float32>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "Col2Im" [|mv.c(input); mv.c(image_shape); mv.c(block_shape)|] ([|Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Col2Im(input: Tensor<double>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<double> "Col2Im" [|mv.c(input); mv.c(image_shape); mv.c(block_shape)|] ([|Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Col2Im(input: Tensor<string>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<string> "Col2Im" [|mv.c(input); mv.c(image_shape); mv.c(block_shape)|] ([|Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Col2Im(input: Tensor<bool>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<bool> "Col2Im" [|mv.c(input); mv.c(image_shape); mv.c(block_shape)|] ([|Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Col2Im(input: Tensor<Complex>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<Complex> "Col2Im" [|mv.c(input); mv.c(image_shape); mv.c(block_shape)|] ([|Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Identity(input: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<string>) =
        MV() |> fun mv -> execNode<string> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Identity" [|mv.c(input)|] [||]
    static member Identity(input: Tensor<Complex>) =
        MV() |> fun mv -> execNode<Complex> "Identity" [|mv.c(input)|] [||]
    static member Expand(input: Tensor<uint8>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<uint16>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint16> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<uint32>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint32> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<uint64>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint64> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<int8>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<int16>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int16> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<int>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<int64>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<float32>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<double>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<string>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<bool>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Expand(input: Tensor<Complex>, shape: Tensor<int64>) =
        MV() |> fun mv -> execNode<Complex> "Expand" [|mv.c(input); mv.c(shape)|] [||]
    static member Squeeze(data: Tensor<uint8>, ?axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Squeeze" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) [||]
    static member Squeeze(data: Tensor<uint16>, ?axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint16> "Squeeze" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) [||]
    static member Squeeze(data: Tensor<uint32>, ?axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint32> "Squeeze" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) [||]
    static member Squeeze(data: Tensor<uint64>, ?axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint64> "Squeeze" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) [||]
    static member Squeeze(data: Tensor<int8>, ?axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Squeeze" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) [||]
    static member Squeeze(data: Tensor<int16>, ?axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<int16> "Squeeze" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) [||]
    static member Squeeze(data: Tensor<int>, ?axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Squeeze" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) [||]
    static member Squeeze(data: Tensor<int64>, ?axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Squeeze" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) [||]
    static member Squeeze(data: Tensor<float32>, ?axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Squeeze" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) [||]
    static member Squeeze(data: Tensor<double>, ?axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "Squeeze" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) [||]
    static member Squeeze(data: Tensor<string>, ?axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Squeeze" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) [||]
    static member Squeeze(data: Tensor<bool>, ?axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Squeeze" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) [||]
    static member Squeeze(data: Tensor<Complex>, ?axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<Complex> "Squeeze" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) [||]
    static member Sum([<ParamArray>]data_0: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Sum" (mv.c(data_0)) [||]
    static member Sum([<ParamArray>]data_0: Tensor<double>[]) =
        MV() |> fun mv -> execNode<double> "Sum" (mv.c(data_0)) [||]
    static member Upsample(X: Tensor<uint8>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<uint8> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<uint16>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<uint16> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<uint32>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<uint32> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<uint64>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<uint64> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<int8>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<int8> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<int16>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<int16> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<int>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<int> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<int64>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<int64> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<float32>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<float32> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<double>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<double> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<string>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<string> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<bool>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<bool> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Upsample(X: Tensor<Complex>, scales: Tensor<float32>, ?mode: string) =
        MV() |> fun mv -> execNode<Complex> "Upsample" [|mv.c(X); mv.c(scales)|] ([|Attr.string("mode", mode, "nearest")|] |> Array.choose id)
    static member Tanh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Tanh" [|mv.c(input)|] [||]
    static member Tanh(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Tanh" [|mv.c(input)|] [||]
    static member LRN(X: Tensor<float32>, size: int64, ?alpha: float32, ?beta: float32, ?bias: float32) =
        MV() |> fun mv -> execNode<float32> "LRN" [|mv.c(X)|] ([|Attr.int("size", size); Attr.float("alpha", alpha, 9.999999747378752e-05f); Attr.float("beta", beta, 0.75f); Attr.float("bias", bias, 1.0f)|] |> Array.choose id)
    static member LRN(X: Tensor<double>, size: int64, ?alpha: float32, ?beta: float32, ?bias: float32) =
        MV() |> fun mv -> execNode<double> "LRN" [|mv.c(X)|] ([|Attr.int("size", size); Attr.float("alpha", alpha, 9.999999747378752e-05f); Attr.float("beta", beta, 0.75f); Attr.float("bias", bias, 1.0f)|] |> Array.choose id)
    static member Unsqueeze(data: Tensor<uint8>, axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Unsqueeze" [|mv.c(data); mv.c(axes)|] [||]
    static member Unsqueeze(data: Tensor<uint16>, axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint16> "Unsqueeze" [|mv.c(data); mv.c(axes)|] [||]
    static member Unsqueeze(data: Tensor<uint32>, axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint32> "Unsqueeze" [|mv.c(data); mv.c(axes)|] [||]
    static member Unsqueeze(data: Tensor<uint64>, axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint64> "Unsqueeze" [|mv.c(data); mv.c(axes)|] [||]
    static member Unsqueeze(data: Tensor<int8>, axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Unsqueeze" [|mv.c(data); mv.c(axes)|] [||]
    static member Unsqueeze(data: Tensor<int16>, axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<int16> "Unsqueeze" [|mv.c(data); mv.c(axes)|] [||]
    static member Unsqueeze(data: Tensor<int>, axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Unsqueeze" [|mv.c(data); mv.c(axes)|] [||]
    static member Unsqueeze(data: Tensor<int64>, axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Unsqueeze" [|mv.c(data); mv.c(axes)|] [||]
    static member Unsqueeze(data: Tensor<float32>, axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Unsqueeze" [|mv.c(data); mv.c(axes)|] [||]
    static member Unsqueeze(data: Tensor<double>, axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "Unsqueeze" [|mv.c(data); mv.c(axes)|] [||]
    static member Unsqueeze(data: Tensor<string>, axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Unsqueeze" [|mv.c(data); mv.c(axes)|] [||]
    static member Unsqueeze(data: Tensor<bool>, axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Unsqueeze" [|mv.c(data); mv.c(axes)|] [||]
    static member Unsqueeze(data: Tensor<Complex>, axes: Tensor<int64>) =
        MV() |> fun mv -> execNode<Complex> "Unsqueeze" [|mv.c(data); mv.c(axes)|] [||]
    static member ThresholdedRelu(X: Tensor<float32>, ?alpha: float32) =
        MV() |> fun mv -> execNode<float32> "ThresholdedRelu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.0f)|] |> Array.choose id)
    static member ThresholdedRelu(X: Tensor<double>, ?alpha: float32) =
        MV() |> fun mv -> execNode<double> "ThresholdedRelu" [|mv.c(X)|] ([|Attr.float("alpha", alpha, 1.0f)|] |> Array.choose id)
    static member Acos(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Acos" [|mv.c(input)|] [||]
    static member Acos(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Acos" [|mv.c(input)|] [||]
    static member Atan(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Atan" [|mv.c(input)|] [||]
    static member Atan(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Atan" [|mv.c(input)|] [||]
    static member Cos(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Cos" [|mv.c(input)|] [||]
    static member Cos(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Cos" [|mv.c(input)|] [||]
    static member Sin(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sin" [|mv.c(input)|] [||]
    static member Sin(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Sin" [|mv.c(input)|] [||]
    static member Transpose(data: Tensor<uint8>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<uint8> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<uint16>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<uint16> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<uint32>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<uint32> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<uint64>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<uint64> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<int8>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<int8> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<int16>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<int16> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<int>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<int> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<int64>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<int64> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<float32>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<float32> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<double>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<double> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<string>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<string> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<bool>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<bool> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member Transpose(data: Tensor<Complex>, ?perm: int64[]) =
        MV() |> fun mv -> execNode<Complex> "Transpose" [|mv.c(data)|] ([|Attr.ints("perm", perm)|] |> Array.choose id)
    static member ReduceProd(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint32> "ReduceProd" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceProd(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<uint64> "ReduceProd" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceProd(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int> "ReduceProd" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceProd(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<int64> "ReduceProd" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceProd(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<float32> "ReduceProd" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member ReduceProd(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        MV() |> fun mv -> execNode<double> "ReduceProd" ([|Some(mv.c(data)); mv.c(axes)|] |> Array.choose id) ([|Attr.int("keepdims", keepdims, 1L); Attr.int("noop_with_empty_axes", noop_with_empty_axes, 0L)|] |> Array.choose id)
    static member Sinh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Sinh" [|mv.c(input)|] [||]
    static member Sinh(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Sinh" [|mv.c(input)|] [||]
    static member Asinh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Asinh" [|mv.c(input)|] [||]
    static member Asinh(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Asinh" [|mv.c(input)|] [||]
    static member Binarizer(X: Tensor<float32>, ?threshold: float32) =
        MV() |> fun mv -> execNode<float32> "Binarizer" [|mv.c(X)|] ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member Binarizer(X: Tensor<double>, ?threshold: float32) =
        MV() |> fun mv -> execNode<double> "Binarizer" [|mv.c(X)|] ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member Binarizer(X: Tensor<int64>, ?threshold: float32) =
        MV() |> fun mv -> execNode<int64> "Binarizer" [|mv.c(X)|] ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member Binarizer(X: Tensor<int>, ?threshold: float32) =
        MV() |> fun mv -> execNode<int> "Binarizer" [|mv.c(X)|] ([|Attr.float("threshold", threshold, 0.0f)|] |> Array.choose id)
    static member Trilu(input: Tensor<uint8>, ?k: Tensor<int64>, ?upper: int64) =
        MV() |> fun mv -> execNode<uint8> "Trilu" ([|Some(mv.c(input)); mv.c(k)|] |> Array.choose id) ([|Attr.int("upper", upper, 1L)|] |> Array.choose id)
    static member Trilu(input: Tensor<uint16>, ?k: Tensor<int64>, ?upper: int64) =
        MV() |> fun mv -> execNode<uint16> "Trilu" ([|Some(mv.c(input)); mv.c(k)|] |> Array.choose id) ([|Attr.int("upper", upper, 1L)|] |> Array.choose id)
    static member Trilu(input: Tensor<uint32>, ?k: Tensor<int64>, ?upper: int64) =
        MV() |> fun mv -> execNode<uint32> "Trilu" ([|Some(mv.c(input)); mv.c(k)|] |> Array.choose id) ([|Attr.int("upper", upper, 1L)|] |> Array.choose id)
    static member Trilu(input: Tensor<uint64>, ?k: Tensor<int64>, ?upper: int64) =
        MV() |> fun mv -> execNode<uint64> "Trilu" ([|Some(mv.c(input)); mv.c(k)|] |> Array.choose id) ([|Attr.int("upper", upper, 1L)|] |> Array.choose id)
    static member Trilu(input: Tensor<int8>, ?k: Tensor<int64>, ?upper: int64) =
        MV() |> fun mv -> execNode<int8> "Trilu" ([|Some(mv.c(input)); mv.c(k)|] |> Array.choose id) ([|Attr.int("upper", upper, 1L)|] |> Array.choose id)
    static member Trilu(input: Tensor<int16>, ?k: Tensor<int64>, ?upper: int64) =
        MV() |> fun mv -> execNode<int16> "Trilu" ([|Some(mv.c(input)); mv.c(k)|] |> Array.choose id) ([|Attr.int("upper", upper, 1L)|] |> Array.choose id)
    static member Trilu(input: Tensor<int>, ?k: Tensor<int64>, ?upper: int64) =
        MV() |> fun mv -> execNode<int> "Trilu" ([|Some(mv.c(input)); mv.c(k)|] |> Array.choose id) ([|Attr.int("upper", upper, 1L)|] |> Array.choose id)
    static member Trilu(input: Tensor<int64>, ?k: Tensor<int64>, ?upper: int64) =
        MV() |> fun mv -> execNode<int64> "Trilu" ([|Some(mv.c(input)); mv.c(k)|] |> Array.choose id) ([|Attr.int("upper", upper, 1L)|] |> Array.choose id)
    static member Trilu(input: Tensor<float32>, ?k: Tensor<int64>, ?upper: int64) =
        MV() |> fun mv -> execNode<float32> "Trilu" ([|Some(mv.c(input)); mv.c(k)|] |> Array.choose id) ([|Attr.int("upper", upper, 1L)|] |> Array.choose id)
    static member Trilu(input: Tensor<double>, ?k: Tensor<int64>, ?upper: int64) =
        MV() |> fun mv -> execNode<double> "Trilu" ([|Some(mv.c(input)); mv.c(k)|] |> Array.choose id) ([|Attr.int("upper", upper, 1L)|] |> Array.choose id)
    static member Trilu(input: Tensor<string>, ?k: Tensor<int64>, ?upper: int64) =
        MV() |> fun mv -> execNode<string> "Trilu" ([|Some(mv.c(input)); mv.c(k)|] |> Array.choose id) ([|Attr.int("upper", upper, 1L)|] |> Array.choose id)
    static member Trilu(input: Tensor<bool>, ?k: Tensor<int64>, ?upper: int64) =
        MV() |> fun mv -> execNode<bool> "Trilu" ([|Some(mv.c(input)); mv.c(k)|] |> Array.choose id) ([|Attr.int("upper", upper, 1L)|] |> Array.choose id)
    static member Trilu(input: Tensor<Complex>, ?k: Tensor<int64>, ?upper: int64) =
        MV() |> fun mv -> execNode<Complex> "Trilu" ([|Some(mv.c(input)); mv.c(k)|] |> Array.choose id) ([|Attr.int("upper", upper, 1L)|] |> Array.choose id)
    static member Acosh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Acosh" [|mv.c(input)|] [||]
    static member Acosh(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Acosh" [|mv.c(input)|] [||]
    static member Cosh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Cosh" [|mv.c(input)|] [||]
    static member Cosh(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Cosh" [|mv.c(input)|] [||]
    static member Atanh(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Atanh" [|mv.c(input)|] [||]
    static member Atanh(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Atanh" [|mv.c(input)|] [||]
    static member Shrink(input: Tensor<uint8>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<uint8> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<uint16>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<uint16> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<uint32>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<uint32> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<uint64>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<uint64> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<int8>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<int8> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<int16>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<int16> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<int>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<int> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<int64>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<int64> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<float32>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<float32> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Shrink(input: Tensor<double>, ?bias: float32, ?lambd: float32) =
        MV() |> fun mv -> execNode<double> "Shrink" [|mv.c(input)|] ([|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|] |> Array.choose id)
    static member Hardmax(input: Tensor<float32>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Hardmax" [|mv.c(input)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member Hardmax(input: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Hardmax" [|mv.c(input)|] ([|Attr.int("axis", axis, -1L)|] |> Array.choose id)
    static member Erf(input: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Erf" [|mv.c(input)|] [||]
    static member Erf(input: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Erf" [|mv.c(input)|] [||]
    static member Erf(input: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Erf" [|mv.c(input)|] [||]
    static member Erf(input: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Erf" [|mv.c(input)|] [||]
    static member Erf(input: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Erf" [|mv.c(input)|] [||]
    static member Erf(input: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Erf" [|mv.c(input)|] [||]
    static member Erf(input: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Erf" [|mv.c(input)|] [||]
    static member Erf(input: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Erf" [|mv.c(input)|] [||]
    static member Erf(input: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Erf" [|mv.c(input)|] [||]
    static member Erf(input: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Erf" [|mv.c(input)|] [||]
    static member OptionalHasElement(?input: Tensor<uint8>) =
        MV() |> fun mv -> execNode<bool> "OptionalHasElement" ([|mv.c(input)|] |> Array.choose id) [||]
    static member OptionalHasElement(?input: Tensor<uint16>) =
        MV() |> fun mv -> execNode<bool> "OptionalHasElement" ([|mv.c(input)|] |> Array.choose id) [||]
    static member OptionalHasElement(?input: Tensor<uint32>) =
        MV() |> fun mv -> execNode<bool> "OptionalHasElement" ([|mv.c(input)|] |> Array.choose id) [||]
    static member OptionalHasElement(?input: Tensor<uint64>) =
        MV() |> fun mv -> execNode<bool> "OptionalHasElement" ([|mv.c(input)|] |> Array.choose id) [||]
    static member OptionalHasElement(?input: Tensor<int8>) =
        MV() |> fun mv -> execNode<bool> "OptionalHasElement" ([|mv.c(input)|] |> Array.choose id) [||]
    static member OptionalHasElement(?input: Tensor<int16>) =
        MV() |> fun mv -> execNode<bool> "OptionalHasElement" ([|mv.c(input)|] |> Array.choose id) [||]
    static member OptionalHasElement(?input: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "OptionalHasElement" ([|mv.c(input)|] |> Array.choose id) [||]
    static member OptionalHasElement(?input: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "OptionalHasElement" ([|mv.c(input)|] |> Array.choose id) [||]
    static member OptionalHasElement(?input: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "OptionalHasElement" ([|mv.c(input)|] |> Array.choose id) [||]
    static member OptionalHasElement(?input: Tensor<double>) =
        MV() |> fun mv -> execNode<bool> "OptionalHasElement" ([|mv.c(input)|] |> Array.choose id) [||]
    static member OptionalHasElement(?input: Tensor<string>) =
        MV() |> fun mv -> execNode<bool> "OptionalHasElement" ([|mv.c(input)|] |> Array.choose id) [||]
    static member OptionalHasElement(?input: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "OptionalHasElement" ([|mv.c(input)|] |> Array.choose id) [||]
    static member OptionalHasElement(?input: Tensor<Complex>) =
        MV() |> fun mv -> execNode<bool> "OptionalHasElement" ([|mv.c(input)|] |> Array.choose id) [||]
    static member GreaterOrEqual(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<bool> "GreaterOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member GreaterOrEqual(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<bool> "GreaterOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member GreaterOrEqual(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<bool> "GreaterOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member GreaterOrEqual(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<bool> "GreaterOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member GreaterOrEqual(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<bool> "GreaterOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member GreaterOrEqual(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<bool> "GreaterOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member GreaterOrEqual(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "GreaterOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member GreaterOrEqual(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "GreaterOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member GreaterOrEqual(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "GreaterOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member GreaterOrEqual(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<bool> "GreaterOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member LessOrEqual(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<bool> "LessOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member LessOrEqual(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<bool> "LessOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member LessOrEqual(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<bool> "LessOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member LessOrEqual(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<bool> "LessOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member LessOrEqual(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<bool> "LessOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member LessOrEqual(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<bool> "LessOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member LessOrEqual(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "LessOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member LessOrEqual(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "LessOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member LessOrEqual(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "LessOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member LessOrEqual(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<bool> "LessOrEqual" [|mv.c(A); mv.c(B)|] [||]
    static member IsInf(X: Tensor<float32>, ?detect_negative: int64, ?detect_positive: int64) =
        MV() |> fun mv -> execNode<bool> "IsInf" [|mv.c(X)|] ([|Attr.int("detect_negative", detect_negative, 1L); Attr.int("detect_positive", detect_positive, 1L)|] |> Array.choose id)
    static member IsInf(X: Tensor<double>, ?detect_negative: int64, ?detect_positive: int64) =
        MV() |> fun mv -> execNode<bool> "IsInf" [|mv.c(X)|] ([|Attr.int("detect_negative", detect_negative, 1L); Attr.int("detect_positive", detect_positive, 1L)|] |> Array.choose id)
    static member Or(A: Tensor<bool>, B: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Or" [|mv.c(A); mv.c(B)|] [||]
    static member TfIdfVectorizer(X: Tensor<string>, max_gram_length: int64, max_skip_count: int64, min_gram_length: int64, mode: string, ngram_counts: int64[], ngram_indexes: int64[], ?pool_int64s: int64[], ?pool_strings: string[], ?weights: float32[]) =
        MV() |> fun mv -> execNode<float32> "TfIdfVectorizer" [|mv.c(X)|] ([|Attr.int("max_gram_length", max_gram_length); Attr.int("max_skip_count", max_skip_count); Attr.int("min_gram_length", min_gram_length); Attr.string("mode", mode); Attr.ints("ngram_counts", ngram_counts); Attr.ints("ngram_indexes", ngram_indexes); Attr.ints("pool_int64s", pool_int64s); Attr.strings("pool_strings", pool_strings); Attr.floats("weights", weights)|] |> Array.choose id)
    static member TfIdfVectorizer(X: Tensor<int>, max_gram_length: int64, max_skip_count: int64, min_gram_length: int64, mode: string, ngram_counts: int64[], ngram_indexes: int64[], ?pool_int64s: int64[], ?pool_strings: string[], ?weights: float32[]) =
        MV() |> fun mv -> execNode<float32> "TfIdfVectorizer" [|mv.c(X)|] ([|Attr.int("max_gram_length", max_gram_length); Attr.int("max_skip_count", max_skip_count); Attr.int("min_gram_length", min_gram_length); Attr.string("mode", mode); Attr.ints("ngram_counts", ngram_counts); Attr.ints("ngram_indexes", ngram_indexes); Attr.ints("pool_int64s", pool_int64s); Attr.strings("pool_strings", pool_strings); Attr.floats("weights", weights)|] |> Array.choose id)
    static member TfIdfVectorizer(X: Tensor<int64>, max_gram_length: int64, max_skip_count: int64, min_gram_length: int64, mode: string, ngram_counts: int64[], ngram_indexes: int64[], ?pool_int64s: int64[], ?pool_strings: string[], ?weights: float32[]) =
        MV() |> fun mv -> execNode<float32> "TfIdfVectorizer" [|mv.c(X)|] ([|Attr.int("max_gram_length", max_gram_length); Attr.int("max_skip_count", max_skip_count); Attr.int("min_gram_length", min_gram_length); Attr.string("mode", mode); Attr.ints("ngram_counts", ngram_counts); Attr.ints("ngram_indexes", ngram_indexes); Attr.ints("pool_int64s", pool_int64s); Attr.strings("pool_strings", pool_strings); Attr.floats("weights", weights)|] |> Array.choose id)
    static member And(A: Tensor<bool>, B: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "And" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Less(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<bool> "Less" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<bool>, B: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Equal(A: Tensor<string>, B: Tensor<string>) =
        MV() |> fun mv -> execNode<bool> "Equal" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<uint8>, B: Tensor<uint8>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<uint16>, B: Tensor<uint16>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<uint32>, B: Tensor<uint32>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<uint64>, B: Tensor<uint64>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<int8>, B: Tensor<int8>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<int16>, B: Tensor<int16>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<int>, B: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<int64>, B: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<float32>, B: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member Greater(A: Tensor<double>, B: Tensor<double>) =
        MV() |> fun mv -> execNode<bool> "Greater" [|mv.c(A); mv.c(B)|] [||]
    static member IsNaN(X: Tensor<float32>) =
        MV() |> fun mv -> execNode<bool> "IsNaN" [|mv.c(X)|] [||]
    static member IsNaN(X: Tensor<double>) =
        MV() |> fun mv -> execNode<bool> "IsNaN" [|mv.c(X)|] [||]
    static member Shape(data: Tensor<uint8>, ?end_: int64, ?start: int64) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] ([|Attr.int("end_", end_); Attr.int("start", start, 0L)|] |> Array.choose id)
    static member Shape(data: Tensor<uint16>, ?end_: int64, ?start: int64) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] ([|Attr.int("end_", end_); Attr.int("start", start, 0L)|] |> Array.choose id)
    static member Shape(data: Tensor<uint32>, ?end_: int64, ?start: int64) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] ([|Attr.int("end_", end_); Attr.int("start", start, 0L)|] |> Array.choose id)
    static member Shape(data: Tensor<uint64>, ?end_: int64, ?start: int64) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] ([|Attr.int("end_", end_); Attr.int("start", start, 0L)|] |> Array.choose id)
    static member Shape(data: Tensor<int8>, ?end_: int64, ?start: int64) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] ([|Attr.int("end_", end_); Attr.int("start", start, 0L)|] |> Array.choose id)
    static member Shape(data: Tensor<int16>, ?end_: int64, ?start: int64) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] ([|Attr.int("end_", end_); Attr.int("start", start, 0L)|] |> Array.choose id)
    static member Shape(data: Tensor<int>, ?end_: int64, ?start: int64) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] ([|Attr.int("end_", end_); Attr.int("start", start, 0L)|] |> Array.choose id)
    static member Shape(data: Tensor<int64>, ?end_: int64, ?start: int64) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] ([|Attr.int("end_", end_); Attr.int("start", start, 0L)|] |> Array.choose id)
    static member Shape(data: Tensor<float32>, ?end_: int64, ?start: int64) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] ([|Attr.int("end_", end_); Attr.int("start", start, 0L)|] |> Array.choose id)
    static member Shape(data: Tensor<double>, ?end_: int64, ?start: int64) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] ([|Attr.int("end_", end_); Attr.int("start", start, 0L)|] |> Array.choose id)
    static member Shape(data: Tensor<string>, ?end_: int64, ?start: int64) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] ([|Attr.int("end_", end_); Attr.int("start", start, 0L)|] |> Array.choose id)
    static member Shape(data: Tensor<bool>, ?end_: int64, ?start: int64) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] ([|Attr.int("end_", end_); Attr.int("start", start, 0L)|] |> Array.choose id)
    static member Shape(data: Tensor<Complex>, ?end_: int64, ?start: int64) =
        MV() |> fun mv -> execNode<int64> "Shape" [|mv.c(data)|] ([|Attr.int("end_", end_); Attr.int("start", start, 0L)|] |> Array.choose id)
    static member Size(data: Tensor<uint8>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<uint16>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<uint32>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<uint64>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<int8>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<int16>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<int>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<float32>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<double>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<string>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<bool>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Size(data: Tensor<Complex>) =
        MV() |> fun mv -> execNode<int64> "Size" [|mv.c(data)|] [||]
    static member Xor(A: Tensor<bool>, B: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Xor" [|mv.c(A); mv.c(B)|] [||]
    static member Adam(R: Tensor<float32>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<float32>[], ?alpha: float32, ?beta: float32, ?epsilon: float32, ?norm_coefficient: float32, ?norm_coefficient_post: float32) =
        MV() |> fun mv -> execNode<float32> "Adam" ([|yield mv.c(R); yield mv.c(T); yield! mv.c(inputs)|]) ([|Attr.float("alpha", alpha, 0.8999999761581421f); Attr.float("beta", beta, 0.9990000128746033f); Attr.float("epsilon", epsilon, 9.999999974752427e-07f); Attr.float("norm_coefficient", norm_coefficient, 0.0f); Attr.float("norm_coefficient_post", norm_coefficient_post, 0.0f)|] |> Array.choose id)
    static member Adam(R: Tensor<float32>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<double>[], ?alpha: float32, ?beta: float32, ?epsilon: float32, ?norm_coefficient: float32, ?norm_coefficient_post: float32) =
        MV() |> fun mv -> execNode<double> "Adam" ([|yield mv.c(R); yield mv.c(T); yield! mv.c(inputs)|]) ([|Attr.float("alpha", alpha, 0.8999999761581421f); Attr.float("beta", beta, 0.9990000128746033f); Attr.float("epsilon", epsilon, 9.999999974752427e-07f); Attr.float("norm_coefficient", norm_coefficient, 0.0f); Attr.float("norm_coefficient_post", norm_coefficient_post, 0.0f)|] |> Array.choose id)
    static member Adam(R: Tensor<double>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<float32>[], ?alpha: float32, ?beta: float32, ?epsilon: float32, ?norm_coefficient: float32, ?norm_coefficient_post: float32) =
        MV() |> fun mv -> execNode<float32> "Adam" ([|yield mv.c(R); yield mv.c(T); yield! mv.c(inputs)|]) ([|Attr.float("alpha", alpha, 0.8999999761581421f); Attr.float("beta", beta, 0.9990000128746033f); Attr.float("epsilon", epsilon, 9.999999974752427e-07f); Attr.float("norm_coefficient", norm_coefficient, 0.0f); Attr.float("norm_coefficient_post", norm_coefficient_post, 0.0f)|] |> Array.choose id)
    static member Adam(R: Tensor<double>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<double>[], ?alpha: float32, ?beta: float32, ?epsilon: float32, ?norm_coefficient: float32, ?norm_coefficient_post: float32) =
        MV() |> fun mv -> execNode<double> "Adam" ([|yield mv.c(R); yield mv.c(T); yield! mv.c(inputs)|]) ([|Attr.float("alpha", alpha, 0.8999999761581421f); Attr.float("beta", beta, 0.9990000128746033f); Attr.float("epsilon", epsilon, 9.999999974752427e-07f); Attr.float("norm_coefficient", norm_coefficient, 0.0f); Attr.float("norm_coefficient_post", norm_coefficient_post, 0.0f)|] |> Array.choose id)
    static member Adagrad(R: Tensor<float32>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<float32>[], ?decay_factor: float32, ?epsilon: float32, ?norm_coefficient: float32) =
        MV() |> fun mv -> execNode<float32> "Adagrad" ([|yield mv.c(R); yield mv.c(T); yield! mv.c(inputs)|]) ([|Attr.float("decay_factor", decay_factor, 0.0f); Attr.float("epsilon", epsilon, 9.999999974752427e-07f); Attr.float("norm_coefficient", norm_coefficient, 0.0f)|] |> Array.choose id)
    static member Adagrad(R: Tensor<float32>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<double>[], ?decay_factor: float32, ?epsilon: float32, ?norm_coefficient: float32) =
        MV() |> fun mv -> execNode<double> "Adagrad" ([|yield mv.c(R); yield mv.c(T); yield! mv.c(inputs)|]) ([|Attr.float("decay_factor", decay_factor, 0.0f); Attr.float("epsilon", epsilon, 9.999999974752427e-07f); Attr.float("norm_coefficient", norm_coefficient, 0.0f)|] |> Array.choose id)
    static member Adagrad(R: Tensor<double>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<float32>[], ?decay_factor: float32, ?epsilon: float32, ?norm_coefficient: float32) =
        MV() |> fun mv -> execNode<float32> "Adagrad" ([|yield mv.c(R); yield mv.c(T); yield! mv.c(inputs)|]) ([|Attr.float("decay_factor", decay_factor, 0.0f); Attr.float("epsilon", epsilon, 9.999999974752427e-07f); Attr.float("norm_coefficient", norm_coefficient, 0.0f)|] |> Array.choose id)
    static member Adagrad(R: Tensor<double>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<double>[], ?decay_factor: float32, ?epsilon: float32, ?norm_coefficient: float32) =
        MV() |> fun mv -> execNode<double> "Adagrad" ([|yield mv.c(R); yield mv.c(T); yield! mv.c(inputs)|]) ([|Attr.float("decay_factor", decay_factor, 0.0f); Attr.float("epsilon", epsilon, 9.999999974752427e-07f); Attr.float("norm_coefficient", norm_coefficient, 0.0f)|] |> Array.choose id)
    static member Momentum(R: Tensor<float32>, T: Tensor<int64>, alpha: float32, beta: float32, mode: string, norm_coefficient: float32, [<ParamArray>]inputs: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Momentum" ([|yield mv.c(R); yield mv.c(T); yield! mv.c(inputs)|]) ([|Attr.float("alpha", alpha); Attr.float("beta", beta); Attr.string("mode", mode); Attr.float("norm_coefficient", norm_coefficient)|] |> Array.choose id)
    static member Momentum(R: Tensor<float32>, T: Tensor<int64>, alpha: float32, beta: float32, mode: string, norm_coefficient: float32, [<ParamArray>]inputs: Tensor<double>[]) =
        MV() |> fun mv -> execNode<double> "Momentum" ([|yield mv.c(R); yield mv.c(T); yield! mv.c(inputs)|]) ([|Attr.float("alpha", alpha); Attr.float("beta", beta); Attr.string("mode", mode); Attr.float("norm_coefficient", norm_coefficient)|] |> Array.choose id)
    static member Momentum(R: Tensor<double>, T: Tensor<int64>, alpha: float32, beta: float32, mode: string, norm_coefficient: float32, [<ParamArray>]inputs: Tensor<float32>[]) =
        MV() |> fun mv -> execNode<float32> "Momentum" ([|yield mv.c(R); yield mv.c(T); yield! mv.c(inputs)|]) ([|Attr.float("alpha", alpha); Attr.float("beta", beta); Attr.string("mode", mode); Attr.float("norm_coefficient", norm_coefficient)|] |> Array.choose id)
    static member Momentum(R: Tensor<double>, T: Tensor<int64>, alpha: float32, beta: float32, mode: string, norm_coefficient: float32, [<ParamArray>]inputs: Tensor<double>[]) =
        MV() |> fun mv -> execNode<double> "Momentum" ([|yield mv.c(R); yield mv.c(T); yield! mv.c(inputs)|]) ([|Attr.float("alpha", alpha); Attr.float("beta", beta); Attr.string("mode", mode); Attr.float("norm_coefficient", norm_coefficient)|] |> Array.choose id)
    static member GridSample(X: Tensor<uint8>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<uint8> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<uint8>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<uint8> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<uint16>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<uint16> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<uint16>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<uint16> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<uint32>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<uint32> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<uint32>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<uint32> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<uint64>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<uint64> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<uint64>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<uint64> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<int8>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<int8> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<int8>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<int8> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<int16>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<int16> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<int16>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<int16> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<int>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<int> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<int>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<int> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<int64>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<int64> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<int64>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<int64> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<float32>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<float32> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<float32>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<float32> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<double>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<double> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<double>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<double> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<string>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<string> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<string>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<string> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<bool>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<bool> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<bool>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<bool> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<Complex>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<Complex> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member GridSample(X: Tensor<Complex>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        MV() |> fun mv -> execNode<Complex> "GridSample" [|mv.c(X); mv.c(grid)|] ([|Attr.int("align_corners", align_corners, 0L); Attr.string("mode", mode, "bilinear"); Attr.string("padding_mode", padding_mode, "zeros")|] |> Array.choose id)
    static member CastLike(input: Tensor<float32>, target_type: Tensor<float32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<float32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<float32>, target_type: Tensor<double>, ?saturate: int64) =
        MV() |> fun mv -> execNode<double> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<float32>, target_type: Tensor<int8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<float32>, target_type: Tensor<int16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<float32>, target_type: Tensor<int>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<float32>, target_type: Tensor<int64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<float32>, target_type: Tensor<uint8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<float32>, target_type: Tensor<uint16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<float32>, target_type: Tensor<uint32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<float32>, target_type: Tensor<uint64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<float32>, target_type: Tensor<bool>, ?saturate: int64) =
        MV() |> fun mv -> execNode<bool> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<float32>, target_type: Tensor<string>, ?saturate: int64) =
        MV() |> fun mv -> execNode<string> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<double>, target_type: Tensor<float32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<float32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<double>, target_type: Tensor<double>, ?saturate: int64) =
        MV() |> fun mv -> execNode<double> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<double>, target_type: Tensor<int8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<double>, target_type: Tensor<int16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<double>, target_type: Tensor<int>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<double>, target_type: Tensor<int64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<double>, target_type: Tensor<uint8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<double>, target_type: Tensor<uint16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<double>, target_type: Tensor<uint32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<double>, target_type: Tensor<uint64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<double>, target_type: Tensor<bool>, ?saturate: int64) =
        MV() |> fun mv -> execNode<bool> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<double>, target_type: Tensor<string>, ?saturate: int64) =
        MV() |> fun mv -> execNode<string> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int8>, target_type: Tensor<float32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<float32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int8>, target_type: Tensor<double>, ?saturate: int64) =
        MV() |> fun mv -> execNode<double> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int8>, target_type: Tensor<int8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int8>, target_type: Tensor<int16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int8>, target_type: Tensor<int>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int8>, target_type: Tensor<int64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int8>, target_type: Tensor<uint8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int8>, target_type: Tensor<uint16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int8>, target_type: Tensor<uint32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int8>, target_type: Tensor<uint64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int8>, target_type: Tensor<bool>, ?saturate: int64) =
        MV() |> fun mv -> execNode<bool> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int8>, target_type: Tensor<string>, ?saturate: int64) =
        MV() |> fun mv -> execNode<string> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int16>, target_type: Tensor<float32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<float32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int16>, target_type: Tensor<double>, ?saturate: int64) =
        MV() |> fun mv -> execNode<double> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int16>, target_type: Tensor<int8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int16>, target_type: Tensor<int16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int16>, target_type: Tensor<int>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int16>, target_type: Tensor<int64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int16>, target_type: Tensor<uint8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int16>, target_type: Tensor<uint16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int16>, target_type: Tensor<uint32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int16>, target_type: Tensor<uint64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int16>, target_type: Tensor<bool>, ?saturate: int64) =
        MV() |> fun mv -> execNode<bool> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int16>, target_type: Tensor<string>, ?saturate: int64) =
        MV() |> fun mv -> execNode<string> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int>, target_type: Tensor<float32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<float32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int>, target_type: Tensor<double>, ?saturate: int64) =
        MV() |> fun mv -> execNode<double> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int>, target_type: Tensor<int8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int>, target_type: Tensor<int16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int>, target_type: Tensor<int>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int>, target_type: Tensor<int64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int>, target_type: Tensor<uint8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int>, target_type: Tensor<uint16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int>, target_type: Tensor<uint32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int>, target_type: Tensor<uint64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int>, target_type: Tensor<bool>, ?saturate: int64) =
        MV() |> fun mv -> execNode<bool> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int>, target_type: Tensor<string>, ?saturate: int64) =
        MV() |> fun mv -> execNode<string> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int64>, target_type: Tensor<float32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<float32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int64>, target_type: Tensor<double>, ?saturate: int64) =
        MV() |> fun mv -> execNode<double> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int64>, target_type: Tensor<int8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int64>, target_type: Tensor<int16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int64>, target_type: Tensor<int>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int64>, target_type: Tensor<int64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int64>, target_type: Tensor<uint8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int64>, target_type: Tensor<uint16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int64>, target_type: Tensor<uint32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int64>, target_type: Tensor<uint64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int64>, target_type: Tensor<bool>, ?saturate: int64) =
        MV() |> fun mv -> execNode<bool> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<int64>, target_type: Tensor<string>, ?saturate: int64) =
        MV() |> fun mv -> execNode<string> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint8>, target_type: Tensor<float32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<float32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint8>, target_type: Tensor<double>, ?saturate: int64) =
        MV() |> fun mv -> execNode<double> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint8>, target_type: Tensor<int8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint8>, target_type: Tensor<int16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint8>, target_type: Tensor<int>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint8>, target_type: Tensor<int64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint8>, target_type: Tensor<uint8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint8>, target_type: Tensor<uint16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint8>, target_type: Tensor<uint32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint8>, target_type: Tensor<uint64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint8>, target_type: Tensor<bool>, ?saturate: int64) =
        MV() |> fun mv -> execNode<bool> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint8>, target_type: Tensor<string>, ?saturate: int64) =
        MV() |> fun mv -> execNode<string> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint16>, target_type: Tensor<float32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<float32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint16>, target_type: Tensor<double>, ?saturate: int64) =
        MV() |> fun mv -> execNode<double> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint16>, target_type: Tensor<int8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint16>, target_type: Tensor<int16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint16>, target_type: Tensor<int>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint16>, target_type: Tensor<int64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint16>, target_type: Tensor<uint8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint16>, target_type: Tensor<uint16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint16>, target_type: Tensor<uint32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint16>, target_type: Tensor<uint64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint16>, target_type: Tensor<bool>, ?saturate: int64) =
        MV() |> fun mv -> execNode<bool> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint16>, target_type: Tensor<string>, ?saturate: int64) =
        MV() |> fun mv -> execNode<string> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint32>, target_type: Tensor<float32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<float32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint32>, target_type: Tensor<double>, ?saturate: int64) =
        MV() |> fun mv -> execNode<double> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint32>, target_type: Tensor<int8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint32>, target_type: Tensor<int16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint32>, target_type: Tensor<int>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint32>, target_type: Tensor<int64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint32>, target_type: Tensor<uint8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint32>, target_type: Tensor<uint16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint32>, target_type: Tensor<uint32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint32>, target_type: Tensor<uint64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint32>, target_type: Tensor<bool>, ?saturate: int64) =
        MV() |> fun mv -> execNode<bool> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint32>, target_type: Tensor<string>, ?saturate: int64) =
        MV() |> fun mv -> execNode<string> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint64>, target_type: Tensor<float32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<float32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint64>, target_type: Tensor<double>, ?saturate: int64) =
        MV() |> fun mv -> execNode<double> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint64>, target_type: Tensor<int8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint64>, target_type: Tensor<int16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint64>, target_type: Tensor<int>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint64>, target_type: Tensor<int64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint64>, target_type: Tensor<uint8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint64>, target_type: Tensor<uint16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint64>, target_type: Tensor<uint32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint64>, target_type: Tensor<uint64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint64>, target_type: Tensor<bool>, ?saturate: int64) =
        MV() |> fun mv -> execNode<bool> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<uint64>, target_type: Tensor<string>, ?saturate: int64) =
        MV() |> fun mv -> execNode<string> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<bool>, target_type: Tensor<float32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<float32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<bool>, target_type: Tensor<double>, ?saturate: int64) =
        MV() |> fun mv -> execNode<double> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<bool>, target_type: Tensor<int8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<bool>, target_type: Tensor<int16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<bool>, target_type: Tensor<int>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<bool>, target_type: Tensor<int64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<bool>, target_type: Tensor<uint8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<bool>, target_type: Tensor<uint16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<bool>, target_type: Tensor<uint32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<bool>, target_type: Tensor<uint64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<bool>, target_type: Tensor<bool>, ?saturate: int64) =
        MV() |> fun mv -> execNode<bool> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<bool>, target_type: Tensor<string>, ?saturate: int64) =
        MV() |> fun mv -> execNode<string> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<string>, target_type: Tensor<float32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<float32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<string>, target_type: Tensor<double>, ?saturate: int64) =
        MV() |> fun mv -> execNode<double> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<string>, target_type: Tensor<int8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<string>, target_type: Tensor<int16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<string>, target_type: Tensor<int>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<string>, target_type: Tensor<int64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<int64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<string>, target_type: Tensor<uint8>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<string>, target_type: Tensor<uint16>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint16> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<string>, target_type: Tensor<uint32>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint32> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<string>, target_type: Tensor<uint64>, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint64> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<string>, target_type: Tensor<bool>, ?saturate: int64) =
        MV() |> fun mv -> execNode<bool> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member CastLike(input: Tensor<string>, target_type: Tensor<string>, ?saturate: int64) =
        MV() |> fun mv -> execNode<string> "CastLike" [|mv.c(input); mv.c(target_type)|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member SoftmaxCrossEntropyLoss(scores: Tensor<float32>, labels: Tensor<int>, ?weights: Tensor<float32>, ?ignore_index: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<float32> "SoftmaxCrossEntropyLoss" ([|Some(mv.c(scores)); Some(mv.c(labels)); mv.c(weights)|] |> Array.choose id) ([|Attr.int("ignore_index", ignore_index); Attr.string("reduction", reduction, "mean")|] |> Array.choose id)
    static member SoftmaxCrossEntropyLoss(scores: Tensor<float32>, labels: Tensor<int64>, ?weights: Tensor<float32>, ?ignore_index: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<float32> "SoftmaxCrossEntropyLoss" ([|Some(mv.c(scores)); Some(mv.c(labels)); mv.c(weights)|] |> Array.choose id) ([|Attr.int("ignore_index", ignore_index); Attr.string("reduction", reduction, "mean")|] |> Array.choose id)
    static member SoftmaxCrossEntropyLoss(scores: Tensor<double>, labels: Tensor<int>, ?weights: Tensor<double>, ?ignore_index: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<double> "SoftmaxCrossEntropyLoss" ([|Some(mv.c(scores)); Some(mv.c(labels)); mv.c(weights)|] |> Array.choose id) ([|Attr.int("ignore_index", ignore_index); Attr.string("reduction", reduction, "mean")|] |> Array.choose id)
    static member SoftmaxCrossEntropyLoss(scores: Tensor<double>, labels: Tensor<int64>, ?weights: Tensor<double>, ?ignore_index: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<double> "SoftmaxCrossEntropyLoss" ([|Some(mv.c(scores)); Some(mv.c(labels)); mv.c(weights)|] |> Array.choose id) ([|Attr.int("ignore_index", ignore_index); Attr.string("reduction", reduction, "mean")|] |> Array.choose id)
    static member GatherElements(data: Tensor<uint8>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<uint8>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<uint16>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<uint16>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<uint32>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<uint32>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<uint64>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<uint64>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<int8>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<int8>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<int16>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<int16>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
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
    static member GatherElements(data: Tensor<double>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<double>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<string>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<string>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<bool>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<bool>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<Complex>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member GatherElements(data: Tensor<Complex>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "GatherElements" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member CumSum(x: Tensor<uint32>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<uint32> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member CumSum(x: Tensor<uint32>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<uint32> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member CumSum(x: Tensor<uint64>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<uint64> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member CumSum(x: Tensor<uint64>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<uint64> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
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
    static member CumSum(x: Tensor<double>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<double> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member CumSum(x: Tensor<double>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        MV() |> fun mv -> execNode<double> "CumSum" [|mv.c(x); mv.c(axis)|] ([|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|] |> Array.choose id)
    static member RoiAlign(X: Tensor<float32>, rois: Tensor<float32>, batch_indices: Tensor<int64>, ?coordinate_transformation_mode: string, ?mode: string, ?output_height: int64, ?output_width: int64, ?sampling_ratio: int64, ?spatial_scale: float32) =
        MV() |> fun mv -> execNode<float32> "RoiAlign" [|mv.c(X); mv.c(rois); mv.c(batch_indices)|] ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.string("mode", mode, "avg"); Attr.int("output_height", output_height, 1L); Attr.int("output_width", output_width, 1L); Attr.int("sampling_ratio", sampling_ratio, 0L); Attr.float("spatial_scale", spatial_scale, 1.0f)|] |> Array.choose id)
    static member RoiAlign(X: Tensor<double>, rois: Tensor<double>, batch_indices: Tensor<int64>, ?coordinate_transformation_mode: string, ?mode: string, ?output_height: int64, ?output_width: int64, ?sampling_ratio: int64, ?spatial_scale: float32) =
        MV() |> fun mv -> execNode<double> "RoiAlign" [|mv.c(X); mv.c(rois); mv.c(batch_indices)|] ([|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.string("mode", mode, "avg"); Attr.int("output_height", output_height, 1L); Attr.int("output_width", output_width, 1L); Attr.int("sampling_ratio", sampling_ratio, 0L); Attr.float("spatial_scale", spatial_scale, 1.0f)|] |> Array.choose id)
    static member DequantizeLinear(x: Tensor<int8>, x_scale: Tensor<float32>, ?x_zero_point: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "DequantizeLinear" ([|Some(mv.c(x)); Some(mv.c(x_scale)); mv.c(x_zero_point)|] |> Array.choose id) ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member DequantizeLinear(x: Tensor<uint8>, x_scale: Tensor<float32>, ?x_zero_point: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "DequantizeLinear" ([|Some(mv.c(x)); Some(mv.c(x_scale)); mv.c(x_zero_point)|] |> Array.choose id) ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member DequantizeLinear(x: Tensor<int>, x_scale: Tensor<float32>, ?x_zero_point: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "DequantizeLinear" ([|Some(mv.c(x)); Some(mv.c(x_scale)); mv.c(x_zero_point)|] |> Array.choose id) ([|Attr.int("axis", axis, 1L)|] |> Array.choose id)
    static member DFT(input: Tensor<float32>, ?dft_length: Tensor<int>, ?axis: int64, ?inverse: int64, ?onesided: int64) =
        MV() |> fun mv -> execNode<float32> "DFT" ([|Some(mv.c(input)); mv.c(dft_length)|] |> Array.choose id) ([|Attr.int("axis", axis, 1L); Attr.int("inverse", inverse, 0L); Attr.int("onesided", onesided, 0L)|] |> Array.choose id)
    static member DFT(input: Tensor<float32>, ?dft_length: Tensor<int64>, ?axis: int64, ?inverse: int64, ?onesided: int64) =
        MV() |> fun mv -> execNode<float32> "DFT" ([|Some(mv.c(input)); mv.c(dft_length)|] |> Array.choose id) ([|Attr.int("axis", axis, 1L); Attr.int("inverse", inverse, 0L); Attr.int("onesided", onesided, 0L)|] |> Array.choose id)
    static member DFT(input: Tensor<double>, ?dft_length: Tensor<int>, ?axis: int64, ?inverse: int64, ?onesided: int64) =
        MV() |> fun mv -> execNode<double> "DFT" ([|Some(mv.c(input)); mv.c(dft_length)|] |> Array.choose id) ([|Attr.int("axis", axis, 1L); Attr.int("inverse", inverse, 0L); Attr.int("onesided", onesided, 0L)|] |> Array.choose id)
    static member DFT(input: Tensor<double>, ?dft_length: Tensor<int64>, ?axis: int64, ?inverse: int64, ?onesided: int64) =
        MV() |> fun mv -> execNode<double> "DFT" ([|Some(mv.c(input)); mv.c(dft_length)|] |> Array.choose id) ([|Attr.int("axis", axis, 1L); Attr.int("inverse", inverse, 0L); Attr.int("onesided", onesided, 0L)|] |> Array.choose id)
    static member QuantizeLinear(x: Tensor<float32>, y_scale: Tensor<float32>, ?y_zero_point: Tensor<int8>, ?axis: int64, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "QuantizeLinear" ([|Some(mv.c(x)); Some(mv.c(y_scale)); mv.c(y_zero_point)|] |> Array.choose id) ([|Attr.int("axis", axis, 1L); Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member QuantizeLinear(x: Tensor<float32>, y_scale: Tensor<float32>, ?y_zero_point: Tensor<uint8>, ?axis: int64, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "QuantizeLinear" ([|Some(mv.c(x)); Some(mv.c(y_scale)); mv.c(y_zero_point)|] |> Array.choose id) ([|Attr.int("axis", axis, 1L); Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member QuantizeLinear(x: Tensor<int>, y_scale: Tensor<int>, ?y_zero_point: Tensor<int8>, ?axis: int64, ?saturate: int64) =
        MV() |> fun mv -> execNode<int8> "QuantizeLinear" ([|Some(mv.c(x)); Some(mv.c(y_scale)); mv.c(y_zero_point)|] |> Array.choose id) ([|Attr.int("axis", axis, 1L); Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member QuantizeLinear(x: Tensor<int>, y_scale: Tensor<int>, ?y_zero_point: Tensor<uint8>, ?axis: int64, ?saturate: int64) =
        MV() |> fun mv -> execNode<uint8> "QuantizeLinear" ([|Some(mv.c(x)); Some(mv.c(y_scale)); mv.c(y_zero_point)|] |> Array.choose id) ([|Attr.int("axis", axis, 1L); Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
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
    static member MatMulInteger(A: Tensor<int8>, B: Tensor<int8>, ?a_zero_point: Tensor<int8>, ?b_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int> "MatMulInteger" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(a_zero_point); mv.c(b_zero_point)|] |> Array.choose id) [||]
    static member MatMulInteger(A: Tensor<int8>, B: Tensor<uint8>, ?a_zero_point: Tensor<int8>, ?b_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<int> "MatMulInteger" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(a_zero_point); mv.c(b_zero_point)|] |> Array.choose id) [||]
    static member MatMulInteger(A: Tensor<uint8>, B: Tensor<int8>, ?a_zero_point: Tensor<uint8>, ?b_zero_point: Tensor<int8>) =
        MV() |> fun mv -> execNode<int> "MatMulInteger" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(a_zero_point); mv.c(b_zero_point)|] |> Array.choose id) [||]
    static member MatMulInteger(A: Tensor<uint8>, B: Tensor<uint8>, ?a_zero_point: Tensor<uint8>, ?b_zero_point: Tensor<uint8>) =
        MV() |> fun mv -> execNode<int> "MatMulInteger" ([|Some(mv.c(A)); Some(mv.c(B)); mv.c(a_zero_point); mv.c(b_zero_point)|] |> Array.choose id) [||]
    static member Pad(data: Tensor<uint8>, pads: Tensor<int64>, ?constant_value: Tensor<uint8>, ?axes: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<uint8> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<uint8>, pads: Tensor<int64>, ?constant_value: Tensor<uint8>, ?axes: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<uint8> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<uint16>, pads: Tensor<int64>, ?constant_value: Tensor<uint16>, ?axes: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<uint16> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<uint16>, pads: Tensor<int64>, ?constant_value: Tensor<uint16>, ?axes: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<uint16> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<uint32>, pads: Tensor<int64>, ?constant_value: Tensor<uint32>, ?axes: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<uint32> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<uint32>, pads: Tensor<int64>, ?constant_value: Tensor<uint32>, ?axes: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<uint32> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<uint64>, pads: Tensor<int64>, ?constant_value: Tensor<uint64>, ?axes: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<uint64> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<uint64>, pads: Tensor<int64>, ?constant_value: Tensor<uint64>, ?axes: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<uint64> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<int8>, pads: Tensor<int64>, ?constant_value: Tensor<int8>, ?axes: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<int8> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<int8>, pads: Tensor<int64>, ?constant_value: Tensor<int8>, ?axes: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<int8> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<int16>, pads: Tensor<int64>, ?constant_value: Tensor<int16>, ?axes: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<int16> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<int16>, pads: Tensor<int64>, ?constant_value: Tensor<int16>, ?axes: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<int16> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<int>, pads: Tensor<int64>, ?constant_value: Tensor<int>, ?axes: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<int> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<int>, pads: Tensor<int64>, ?constant_value: Tensor<int>, ?axes: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<int> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<int64>, pads: Tensor<int64>, ?constant_value: Tensor<int64>, ?axes: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<int64> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<int64>, pads: Tensor<int64>, ?constant_value: Tensor<int64>, ?axes: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<int64> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<float32>, pads: Tensor<int64>, ?constant_value: Tensor<float32>, ?axes: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<float32> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<float32>, pads: Tensor<int64>, ?constant_value: Tensor<float32>, ?axes: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<float32> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<double>, pads: Tensor<int64>, ?constant_value: Tensor<double>, ?axes: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<double> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<double>, pads: Tensor<int64>, ?constant_value: Tensor<double>, ?axes: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<double> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<string>, pads: Tensor<int64>, ?constant_value: Tensor<string>, ?axes: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<string> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<string>, pads: Tensor<int64>, ?constant_value: Tensor<string>, ?axes: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<string> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<bool>, pads: Tensor<int64>, ?constant_value: Tensor<bool>, ?axes: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<bool> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<bool>, pads: Tensor<int64>, ?constant_value: Tensor<bool>, ?axes: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<bool> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<Complex>, pads: Tensor<int64>, ?constant_value: Tensor<Complex>, ?axes: Tensor<int>, ?mode: string) =
        MV() |> fun mv -> execNode<Complex> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member Pad(data: Tensor<Complex>, pads: Tensor<int64>, ?constant_value: Tensor<Complex>, ?axes: Tensor<int64>, ?mode: string) =
        MV() |> fun mv -> execNode<Complex> "Pad" ([|Some(mv.c(data)); Some(mv.c(pads)); mv.c(constant_value); mv.c(axes)|] |> Array.choose id) ([|Attr.string("mode", mode, "constant")|] |> Array.choose id)
    static member MaxUnpool(X: Tensor<float32>, I: Tensor<int64>, kernel_shape: int64[], ?output_shape: Tensor<int64>, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<float32> "MaxUnpool" ([|Some(mv.c(X)); Some(mv.c(I)); mv.c(output_shape)|] |> Array.choose id) ([|Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member MaxUnpool(X: Tensor<double>, I: Tensor<int64>, kernel_shape: int64[], ?output_shape: Tensor<int64>, ?pads: int64[], ?strides: int64[]) =
        MV() |> fun mv -> execNode<double> "MaxUnpool" ([|Some(mv.c(X)); Some(mv.c(I)); mv.c(output_shape)|] |> Array.choose id) ([|Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|] |> Array.choose id)
    static member Slice(data: Tensor<uint8>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<uint8> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<uint8>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<uint16>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<uint16> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<uint16>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint16> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<uint32>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<uint32> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<uint32>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint32> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<uint64>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<uint64> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<uint64>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint64> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<int8>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<int8> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<int8>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<int16>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<int16> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<int16>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<int16> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
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
    static member Slice(data: Tensor<double>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<double> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<double>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<string>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<string> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<string>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<bool>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<bool> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<bool>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<Complex>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        MV() |> fun mv -> execNode<Complex> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member Slice(data: Tensor<Complex>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        MV() |> fun mv -> execNode<Complex> "Slice" ([|Some(mv.c(data)); Some(mv.c(starts)); Some(mv.c(ends)); mv.c(axes); mv.c(steps)|] |> Array.choose id) [||]
    static member NegativeLogLikelihoodLoss(input: Tensor<float32>, target: Tensor<int>, ?weight: Tensor<float32>, ?ignore_index: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<float32> "NegativeLogLikelihoodLoss" ([|Some(mv.c(input)); Some(mv.c(target)); mv.c(weight)|] |> Array.choose id) ([|Attr.int("ignore_index", ignore_index); Attr.string("reduction", reduction, "mean")|] |> Array.choose id)
    static member NegativeLogLikelihoodLoss(input: Tensor<float32>, target: Tensor<int64>, ?weight: Tensor<float32>, ?ignore_index: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<float32> "NegativeLogLikelihoodLoss" ([|Some(mv.c(input)); Some(mv.c(target)); mv.c(weight)|] |> Array.choose id) ([|Attr.int("ignore_index", ignore_index); Attr.string("reduction", reduction, "mean")|] |> Array.choose id)
    static member NegativeLogLikelihoodLoss(input: Tensor<double>, target: Tensor<int>, ?weight: Tensor<double>, ?ignore_index: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<double> "NegativeLogLikelihoodLoss" ([|Some(mv.c(input)); Some(mv.c(target)); mv.c(weight)|] |> Array.choose id) ([|Attr.int("ignore_index", ignore_index); Attr.string("reduction", reduction, "mean")|] |> Array.choose id)
    static member NegativeLogLikelihoodLoss(input: Tensor<double>, target: Tensor<int64>, ?weight: Tensor<double>, ?ignore_index: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<double> "NegativeLogLikelihoodLoss" ([|Some(mv.c(input)); Some(mv.c(target)); mv.c(weight)|] |> Array.choose id) ([|Attr.int("ignore_index", ignore_index); Attr.string("reduction", reduction, "mean")|] |> Array.choose id)
    static member Gather(data: Tensor<uint8>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<uint8>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<uint16>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<uint16>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<uint32>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<uint32>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<uint64>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<uint64>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<int8>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<int8>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<int16>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<int16>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
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
    static member Gather(data: Tensor<double>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<double>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<string>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<string>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<bool>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<bool>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<Complex>, indices: Tensor<int>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Gather(data: Tensor<Complex>, indices: Tensor<int64>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "Gather" [|mv.c(data); mv.c(indices)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member ScatterElements(data: Tensor<uint8>, indices: Tensor<int>, updates: Tensor<uint8>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<uint8> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<uint8>, indices: Tensor<int64>, updates: Tensor<uint8>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<uint8> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<uint16>, indices: Tensor<int>, updates: Tensor<uint16>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<uint16> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<uint16>, indices: Tensor<int64>, updates: Tensor<uint16>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<uint16> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<uint32>, indices: Tensor<int>, updates: Tensor<uint32>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<uint32> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<uint32>, indices: Tensor<int64>, updates: Tensor<uint32>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<uint32> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<uint64>, indices: Tensor<int>, updates: Tensor<uint64>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<uint64> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<uint64>, indices: Tensor<int64>, updates: Tensor<uint64>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<uint64> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int8>, indices: Tensor<int>, updates: Tensor<int8>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<int8> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int8>, indices: Tensor<int64>, updates: Tensor<int8>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<int8> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int16>, indices: Tensor<int>, updates: Tensor<int16>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<int16> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int16>, indices: Tensor<int64>, updates: Tensor<int16>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<int16> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int>, indices: Tensor<int>, updates: Tensor<int>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<int> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int>, indices: Tensor<int64>, updates: Tensor<int>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<int> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int64>, indices: Tensor<int>, updates: Tensor<int64>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<int64> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<int64>, indices: Tensor<int64>, updates: Tensor<int64>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<int64> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<float32>, indices: Tensor<int>, updates: Tensor<float32>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<float32> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<float32>, indices: Tensor<int64>, updates: Tensor<float32>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<float32> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<double>, indices: Tensor<int>, updates: Tensor<double>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<double> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<double>, indices: Tensor<int64>, updates: Tensor<double>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<double> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<string>, indices: Tensor<int>, updates: Tensor<string>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<string> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<string>, indices: Tensor<int64>, updates: Tensor<string>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<string> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<bool>, indices: Tensor<int>, updates: Tensor<bool>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<bool> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<bool>, indices: Tensor<int64>, updates: Tensor<bool>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<bool> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<Complex>, indices: Tensor<int>, updates: Tensor<Complex>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<Complex> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member ScatterElements(data: Tensor<Complex>, indices: Tensor<int64>, updates: Tensor<Complex>, ?axis: int64, ?reduction: string) =
        MV() |> fun mv -> execNode<Complex> "ScatterElements" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L); Attr.string("reduction", reduction, "none")|] |> Array.choose id)
    static member STFT(signal: Tensor<float32>, frame_step: Tensor<int>, ?window: Tensor<float32>, ?frame_length: Tensor<int>, ?onesided: int64) =
        MV() |> fun mv -> execNode<float32> "STFT" ([|Some(mv.c(signal)); Some(mv.c(frame_step)); mv.c(window); mv.c(frame_length)|] |> Array.choose id) ([|Attr.int("onesided", onesided, 1L)|] |> Array.choose id)
    static member STFT(signal: Tensor<float32>, frame_step: Tensor<int64>, ?window: Tensor<float32>, ?frame_length: Tensor<int64>, ?onesided: int64) =
        MV() |> fun mv -> execNode<float32> "STFT" ([|Some(mv.c(signal)); Some(mv.c(frame_step)); mv.c(window); mv.c(frame_length)|] |> Array.choose id) ([|Attr.int("onesided", onesided, 1L)|] |> Array.choose id)
    static member STFT(signal: Tensor<double>, frame_step: Tensor<int>, ?window: Tensor<double>, ?frame_length: Tensor<int>, ?onesided: int64) =
        MV() |> fun mv -> execNode<double> "STFT" ([|Some(mv.c(signal)); Some(mv.c(frame_step)); mv.c(window); mv.c(frame_length)|] |> Array.choose id) ([|Attr.int("onesided", onesided, 1L)|] |> Array.choose id)
    static member STFT(signal: Tensor<double>, frame_step: Tensor<int64>, ?window: Tensor<double>, ?frame_length: Tensor<int64>, ?onesided: int64) =
        MV() |> fun mv -> execNode<double> "STFT" ([|Some(mv.c(signal)); Some(mv.c(frame_step)); mv.c(window); mv.c(frame_length)|] |> Array.choose id) ([|Attr.int("onesided", onesided, 1L)|] |> Array.choose id)
    static member Tile(input: Tensor<uint8>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint8> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<uint16>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint16> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<uint32>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint32> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<uint64>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<uint64> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<int8>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<int8> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<int16>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<int16> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<int>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<int64>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<float32>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<double>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<string>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<string> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<bool>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<bool> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Tile(input: Tensor<Complex>, repeats: Tensor<int64>) =
        MV() |> fun mv -> execNode<Complex> "Tile" [|mv.c(input); mv.c(repeats)|] [||]
    static member Pow(X: Tensor<int>, Y: Tensor<uint8>) =
        MV() |> fun mv -> execNode<int> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int>, Y: Tensor<uint16>) =
        MV() |> fun mv -> execNode<int> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int>, Y: Tensor<uint32>) =
        MV() |> fun mv -> execNode<int> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int>, Y: Tensor<uint64>) =
        MV() |> fun mv -> execNode<int> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int>, Y: Tensor<int8>) =
        MV() |> fun mv -> execNode<int> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int>, Y: Tensor<int16>) =
        MV() |> fun mv -> execNode<int> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int>, Y: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<int> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int>, Y: Tensor<float32>) =
        MV() |> fun mv -> execNode<int> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int>, Y: Tensor<double>) =
        MV() |> fun mv -> execNode<int> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int64>, Y: Tensor<uint8>) =
        MV() |> fun mv -> execNode<int64> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int64>, Y: Tensor<uint16>) =
        MV() |> fun mv -> execNode<int64> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int64>, Y: Tensor<uint32>) =
        MV() |> fun mv -> execNode<int64> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int64>, Y: Tensor<uint64>) =
        MV() |> fun mv -> execNode<int64> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int64>, Y: Tensor<int8>) =
        MV() |> fun mv -> execNode<int64> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int64>, Y: Tensor<int16>) =
        MV() |> fun mv -> execNode<int64> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int64>, Y: Tensor<int>) =
        MV() |> fun mv -> execNode<int64> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int64>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int64>, Y: Tensor<float32>) =
        MV() |> fun mv -> execNode<int64> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<int64>, Y: Tensor<double>) =
        MV() |> fun mv -> execNode<int64> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<float32>, Y: Tensor<uint8>) =
        MV() |> fun mv -> execNode<float32> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<float32>, Y: Tensor<uint16>) =
        MV() |> fun mv -> execNode<float32> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<float32>, Y: Tensor<uint32>) =
        MV() |> fun mv -> execNode<float32> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<float32>, Y: Tensor<uint64>) =
        MV() |> fun mv -> execNode<float32> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<float32>, Y: Tensor<int8>) =
        MV() |> fun mv -> execNode<float32> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<float32>, Y: Tensor<int16>) =
        MV() |> fun mv -> execNode<float32> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<float32>, Y: Tensor<int>) =
        MV() |> fun mv -> execNode<float32> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<float32>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<float32> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<float32>, Y: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<float32>, Y: Tensor<double>) =
        MV() |> fun mv -> execNode<float32> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<double>, Y: Tensor<uint8>) =
        MV() |> fun mv -> execNode<double> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<double>, Y: Tensor<uint16>) =
        MV() |> fun mv -> execNode<double> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<double>, Y: Tensor<uint32>) =
        MV() |> fun mv -> execNode<double> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<double>, Y: Tensor<uint64>) =
        MV() |> fun mv -> execNode<double> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<double>, Y: Tensor<int8>) =
        MV() |> fun mv -> execNode<double> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<double>, Y: Tensor<int16>) =
        MV() |> fun mv -> execNode<double> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<double>, Y: Tensor<int>) =
        MV() |> fun mv -> execNode<double> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<double>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<double> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<double>, Y: Tensor<float32>) =
        MV() |> fun mv -> execNode<double> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Pow(X: Tensor<double>, Y: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Pow" [|mv.c(X); mv.c(Y)|] [||]
    static member Compress(input: Tensor<uint8>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<uint16>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<uint32>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<uint64>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<int8>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<int16>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<int>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<int> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<int64>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<int64> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<float32>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<float32> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<double>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<string>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<bool>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Compress(input: Tensor<Complex>, condition: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "Compress" [|mv.c(input); mv.c(condition)|] ([|Attr.int("axis", axis)|] |> Array.choose id)
    static member Scatter(data: Tensor<uint8>, indices: Tensor<int>, updates: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<uint8>, indices: Tensor<int64>, updates: Tensor<uint8>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint8> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<uint16>, indices: Tensor<int>, updates: Tensor<uint16>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<uint16>, indices: Tensor<int64>, updates: Tensor<uint16>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint16> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<uint32>, indices: Tensor<int>, updates: Tensor<uint32>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<uint32>, indices: Tensor<int64>, updates: Tensor<uint32>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint32> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<uint64>, indices: Tensor<int>, updates: Tensor<uint64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<uint64>, indices: Tensor<int64>, updates: Tensor<uint64>, ?axis: int64) =
        MV() |> fun mv -> execNode<uint64> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<int8>, indices: Tensor<int>, updates: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<int8>, indices: Tensor<int64>, updates: Tensor<int8>, ?axis: int64) =
        MV() |> fun mv -> execNode<int8> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<int16>, indices: Tensor<int>, updates: Tensor<int16>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<int16>, indices: Tensor<int64>, updates: Tensor<int16>, ?axis: int64) =
        MV() |> fun mv -> execNode<int16> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
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
    static member Scatter(data: Tensor<double>, indices: Tensor<int>, updates: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<double>, indices: Tensor<int64>, updates: Tensor<double>, ?axis: int64) =
        MV() |> fun mv -> execNode<double> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<string>, indices: Tensor<int>, updates: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<string>, indices: Tensor<int64>, updates: Tensor<string>, ?axis: int64) =
        MV() |> fun mv -> execNode<string> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<bool>, indices: Tensor<int>, updates: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<bool>, indices: Tensor<int64>, updates: Tensor<bool>, ?axis: int64) =
        MV() |> fun mv -> execNode<bool> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<Complex>, indices: Tensor<int>, updates: Tensor<Complex>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member Scatter(data: Tensor<Complex>, indices: Tensor<int64>, updates: Tensor<Complex>, ?axis: int64) =
        MV() |> fun mv -> execNode<Complex> "Scatter" [|mv.c(data); mv.c(indices); mv.c(updates)|] ([|Attr.int("axis", axis, 0L)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<uint8>, shape: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<uint8> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<uint8>, shape: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<uint8> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<uint16>, shape: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<uint16> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<uint16>, shape: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<uint16> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<uint32>, shape: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<uint32> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<uint32>, shape: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<uint32> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<uint64>, shape: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<uint64> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<uint64>, shape: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<uint64> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<int8>, shape: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int8> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<int8>, shape: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int8> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<int16>, shape: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int16> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<int16>, shape: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int16> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<int>, shape: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<int>, shape: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<int64>, shape: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int64> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<int64>, shape: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<int64> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<float32>, shape: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<float32> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<float32>, shape: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<float32> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<double>, shape: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<double> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<double>, shape: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<double> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<string>, shape: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<string> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<string>, shape: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<string> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<bool>, shape: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<bool> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<bool>, shape: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<bool> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<Complex>, shape: Tensor<int>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<Complex> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member CenterCropPad(input_data: Tensor<Complex>, shape: Tensor<int64>, ?axes: int64[]) =
        MV() |> fun mv -> execNode<Complex> "CenterCropPad" [|mv.c(input_data); mv.c(shape)|] ([|Attr.ints("axes", axes)|] |> Array.choose id)
    static member Where(condition: Tensor<bool>, X: Tensor<uint8>, Y: Tensor<uint8>) =
        MV() |> fun mv -> execNode<uint8> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<uint16>, Y: Tensor<uint16>) =
        MV() |> fun mv -> execNode<uint16> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<uint32>, Y: Tensor<uint32>) =
        MV() |> fun mv -> execNode<uint32> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<uint64>, Y: Tensor<uint64>) =
        MV() |> fun mv -> execNode<uint64> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<int8>, Y: Tensor<int8>) =
        MV() |> fun mv -> execNode<int8> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<int16>, Y: Tensor<int16>) =
        MV() |> fun mv -> execNode<int16> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<int>, Y: Tensor<int>) =
        MV() |> fun mv -> execNode<int> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<int64>, Y: Tensor<int64>) =
        MV() |> fun mv -> execNode<int64> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<float32>, Y: Tensor<float32>) =
        MV() |> fun mv -> execNode<float32> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<double>, Y: Tensor<double>) =
        MV() |> fun mv -> execNode<double> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<string>, Y: Tensor<string>) =
        MV() |> fun mv -> execNode<string> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<bool>, Y: Tensor<bool>) =
        MV() |> fun mv -> execNode<bool> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Where(condition: Tensor<bool>, X: Tensor<Complex>, Y: Tensor<Complex>) =
        MV() |> fun mv -> execNode<Complex> "Where" [|mv.c(condition); mv.c(X); mv.c(Y)|] [||]
    static member Resize(X: Tensor<uint8>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint8> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<uint8>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint8> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<uint16>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint16> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<uint16>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint16> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<uint32>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint32> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<uint32>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint32> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<uint64>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint64> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<uint64>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<uint64> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<int8>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int8> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<int8>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int8> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<int16>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int16> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<int16>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int16> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<int>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<int>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<int64>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int64> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<int64>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<int64> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<float32>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<float32> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<float32>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<float32> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<double>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<double> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<double>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<double> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<string>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<string> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<string>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<string> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<bool>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<bool> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<bool>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<bool> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<Complex>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<Complex> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
    static member Resize(X: Tensor<Complex>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        MV() |> fun mv -> execNode<Complex> "Resize" ([|Some(mv.c(X)); mv.c(roi); mv.c(scales); mv.c(sizes)|] |> Array.choose id) ([|Attr.int("antialias", antialias, 0L); Attr.ints("axes", axes); Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("keep_aspect_ratio_policy", keep_aspect_ratio_policy, "stretch"); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|] |> Array.choose id)
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
    static member EyeLike<'a>(input: Tensor<uint8>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<uint8>, ?k: int64) =
        execNodeCheck<uint8> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<int16>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<int16>, ?k: int64) =
        execNodeCheck<int16> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<float32>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<float32>, ?k: int64) =
        execNodeCheck<float32> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<uint32>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<uint32>, ?k: int64) =
        execNodeCheck<uint32> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<int8>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<int8>, ?k: int64) =
        execNodeCheck<int8> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<uint64>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<uint64>, ?k: int64) =
        execNodeCheck<uint64> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<int>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<int>, ?k: int64) =
        execNodeCheck<int> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<int64>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<int64>, ?k: int64) =
        execNodeCheck<int64> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<uint16>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<uint16>, ?k: int64) =
        execNodeCheck<uint16> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<bool>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<bool>, ?k: int64) =
        execNodeCheck<bool> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike<'a>(input: Tensor<double>, ?k: int64) =
        execNodeCheck<'a> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member EyeLike(input: Tensor<double>, ?k: int64) =
        execNodeCheck<double> "EyeLike" [|MV.mv(1,input)|] [|2L; 5L; 1L; 12L; 3L; 13L; 6L; 7L; 4L; 9L; 11L|] ([|Attr.int("k", k, 0L)|] |> Array.choose id)
    static member Multinomial<'a>(input: Tensor<float32>, ?sample_size: int64, ?seed: float32) =
        execNodeCheck<'a> "Multinomial" [|MV.mv(1,input)|] [|6L; 7L|] ([|Attr.int("sample_size", sample_size, 1L); Attr.float("seed", seed)|] |> Array.choose id)
    static member Multinomial(input: Tensor<float32>, ?sample_size: int64, ?seed: float32) =
        execNodeCheck<float32> "Multinomial" [|MV.mv(1,input)|] [|6L; 7L|] ([|Attr.int("sample_size", sample_size, 1L); Attr.float("seed", seed)|] |> Array.choose id)
    static member Multinomial<'a>(input: Tensor<double>, ?sample_size: int64, ?seed: float32) =
        execNodeCheck<'a> "Multinomial" [|MV.mv(1,input)|] [|6L; 7L|] ([|Attr.int("sample_size", sample_size, 1L); Attr.float("seed", seed)|] |> Array.choose id)
    static member Multinomial(input: Tensor<double>, ?sample_size: int64, ?seed: float32) =
        execNodeCheck<double> "Multinomial" [|MV.mv(1,input)|] [|6L; 7L|] ([|Attr.int("sample_size", sample_size, 1L); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<uint8>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<uint8>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<uint8> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<int16>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<int16>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<int16> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<int64>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<int64>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<int64> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<int8>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<int8>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<int8> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<float32>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<float32>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<float32> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<string>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<string>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<string> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<uint64>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<uint64>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<uint64> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<int>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<int>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<int> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<Complex>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<Complex>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<Complex> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<uint32>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<uint32>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<uint32> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<uint16>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<uint16>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<uint16> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<bool>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<bool>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<bool> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike<'a>(input: Tensor<double>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniformLike(input: Tensor<double>, ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<double> "RandomUniformLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<uint8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<uint8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<uint8> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<int16>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<int16>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<int16> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<int64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<int64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<int64> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<int8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<int8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<int8> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<float32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<float32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<float32> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<string>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<string>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<string> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<uint64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<uint64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<uint64> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<int>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<int>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<int> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<Complex>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<Complex>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<Complex> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<uint32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<uint32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<uint32> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<uint16>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<uint16>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<uint16> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<bool>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<bool>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<bool> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike<'a>(input: Tensor<double>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormalLike(input: Tensor<double>, ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<double> "RandomNormalLike" [|MV.mv(1,input)|] [|1L; 11L|] ([|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomNormal<'a>(shape: int64[], ?mean: float32, ?scale: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomNormal" [||] [|1L; 11L|] ([|Attr.ints("shape", shape); Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member RandomUniform<'a>(shape: int64[], ?high: float32, ?low: float32, ?seed: float32) =
        execNodeCheck<'a> "RandomUniform" [||] [|1L; 11L|] ([|Attr.ints("shape", shape); Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|] |> Array.choose id)
    static member Cast<'a>(input: Tensor<uint8>, ?saturate: int64) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|2L; 5L; 7L; 3L; 13L; 1L; 6L; 8L; 12L; 4L; 9L; 11L|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member Cast<'a>(input: Tensor<int16>, ?saturate: int64) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|2L; 5L; 7L; 3L; 13L; 1L; 6L; 8L; 12L; 4L; 9L; 11L|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member Cast<'a>(input: Tensor<int64>, ?saturate: int64) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|2L; 5L; 7L; 3L; 13L; 1L; 6L; 8L; 12L; 4L; 9L; 11L|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member Cast<'a>(input: Tensor<int8>, ?saturate: int64) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|2L; 5L; 7L; 3L; 13L; 1L; 6L; 8L; 12L; 4L; 9L; 11L|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member Cast<'a>(input: Tensor<uint64>, ?saturate: int64) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|2L; 5L; 7L; 3L; 13L; 1L; 6L; 8L; 12L; 4L; 9L; 11L|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member Cast<'a>(input: Tensor<float32>, ?saturate: int64) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|2L; 5L; 7L; 3L; 13L; 1L; 6L; 8L; 12L; 4L; 9L; 11L|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member Cast<'a>(input: Tensor<int>, ?saturate: int64) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|2L; 5L; 7L; 3L; 13L; 1L; 6L; 8L; 12L; 4L; 9L; 11L|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member Cast<'a>(input: Tensor<string>, ?saturate: int64) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|2L; 5L; 7L; 3L; 13L; 1L; 6L; 8L; 12L; 4L; 9L; 11L|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member Cast<'a>(input: Tensor<uint32>, ?saturate: int64) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|2L; 5L; 7L; 3L; 13L; 1L; 6L; 8L; 12L; 4L; 9L; 11L|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member Cast<'a>(input: Tensor<uint16>, ?saturate: int64) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|2L; 5L; 7L; 3L; 13L; 1L; 6L; 8L; 12L; 4L; 9L; 11L|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member Cast<'a>(input: Tensor<bool>, ?saturate: int64) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|2L; 5L; 7L; 3L; 13L; 1L; 6L; 8L; 12L; 4L; 9L; 11L|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member Cast<'a>(input: Tensor<double>, ?saturate: int64) =
        execNodeCheck<'a> "Cast" [|MV.mv(1,input)|] [|2L; 5L; 7L; 3L; 13L; 1L; 6L; 8L; 12L; 4L; 9L; 11L|] ([|Attr.int("saturate", saturate, 1L)|] |> Array.choose id)
    static member LSTM(X: Tensor<float32>, W: Tensor<float32>, R: Tensor<float32>, ?B: Tensor<float32>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<float32>, ?initial_c: Tensor<float32>, ?P: Tensor<float32>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?input_forget: int64, ?layout: int64) =
        MV() |> fun mv -> execNodeTuple3<float32, float32, float32> "LSTM" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h); mv.c(initial_c); mv.c(P)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("input_forget", input_forget, 0L); Attr.int("layout", layout, 0L)|] |> Array.choose id)
    static member LSTM(X: Tensor<double>, W: Tensor<double>, R: Tensor<double>, ?B: Tensor<double>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<double>, ?initial_c: Tensor<double>, ?P: Tensor<double>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?input_forget: int64, ?layout: int64) =
        MV() |> fun mv -> execNodeTuple3<double, double, double> "LSTM" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h); mv.c(initial_c); mv.c(P)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("input_forget", input_forget, 0L); Attr.int("layout", layout, 0L)|] |> Array.choose id)
    static member LinearClassifier(X: Tensor<float32>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "LinearClassifier" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("intercepts", intercepts); Attr.int("multi_class", multi_class, 0L); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member LinearClassifier(X: Tensor<double>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "LinearClassifier" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("intercepts", intercepts); Attr.int("multi_class", multi_class, 0L); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member LinearClassifier(X: Tensor<int64>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "LinearClassifier" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("intercepts", intercepts); Attr.int("multi_class", multi_class, 0L); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member LinearClassifier(X: Tensor<int>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "LinearClassifier" [|mv.c(X)|] ([|Attr.floats("coefficients", coefficients); Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("intercepts", intercepts); Attr.int("multi_class", multi_class, 0L); Attr.string("post_transform", post_transform, "NONE")|] |> Array.choose id)
    static member SVMClassifier(X: Tensor<float32>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "SVMClassifier" [|mv.c(X)|] ([|Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("prob_a", prob_a); Attr.floats("prob_b", prob_b); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors); Attr.ints("vectors_per_class", vectors_per_class)|] |> Array.choose id)
    static member SVMClassifier(X: Tensor<double>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "SVMClassifier" [|mv.c(X)|] ([|Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("prob_a", prob_a); Attr.floats("prob_b", prob_b); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors); Attr.ints("vectors_per_class", vectors_per_class)|] |> Array.choose id)
    static member SVMClassifier(X: Tensor<int64>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "SVMClassifier" [|mv.c(X)|] ([|Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("prob_a", prob_a); Attr.floats("prob_b", prob_b); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors); Attr.ints("vectors_per_class", vectors_per_class)|] |> Array.choose id)
    static member SVMClassifier(X: Tensor<int>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        MV() |> fun mv -> execNodeTuple2<int64, float32> "SVMClassifier" [|mv.c(X)|] ([|Attr.ints("classlabels_ints", classlabels_ints); Attr.strings("classlabels_strings", classlabels_strings); Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("prob_a", prob_a); Attr.floats("prob_b", prob_b); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors); Attr.ints("vectors_per_class", vectors_per_class)|] |> Array.choose id)
    static member MaxPool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?pads: int64[], ?storage_order: int64, ?strides: int64[]) =
        MV() |> fun mv -> execNodeTuple2<float32, int64> "MaxPool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.int("storage_order", storage_order, 0L); Attr.ints("strides", strides)|] |> Array.choose id)
    static member MaxPool(X: Tensor<double>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?pads: int64[], ?storage_order: int64, ?strides: int64[]) =
        MV() |> fun mv -> execNodeTuple2<double, int64> "MaxPool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.int("storage_order", storage_order, 0L); Attr.ints("strides", strides)|] |> Array.choose id)
    static member MaxPool(X: Tensor<int8>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?pads: int64[], ?storage_order: int64, ?strides: int64[]) =
        MV() |> fun mv -> execNodeTuple2<int8, int64> "MaxPool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.int("storage_order", storage_order, 0L); Attr.ints("strides", strides)|] |> Array.choose id)
    static member MaxPool(X: Tensor<uint8>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?pads: int64[], ?storage_order: int64, ?strides: int64[]) =
        MV() |> fun mv -> execNodeTuple2<uint8, int64> "MaxPool" [|mv.c(X)|] ([|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.int("storage_order", storage_order, 0L); Attr.ints("strides", strides)|] |> Array.choose id)
    static member GRU(X: Tensor<float32>, W: Tensor<float32>, R: Tensor<float32>, ?B: Tensor<float32>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<float32>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?layout: int64, ?linear_before_reset: int64) =
        MV() |> fun mv -> execNodeTuple2<float32, float32> "GRU" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("layout", layout, 0L); Attr.int("linear_before_reset", linear_before_reset, 0L)|] |> Array.choose id)
    static member GRU(X: Tensor<double>, W: Tensor<double>, R: Tensor<double>, ?B: Tensor<double>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<double>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?layout: int64, ?linear_before_reset: int64) =
        MV() |> fun mv -> execNodeTuple2<double, double> "GRU" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("layout", layout, 0L); Attr.int("linear_before_reset", linear_before_reset, 0L)|] |> Array.choose id)
    static member TopK(X: Tensor<uint8>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<uint8, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member TopK(X: Tensor<uint16>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<uint16, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member TopK(X: Tensor<uint32>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<uint32, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member TopK(X: Tensor<uint64>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<uint64, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member TopK(X: Tensor<int8>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<int8, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member TopK(X: Tensor<int16>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<int16, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member TopK(X: Tensor<int>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<int, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member TopK(X: Tensor<int64>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<int64, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member TopK(X: Tensor<float32>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<float32, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member TopK(X: Tensor<double>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple2<double, int64> "TopK" [|mv.c(X); mv.c(K)|] ([|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Dropout(data: Tensor<float32>, ?ratio: Tensor<float32>, ?training_mode: Tensor<bool>, ?seed: int64) =
        MV() |> fun mv -> execNodeTuple2<float32, bool> "Dropout" ([|Some(mv.c(data)); mv.c(ratio); mv.c(training_mode)|] |> Array.choose id) ([|Attr.int("seed", seed)|] |> Array.choose id)
    static member Dropout(data: Tensor<float32>, ?ratio: Tensor<double>, ?training_mode: Tensor<bool>, ?seed: int64) =
        MV() |> fun mv -> execNodeTuple2<float32, bool> "Dropout" ([|Some(mv.c(data)); mv.c(ratio); mv.c(training_mode)|] |> Array.choose id) ([|Attr.int("seed", seed)|] |> Array.choose id)
    static member Dropout(data: Tensor<double>, ?ratio: Tensor<float32>, ?training_mode: Tensor<bool>, ?seed: int64) =
        MV() |> fun mv -> execNodeTuple2<double, bool> "Dropout" ([|Some(mv.c(data)); mv.c(ratio); mv.c(training_mode)|] |> Array.choose id) ([|Attr.int("seed", seed)|] |> Array.choose id)
    static member Dropout(data: Tensor<double>, ?ratio: Tensor<double>, ?training_mode: Tensor<bool>, ?seed: int64) =
        MV() |> fun mv -> execNodeTuple2<double, bool> "Dropout" ([|Some(mv.c(data)); mv.c(ratio); mv.c(training_mode)|] |> Array.choose id) ([|Attr.int("seed", seed)|] |> Array.choose id)
    static member Unique(X: Tensor<uint8>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<uint8, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<uint16>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<uint16, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<uint32>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<uint32, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<uint64>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<uint64, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<int8>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<int8, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<int16>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<int16, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<int>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<int, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<int64>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<int64, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<float32>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<float32, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<double>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<double, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<string>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<string, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<bool>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<bool, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member Unique(X: Tensor<Complex>, ?axis: int64, ?sorted: int64) =
        MV() |> fun mv -> execNodeTuple4<Complex, int64, int64, int64> "Unique" [|mv.c(X)|] ([|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|] |> Array.choose id)
    static member DynamicQuantizeLinear(x: Tensor<float32>) =
        MV() |> fun mv -> execNodeTuple3<uint8, float32, uint8> "DynamicQuantizeLinear" [|mv.c(x)|] [||]
    static member RNN(X: Tensor<float32>, W: Tensor<float32>, R: Tensor<float32>, ?B: Tensor<float32>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<float32>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?layout: int64) =
        MV() |> fun mv -> execNodeTuple2<float32, float32> "RNN" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations, [|"Tanh";"Tanh"|]); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("layout", layout, 0L)|] |> Array.choose id)
    static member RNN(X: Tensor<double>, W: Tensor<double>, R: Tensor<double>, ?B: Tensor<double>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<double>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?layout: int64) =
        MV() |> fun mv -> execNodeTuple2<double, double> "RNN" ([|Some(mv.c(X)); Some(mv.c(W)); Some(mv.c(R)); mv.c(B); mv.c(sequence_lens); mv.c(initial_h)|] |> Array.choose id) ([|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations, [|"Tanh";"Tanh"|]); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("layout", layout, 0L)|] |> Array.choose id)
    static member BatchNormalization(X: Tensor<float32>, scale: Tensor<float32>, B: Tensor<float32>, input_mean: Tensor<float32>, input_var: Tensor<float32>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        MV() |> fun mv -> execNodeTuple3<float32, float32, float32> "BatchNormalization" [|mv.c(X); mv.c(scale); mv.c(B); mv.c(input_mean); mv.c(input_var)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f); Attr.float("momentum", momentum, 0.8999999761581421f); Attr.int("training_mode", training_mode, 0L)|] |> Array.choose id)
    static member BatchNormalization(X: Tensor<float32>, scale: Tensor<float32>, B: Tensor<float32>, input_mean: Tensor<double>, input_var: Tensor<double>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        MV() |> fun mv -> execNodeTuple3<float32, double, double> "BatchNormalization" [|mv.c(X); mv.c(scale); mv.c(B); mv.c(input_mean); mv.c(input_var)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f); Attr.float("momentum", momentum, 0.8999999761581421f); Attr.int("training_mode", training_mode, 0L)|] |> Array.choose id)
    static member BatchNormalization(X: Tensor<float32>, scale: Tensor<double>, B: Tensor<double>, input_mean: Tensor<float32>, input_var: Tensor<float32>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        MV() |> fun mv -> execNodeTuple3<float32, float32, float32> "BatchNormalization" [|mv.c(X); mv.c(scale); mv.c(B); mv.c(input_mean); mv.c(input_var)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f); Attr.float("momentum", momentum, 0.8999999761581421f); Attr.int("training_mode", training_mode, 0L)|] |> Array.choose id)
    static member BatchNormalization(X: Tensor<float32>, scale: Tensor<double>, B: Tensor<double>, input_mean: Tensor<double>, input_var: Tensor<double>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        MV() |> fun mv -> execNodeTuple3<float32, double, double> "BatchNormalization" [|mv.c(X); mv.c(scale); mv.c(B); mv.c(input_mean); mv.c(input_var)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f); Attr.float("momentum", momentum, 0.8999999761581421f); Attr.int("training_mode", training_mode, 0L)|] |> Array.choose id)
    static member BatchNormalization(X: Tensor<double>, scale: Tensor<float32>, B: Tensor<float32>, input_mean: Tensor<float32>, input_var: Tensor<float32>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        MV() |> fun mv -> execNodeTuple3<double, float32, float32> "BatchNormalization" [|mv.c(X); mv.c(scale); mv.c(B); mv.c(input_mean); mv.c(input_var)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f); Attr.float("momentum", momentum, 0.8999999761581421f); Attr.int("training_mode", training_mode, 0L)|] |> Array.choose id)
    static member BatchNormalization(X: Tensor<double>, scale: Tensor<float32>, B: Tensor<float32>, input_mean: Tensor<double>, input_var: Tensor<double>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        MV() |> fun mv -> execNodeTuple3<double, double, double> "BatchNormalization" [|mv.c(X); mv.c(scale); mv.c(B); mv.c(input_mean); mv.c(input_var)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f); Attr.float("momentum", momentum, 0.8999999761581421f); Attr.int("training_mode", training_mode, 0L)|] |> Array.choose id)
    static member BatchNormalization(X: Tensor<double>, scale: Tensor<double>, B: Tensor<double>, input_mean: Tensor<float32>, input_var: Tensor<float32>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        MV() |> fun mv -> execNodeTuple3<double, float32, float32> "BatchNormalization" [|mv.c(X); mv.c(scale); mv.c(B); mv.c(input_mean); mv.c(input_var)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f); Attr.float("momentum", momentum, 0.8999999761581421f); Attr.int("training_mode", training_mode, 0L)|] |> Array.choose id)
    static member BatchNormalization(X: Tensor<double>, scale: Tensor<double>, B: Tensor<double>, input_mean: Tensor<double>, input_var: Tensor<double>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        MV() |> fun mv -> execNodeTuple3<double, double, double> "BatchNormalization" [|mv.c(X); mv.c(scale); mv.c(B); mv.c(input_mean); mv.c(input_var)|] ([|Attr.float("epsilon", epsilon, 9.999999747378752e-06f); Attr.float("momentum", momentum, 0.8999999761581421f); Attr.int("training_mode", training_mode, 0L)|] |> Array.choose id)
