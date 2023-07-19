module FSharp.ML.Onnx.API.SnakeCase

open System
open System.Numerics
open Microsoft.ML.OnnxRuntime.Tensors
type on = FSharp.ML.Onnx.API.PascalCase.Onnx

[<ReflectedDefinition>]
type Onnx() =
    static member scaler(X: Tensor<float32>, ?offset: float32[], ?scale: float32[]) =
        on.Scaler(X = X, ?offset = offset, ?scale = scale)
    static member scaler(X: Tensor<double>, ?offset: float32[], ?scale: float32[]) =
        on.Scaler(X = X, ?offset = offset, ?scale = scale)
    static member scaler(X: Tensor<int64>, ?offset: float32[], ?scale: float32[]) =
        on.Scaler(X = X, ?offset = offset, ?scale = scale)
    static member scaler(X: Tensor<int>, ?offset: float32[], ?scale: float32[]) =
        on.Scaler(X = X, ?offset = offset, ?scale = scale)
    static member svm_regressor(X: Tensor<float32>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        on.SVMRegressor(X = X, ?coefficients = coefficients, ?kernel_params = kernel_params, ?kernel_type = kernel_type, ?n_supports = n_supports, ?one_class = one_class, ?post_transform = post_transform, ?rho = rho, ?support_vectors = support_vectors)
    static member svm_regressor(X: Tensor<double>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        on.SVMRegressor(X = X, ?coefficients = coefficients, ?kernel_params = kernel_params, ?kernel_type = kernel_type, ?n_supports = n_supports, ?one_class = one_class, ?post_transform = post_transform, ?rho = rho, ?support_vectors = support_vectors)
    static member svm_regressor(X: Tensor<int64>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        on.SVMRegressor(X = X, ?coefficients = coefficients, ?kernel_params = kernel_params, ?kernel_type = kernel_type, ?n_supports = n_supports, ?one_class = one_class, ?post_transform = post_transform, ?rho = rho, ?support_vectors = support_vectors)
    static member svm_regressor(X: Tensor<int>, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        on.SVMRegressor(X = X, ?coefficients = coefficients, ?kernel_params = kernel_params, ?kernel_type = kernel_type, ?n_supports = n_supports, ?one_class = one_class, ?post_transform = post_transform, ?rho = rho, ?support_vectors = support_vectors)
    static member bitwise_or(A: Tensor<uint8>, B: Tensor<uint8>) =
        on.BitwiseOr(A = A, B = B)
    static member bitwise_or(A: Tensor<uint16>, B: Tensor<uint16>) =
        on.BitwiseOr(A = A, B = B)
    static member bitwise_or(A: Tensor<uint32>, B: Tensor<uint32>) =
        on.BitwiseOr(A = A, B = B)
    static member bitwise_or(A: Tensor<uint64>, B: Tensor<uint64>) =
        on.BitwiseOr(A = A, B = B)
    static member bitwise_or(A: Tensor<int8>, B: Tensor<int8>) =
        on.BitwiseOr(A = A, B = B)
    static member bitwise_or(A: Tensor<int16>, B: Tensor<int16>) =
        on.BitwiseOr(A = A, B = B)
    static member bitwise_or(A: Tensor<int>, B: Tensor<int>) =
        on.BitwiseOr(A = A, B = B)
    static member bitwise_or(A: Tensor<int64>, B: Tensor<int64>) =
        on.BitwiseOr(A = A, B = B)
    static member bitwise_and(A: Tensor<uint8>, B: Tensor<uint8>) =
        on.BitwiseAnd(A = A, B = B)
    static member bitwise_and(A: Tensor<uint16>, B: Tensor<uint16>) =
        on.BitwiseAnd(A = A, B = B)
    static member bitwise_and(A: Tensor<uint32>, B: Tensor<uint32>) =
        on.BitwiseAnd(A = A, B = B)
    static member bitwise_and(A: Tensor<uint64>, B: Tensor<uint64>) =
        on.BitwiseAnd(A = A, B = B)
    static member bitwise_and(A: Tensor<int8>, B: Tensor<int8>) =
        on.BitwiseAnd(A = A, B = B)
    static member bitwise_and(A: Tensor<int16>, B: Tensor<int16>) =
        on.BitwiseAnd(A = A, B = B)
    static member bitwise_and(A: Tensor<int>, B: Tensor<int>) =
        on.BitwiseAnd(A = A, B = B)
    static member bitwise_and(A: Tensor<int64>, B: Tensor<int64>) =
        on.BitwiseAnd(A = A, B = B)
    static member mish(X: Tensor<float32>) =
        on.Mish(X = X)
    static member mish(X: Tensor<double>) =
        on.Mish(X = X)
    static member celu(X: Tensor<float32>, ?alpha: float32) =
        on.Celu(X = X, ?alpha = alpha)
    static member imputer(X: Tensor<float32>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        on.Imputer(X = X, ?imputed_value_floats = imputed_value_floats, ?imputed_value_int64s = imputed_value_int64s, ?replaced_value_float = replaced_value_float, ?replaced_value_int64 = replaced_value_int64)
    static member imputer(X: Tensor<double>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        on.Imputer(X = X, ?imputed_value_floats = imputed_value_floats, ?imputed_value_int64s = imputed_value_int64s, ?replaced_value_float = replaced_value_float, ?replaced_value_int64 = replaced_value_int64)
    static member imputer(X: Tensor<int64>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        on.Imputer(X = X, ?imputed_value_floats = imputed_value_floats, ?imputed_value_int64s = imputed_value_int64s, ?replaced_value_float = replaced_value_float, ?replaced_value_int64 = replaced_value_int64)
    static member imputer(X: Tensor<int>, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        on.Imputer(X = X, ?imputed_value_floats = imputed_value_floats, ?imputed_value_int64s = imputed_value_int64s, ?replaced_value_float = replaced_value_float, ?replaced_value_int64 = replaced_value_int64)
    static member gathernd(data: Tensor<uint8>, indices: Tensor<int64>, ?batch_dims: int64) =
        on.GatherND(data = data, indices = indices, ?batch_dims = batch_dims)
    static member gathernd(data: Tensor<uint16>, indices: Tensor<int64>, ?batch_dims: int64) =
        on.GatherND(data = data, indices = indices, ?batch_dims = batch_dims)
    static member gathernd(data: Tensor<uint32>, indices: Tensor<int64>, ?batch_dims: int64) =
        on.GatherND(data = data, indices = indices, ?batch_dims = batch_dims)
    static member gathernd(data: Tensor<uint64>, indices: Tensor<int64>, ?batch_dims: int64) =
        on.GatherND(data = data, indices = indices, ?batch_dims = batch_dims)
    static member gathernd(data: Tensor<int8>, indices: Tensor<int64>, ?batch_dims: int64) =
        on.GatherND(data = data, indices = indices, ?batch_dims = batch_dims)
    static member gathernd(data: Tensor<int16>, indices: Tensor<int64>, ?batch_dims: int64) =
        on.GatherND(data = data, indices = indices, ?batch_dims = batch_dims)
    static member gathernd(data: Tensor<int>, indices: Tensor<int64>, ?batch_dims: int64) =
        on.GatherND(data = data, indices = indices, ?batch_dims = batch_dims)
    static member gathernd(data: Tensor<int64>, indices: Tensor<int64>, ?batch_dims: int64) =
        on.GatherND(data = data, indices = indices, ?batch_dims = batch_dims)
    static member gathernd(data: Tensor<float32>, indices: Tensor<int64>, ?batch_dims: int64) =
        on.GatherND(data = data, indices = indices, ?batch_dims = batch_dims)
    static member gathernd(data: Tensor<double>, indices: Tensor<int64>, ?batch_dims: int64) =
        on.GatherND(data = data, indices = indices, ?batch_dims = batch_dims)
    static member gathernd(data: Tensor<string>, indices: Tensor<int64>, ?batch_dims: int64) =
        on.GatherND(data = data, indices = indices, ?batch_dims = batch_dims)
    static member gathernd(data: Tensor<bool>, indices: Tensor<int64>, ?batch_dims: int64) =
        on.GatherND(data = data, indices = indices, ?batch_dims = batch_dims)
    static member gathernd(data: Tensor<Complex>, indices: Tensor<int64>, ?batch_dims: int64) =
        on.GatherND(data = data, indices = indices, ?batch_dims = batch_dims)
    static member scatternd(data: Tensor<uint8>, indices: Tensor<int64>, updates: Tensor<uint8>, ?reduction: string) =
        on.ScatterND(data = data, indices = indices, updates = updates, ?reduction = reduction)
    static member scatternd(data: Tensor<uint16>, indices: Tensor<int64>, updates: Tensor<uint16>, ?reduction: string) =
        on.ScatterND(data = data, indices = indices, updates = updates, ?reduction = reduction)
    static member scatternd(data: Tensor<uint32>, indices: Tensor<int64>, updates: Tensor<uint32>, ?reduction: string) =
        on.ScatterND(data = data, indices = indices, updates = updates, ?reduction = reduction)
    static member scatternd(data: Tensor<uint64>, indices: Tensor<int64>, updates: Tensor<uint64>, ?reduction: string) =
        on.ScatterND(data = data, indices = indices, updates = updates, ?reduction = reduction)
    static member scatternd(data: Tensor<int8>, indices: Tensor<int64>, updates: Tensor<int8>, ?reduction: string) =
        on.ScatterND(data = data, indices = indices, updates = updates, ?reduction = reduction)
    static member scatternd(data: Tensor<int16>, indices: Tensor<int64>, updates: Tensor<int16>, ?reduction: string) =
        on.ScatterND(data = data, indices = indices, updates = updates, ?reduction = reduction)
    static member scatternd(data: Tensor<int>, indices: Tensor<int64>, updates: Tensor<int>, ?reduction: string) =
        on.ScatterND(data = data, indices = indices, updates = updates, ?reduction = reduction)
    static member scatternd(data: Tensor<int64>, indices: Tensor<int64>, updates: Tensor<int64>, ?reduction: string) =
        on.ScatterND(data = data, indices = indices, updates = updates, ?reduction = reduction)
    static member scatternd(data: Tensor<float32>, indices: Tensor<int64>, updates: Tensor<float32>, ?reduction: string) =
        on.ScatterND(data = data, indices = indices, updates = updates, ?reduction = reduction)
    static member scatternd(data: Tensor<double>, indices: Tensor<int64>, updates: Tensor<double>, ?reduction: string) =
        on.ScatterND(data = data, indices = indices, updates = updates, ?reduction = reduction)
    static member scatternd(data: Tensor<string>, indices: Tensor<int64>, updates: Tensor<string>, ?reduction: string) =
        on.ScatterND(data = data, indices = indices, updates = updates, ?reduction = reduction)
    static member scatternd(data: Tensor<bool>, indices: Tensor<int64>, updates: Tensor<bool>, ?reduction: string) =
        on.ScatterND(data = data, indices = indices, updates = updates, ?reduction = reduction)
    static member scatternd(data: Tensor<Complex>, indices: Tensor<int64>, updates: Tensor<Complex>, ?reduction: string) =
        on.ScatterND(data = data, indices = indices, updates = updates, ?reduction = reduction)
    static member det(X: Tensor<float32>) =
        on.Det(X = X)
    static member det(X: Tensor<double>) =
        on.Det(X = X)
    static member normalizer(X: Tensor<float32>, ?norm: string) =
        on.Normalizer(X = X, ?norm = norm)
    static member normalizer(X: Tensor<double>, ?norm: string) =
        on.Normalizer(X = X, ?norm = norm)
    static member normalizer(X: Tensor<int64>, ?norm: string) =
        on.Normalizer(X = X, ?norm = norm)
    static member normalizer(X: Tensor<int>, ?norm: string) =
        on.Normalizer(X = X, ?norm = norm)
    static member feature_vectorizer([<ParamArray>]X: Tensor<int>[], ?inputdimensions: int64[]) =
        on.FeatureVectorizer(X = X, ?inputdimensions = inputdimensions)
    static member feature_vectorizer([<ParamArray>]X: Tensor<int64>[], ?inputdimensions: int64[]) =
        on.FeatureVectorizer(X = X, ?inputdimensions = inputdimensions)
    static member feature_vectorizer([<ParamArray>]X: Tensor<float32>[], ?inputdimensions: int64[]) =
        on.FeatureVectorizer(X = X, ?inputdimensions = inputdimensions)
    static member feature_vectorizer([<ParamArray>]X: Tensor<double>[], ?inputdimensions: int64[]) =
        on.FeatureVectorizer(X = X, ?inputdimensions = inputdimensions)
    static member mul(A: Tensor<uint8>, B: Tensor<uint8>) =
        on.Mul(A = A, B = B)
    static member mul(A: Tensor<uint16>, B: Tensor<uint16>) =
        on.Mul(A = A, B = B)
    static member mul(A: Tensor<uint32>, B: Tensor<uint32>) =
        on.Mul(A = A, B = B)
    static member mul(A: Tensor<uint64>, B: Tensor<uint64>) =
        on.Mul(A = A, B = B)
    static member mul(A: Tensor<int8>, B: Tensor<int8>) =
        on.Mul(A = A, B = B)
    static member mul(A: Tensor<int16>, B: Tensor<int16>) =
        on.Mul(A = A, B = B)
    static member mul(A: Tensor<int>, B: Tensor<int>) =
        on.Mul(A = A, B = B)
    static member mul(A: Tensor<int64>, B: Tensor<int64>) =
        on.Mul(A = A, B = B)
    static member mul(A: Tensor<float32>, B: Tensor<float32>) =
        on.Mul(A = A, B = B)
    static member mul(A: Tensor<double>, B: Tensor<double>) =
        on.Mul(A = A, B = B)
    static member max([<ParamArray>]data_0: Tensor<uint8>[]) =
        on.Max(data_0 = data_0)
    static member max([<ParamArray>]data_0: Tensor<uint16>[]) =
        on.Max(data_0 = data_0)
    static member max([<ParamArray>]data_0: Tensor<uint32>[]) =
        on.Max(data_0 = data_0)
    static member max([<ParamArray>]data_0: Tensor<uint64>[]) =
        on.Max(data_0 = data_0)
    static member max([<ParamArray>]data_0: Tensor<int8>[]) =
        on.Max(data_0 = data_0)
    static member max([<ParamArray>]data_0: Tensor<int16>[]) =
        on.Max(data_0 = data_0)
    static member max([<ParamArray>]data_0: Tensor<int>[]) =
        on.Max(data_0 = data_0)
    static member max([<ParamArray>]data_0: Tensor<int64>[]) =
        on.Max(data_0 = data_0)
    static member max([<ParamArray>]data_0: Tensor<float32>[]) =
        on.Max(data_0 = data_0)
    static member max([<ParamArray>]data_0: Tensor<double>[]) =
        on.Max(data_0 = data_0)
    static member group_normalization(X: Tensor<float32>, scale: Tensor<float32>, bias: Tensor<float32>, num_groups: int64, ?epsilon: float32) =
        on.GroupNormalization(X = X, scale = scale, bias = bias, num_groups = num_groups, ?epsilon = epsilon)
    static member group_normalization(X: Tensor<double>, scale: Tensor<double>, bias: Tensor<double>, num_groups: int64, ?epsilon: float32) =
        on.GroupNormalization(X = X, scale = scale, bias = bias, num_groups = num_groups, ?epsilon = epsilon)
    static member mod_(A: Tensor<uint8>, B: Tensor<uint8>, ?fmod: int64) =
        on.Mod(A = A, B = B, ?fmod = fmod)
    static member mod_(A: Tensor<uint16>, B: Tensor<uint16>, ?fmod: int64) =
        on.Mod(A = A, B = B, ?fmod = fmod)
    static member mod_(A: Tensor<uint32>, B: Tensor<uint32>, ?fmod: int64) =
        on.Mod(A = A, B = B, ?fmod = fmod)
    static member mod_(A: Tensor<uint64>, B: Tensor<uint64>, ?fmod: int64) =
        on.Mod(A = A, B = B, ?fmod = fmod)
    static member mod_(A: Tensor<int8>, B: Tensor<int8>, ?fmod: int64) =
        on.Mod(A = A, B = B, ?fmod = fmod)
    static member mod_(A: Tensor<int16>, B: Tensor<int16>, ?fmod: int64) =
        on.Mod(A = A, B = B, ?fmod = fmod)
    static member mod_(A: Tensor<int>, B: Tensor<int>, ?fmod: int64) =
        on.Mod(A = A, B = B, ?fmod = fmod)
    static member mod_(A: Tensor<int64>, B: Tensor<int64>, ?fmod: int64) =
        on.Mod(A = A, B = B, ?fmod = fmod)
    static member mod_(A: Tensor<float32>, B: Tensor<float32>, ?fmod: int64) =
        on.Mod(A = A, B = B, ?fmod = fmod)
    static member mod_(A: Tensor<double>, B: Tensor<double>, ?fmod: int64) =
        on.Mod(A = A, B = B, ?fmod = fmod)
    static member log(input: Tensor<float32>) =
        on.Log(input = input)
    static member log(input: Tensor<double>) =
        on.Log(input = input)
    static member arg_max(data: Tensor<uint8>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMax(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_max(data: Tensor<uint16>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMax(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_max(data: Tensor<uint32>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMax(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_max(data: Tensor<uint64>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMax(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_max(data: Tensor<int8>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMax(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_max(data: Tensor<int16>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMax(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_max(data: Tensor<int>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMax(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_max(data: Tensor<int64>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMax(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_max(data: Tensor<float32>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMax(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_max(data: Tensor<double>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMax(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member reduce_max(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMax(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_max(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMax(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_max(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMax(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_max(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMax(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_max(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMax(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_max(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMax(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_max(data: Tensor<uint8>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMax(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_max(data: Tensor<int8>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMax(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_min(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMin(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_min(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMin(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_min(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMin(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_min(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMin(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_min(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMin(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_min(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMin(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_min(data: Tensor<uint8>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMin(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_min(data: Tensor<int8>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMin(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member deform_conv(X: Tensor<float32>, W: Tensor<float32>, offset: Tensor<float32>, ?B: Tensor<float32>, ?mask: Tensor<float32>, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?offset_group: int64, ?pads: int64[], ?strides: int64[]) =
        on.DeformConv(X = X, W = W, offset = offset, ?B = B, ?mask = mask, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?offset_group = offset_group, ?pads = pads, ?strides = strides)
    static member deform_conv(X: Tensor<double>, W: Tensor<double>, offset: Tensor<double>, ?B: Tensor<double>, ?mask: Tensor<double>, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?offset_group: int64, ?pads: int64[], ?strides: int64[]) =
        on.DeformConv(X = X, W = W, offset = offset, ?B = B, ?mask = mask, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?offset_group = offset_group, ?pads = pads, ?strides = strides)
    static member sign(input: Tensor<uint8>) =
        on.Sign(input = input)
    static member sign(input: Tensor<uint16>) =
        on.Sign(input = input)
    static member sign(input: Tensor<uint32>) =
        on.Sign(input = input)
    static member sign(input: Tensor<uint64>) =
        on.Sign(input = input)
    static member sign(input: Tensor<int8>) =
        on.Sign(input = input)
    static member sign(input: Tensor<int16>) =
        on.Sign(input = input)
    static member sign(input: Tensor<int>) =
        on.Sign(input = input)
    static member sign(input: Tensor<int64>) =
        on.Sign(input = input)
    static member sign(input: Tensor<float32>) =
        on.Sign(input = input)
    static member sign(input: Tensor<double>) =
        on.Sign(input = input)
    static member min([<ParamArray>]data_0: Tensor<uint8>[]) =
        on.Min(data_0 = data_0)
    static member min([<ParamArray>]data_0: Tensor<uint16>[]) =
        on.Min(data_0 = data_0)
    static member min([<ParamArray>]data_0: Tensor<uint32>[]) =
        on.Min(data_0 = data_0)
    static member min([<ParamArray>]data_0: Tensor<uint64>[]) =
        on.Min(data_0 = data_0)
    static member min([<ParamArray>]data_0: Tensor<int8>[]) =
        on.Min(data_0 = data_0)
    static member min([<ParamArray>]data_0: Tensor<int16>[]) =
        on.Min(data_0 = data_0)
    static member min([<ParamArray>]data_0: Tensor<int>[]) =
        on.Min(data_0 = data_0)
    static member min([<ParamArray>]data_0: Tensor<int64>[]) =
        on.Min(data_0 = data_0)
    static member min([<ParamArray>]data_0: Tensor<float32>[]) =
        on.Min(data_0 = data_0)
    static member min([<ParamArray>]data_0: Tensor<double>[]) =
        on.Min(data_0 = data_0)
    static member range(start: Tensor<float32>, limit: Tensor<float32>, delta: Tensor<float32>) =
        on.Range(start = start, limit = limit, delta = delta)
    static member range(start: Tensor<double>, limit: Tensor<double>, delta: Tensor<double>) =
        on.Range(start = start, limit = limit, delta = delta)
    static member range(start: Tensor<int16>, limit: Tensor<int16>, delta: Tensor<int16>) =
        on.Range(start = start, limit = limit, delta = delta)
    static member range(start: Tensor<int>, limit: Tensor<int>, delta: Tensor<int>) =
        on.Range(start = start, limit = limit, delta = delta)
    static member range(start: Tensor<int64>, limit: Tensor<int64>, delta: Tensor<int64>) =
        on.Range(start = start, limit = limit, delta = delta)
    static member p_relu(X: Tensor<float32>, slope: Tensor<float32>) =
        on.PRelu(X = X, slope = slope)
    static member p_relu(X: Tensor<double>, slope: Tensor<double>) =
        on.PRelu(X = X, slope = slope)
    static member p_relu(X: Tensor<uint32>, slope: Tensor<uint32>) =
        on.PRelu(X = X, slope = slope)
    static member p_relu(X: Tensor<uint64>, slope: Tensor<uint64>) =
        on.PRelu(X = X, slope = slope)
    static member p_relu(X: Tensor<int>, slope: Tensor<int>) =
        on.PRelu(X = X, slope = slope)
    static member p_relu(X: Tensor<int64>, slope: Tensor<int64>) =
        on.PRelu(X = X, slope = slope)
    static member non_zero(X: Tensor<uint8>) =
        on.NonZero(X = X)
    static member non_zero(X: Tensor<uint16>) =
        on.NonZero(X = X)
    static member non_zero(X: Tensor<uint32>) =
        on.NonZero(X = X)
    static member non_zero(X: Tensor<uint64>) =
        on.NonZero(X = X)
    static member non_zero(X: Tensor<int8>) =
        on.NonZero(X = X)
    static member non_zero(X: Tensor<int16>) =
        on.NonZero(X = X)
    static member non_zero(X: Tensor<int>) =
        on.NonZero(X = X)
    static member non_zero(X: Tensor<int64>) =
        on.NonZero(X = X)
    static member non_zero(X: Tensor<float32>) =
        on.NonZero(X = X)
    static member non_zero(X: Tensor<double>) =
        on.NonZero(X = X)
    static member non_zero(X: Tensor<string>) =
        on.NonZero(X = X)
    static member non_zero(X: Tensor<bool>) =
        on.NonZero(X = X)
    static member non_zero(X: Tensor<Complex>) =
        on.NonZero(X = X)
    static member ceil(X: Tensor<float32>) =
        on.Ceil(X = X)
    static member ceil(X: Tensor<double>) =
        on.Ceil(X = X)
    static member tan(input: Tensor<float32>) =
        on.Tan(input = input)
    static member tan(input: Tensor<double>) =
        on.Tan(input = input)
    static member not_(X: Tensor<bool>) =
        on.Not(X = X)
    static member clip(input: Tensor<uint8>, ?min: Tensor<uint8>, ?max: Tensor<uint8>) =
        on.Clip(input = input, ?min = min, ?max = max)
    static member clip(input: Tensor<uint16>, ?min: Tensor<uint16>, ?max: Tensor<uint16>) =
        on.Clip(input = input, ?min = min, ?max = max)
    static member clip(input: Tensor<uint32>, ?min: Tensor<uint32>, ?max: Tensor<uint32>) =
        on.Clip(input = input, ?min = min, ?max = max)
    static member clip(input: Tensor<uint64>, ?min: Tensor<uint64>, ?max: Tensor<uint64>) =
        on.Clip(input = input, ?min = min, ?max = max)
    static member clip(input: Tensor<int8>, ?min: Tensor<int8>, ?max: Tensor<int8>) =
        on.Clip(input = input, ?min = min, ?max = max)
    static member clip(input: Tensor<int16>, ?min: Tensor<int16>, ?max: Tensor<int16>) =
        on.Clip(input = input, ?min = min, ?max = max)
    static member clip(input: Tensor<int>, ?min: Tensor<int>, ?max: Tensor<int>) =
        on.Clip(input = input, ?min = min, ?max = max)
    static member clip(input: Tensor<int64>, ?min: Tensor<int64>, ?max: Tensor<int64>) =
        on.Clip(input = input, ?min = min, ?max = max)
    static member clip(input: Tensor<float32>, ?min: Tensor<float32>, ?max: Tensor<float32>) =
        on.Clip(input = input, ?min = min, ?max = max)
    static member clip(input: Tensor<double>, ?min: Tensor<double>, ?max: Tensor<double>) =
        on.Clip(input = input, ?min = min, ?max = max)
    static member reducel2(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceL2(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reducel2(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceL2(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reducel2(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceL2(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reducel2(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceL2(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reducel2(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceL2(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reducel2(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceL2(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member neg(X: Tensor<float32>) =
        on.Neg(X = X)
    static member neg(X: Tensor<int>) =
        on.Neg(X = X)
    static member neg(X: Tensor<int8>) =
        on.Neg(X = X)
    static member neg(X: Tensor<int16>) =
        on.Neg(X = X)
    static member neg(X: Tensor<int64>) =
        on.Neg(X = X)
    static member neg(X: Tensor<double>) =
        on.Neg(X = X)
    static member linear_regressor(X: Tensor<float32>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        on.LinearRegressor(X = X, ?coefficients = coefficients, ?intercepts = intercepts, ?post_transform = post_transform, ?targets = targets)
    static member linear_regressor(X: Tensor<double>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        on.LinearRegressor(X = X, ?coefficients = coefficients, ?intercepts = intercepts, ?post_transform = post_transform, ?targets = targets)
    static member linear_regressor(X: Tensor<int64>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        on.LinearRegressor(X = X, ?coefficients = coefficients, ?intercepts = intercepts, ?post_transform = post_transform, ?targets = targets)
    static member linear_regressor(X: Tensor<int>, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        on.LinearRegressor(X = X, ?coefficients = coefficients, ?intercepts = intercepts, ?post_transform = post_transform, ?targets = targets)
    static member bitwise_xor(A: Tensor<uint8>, B: Tensor<uint8>) =
        on.BitwiseXor(A = A, B = B)
    static member bitwise_xor(A: Tensor<uint16>, B: Tensor<uint16>) =
        on.BitwiseXor(A = A, B = B)
    static member bitwise_xor(A: Tensor<uint32>, B: Tensor<uint32>) =
        on.BitwiseXor(A = A, B = B)
    static member bitwise_xor(A: Tensor<uint64>, B: Tensor<uint64>) =
        on.BitwiseXor(A = A, B = B)
    static member bitwise_xor(A: Tensor<int8>, B: Tensor<int8>) =
        on.BitwiseXor(A = A, B = B)
    static member bitwise_xor(A: Tensor<int16>, B: Tensor<int16>) =
        on.BitwiseXor(A = A, B = B)
    static member bitwise_xor(A: Tensor<int>, B: Tensor<int>) =
        on.BitwiseXor(A = A, B = B)
    static member bitwise_xor(A: Tensor<int64>, B: Tensor<int64>) =
        on.BitwiseXor(A = A, B = B)
    static member conv(X: Tensor<float32>, W: Tensor<float32>, ?B: Tensor<float32>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Conv(X = X, W = W, ?B = B, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member conv(X: Tensor<double>, W: Tensor<double>, ?B: Tensor<double>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Conv(X = X, W = W, ?B = B, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member abs(X: Tensor<uint8>) =
        on.Abs(X = X)
    static member abs(X: Tensor<uint16>) =
        on.Abs(X = X)
    static member abs(X: Tensor<uint32>) =
        on.Abs(X = X)
    static member abs(X: Tensor<uint64>) =
        on.Abs(X = X)
    static member abs(X: Tensor<int8>) =
        on.Abs(X = X)
    static member abs(X: Tensor<int16>) =
        on.Abs(X = X)
    static member abs(X: Tensor<int>) =
        on.Abs(X = X)
    static member abs(X: Tensor<int64>) =
        on.Abs(X = X)
    static member abs(X: Tensor<float32>) =
        on.Abs(X = X)
    static member abs(X: Tensor<double>) =
        on.Abs(X = X)
    static member softplus(X: Tensor<float32>) =
        on.Softplus(X = X)
    static member softplus(X: Tensor<double>) =
        on.Softplus(X = X)
    static member conv_transpose(X: Tensor<float32>, W: Tensor<float32>, ?B: Tensor<float32>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?output_padding: int64[], ?output_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.ConvTranspose(X = X, W = W, ?B = B, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?output_padding = output_padding, ?output_shape = output_shape, ?pads = pads, ?strides = strides)
    static member conv_transpose(X: Tensor<double>, W: Tensor<double>, ?B: Tensor<double>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?output_padding: int64[], ?output_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.ConvTranspose(X = X, W = W, ?B = B, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?output_padding = output_padding, ?output_shape = output_shape, ?pads = pads, ?strides = strides)
    static member flatten(input: Tensor<uint8>, ?axis: int64) =
        on.Flatten(input = input, ?axis = axis)
    static member flatten(input: Tensor<uint16>, ?axis: int64) =
        on.Flatten(input = input, ?axis = axis)
    static member flatten(input: Tensor<uint32>, ?axis: int64) =
        on.Flatten(input = input, ?axis = axis)
    static member flatten(input: Tensor<uint64>, ?axis: int64) =
        on.Flatten(input = input, ?axis = axis)
    static member flatten(input: Tensor<int8>, ?axis: int64) =
        on.Flatten(input = input, ?axis = axis)
    static member flatten(input: Tensor<int16>, ?axis: int64) =
        on.Flatten(input = input, ?axis = axis)
    static member flatten(input: Tensor<int>, ?axis: int64) =
        on.Flatten(input = input, ?axis = axis)
    static member flatten(input: Tensor<int64>, ?axis: int64) =
        on.Flatten(input = input, ?axis = axis)
    static member flatten(input: Tensor<float32>, ?axis: int64) =
        on.Flatten(input = input, ?axis = axis)
    static member flatten(input: Tensor<double>, ?axis: int64) =
        on.Flatten(input = input, ?axis = axis)
    static member flatten(input: Tensor<string>, ?axis: int64) =
        on.Flatten(input = input, ?axis = axis)
    static member flatten(input: Tensor<bool>, ?axis: int64) =
        on.Flatten(input = input, ?axis = axis)
    static member flatten(input: Tensor<Complex>, ?axis: int64) =
        on.Flatten(input = input, ?axis = axis)
    static member reduce_log_sum(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceLogSum(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_log_sum(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceLogSum(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_log_sum(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceLogSum(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_log_sum(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceLogSum(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_log_sum(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceLogSum(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_log_sum(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceLogSum(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member einsum(equation: string, [<ParamArray>]Inputs: Tensor<uint8>[]) =
        on.Einsum(equation = equation, Inputs = Inputs)
    static member einsum(equation: string, [<ParamArray>]Inputs: Tensor<uint16>[]) =
        on.Einsum(equation = equation, Inputs = Inputs)
    static member einsum(equation: string, [<ParamArray>]Inputs: Tensor<uint32>[]) =
        on.Einsum(equation = equation, Inputs = Inputs)
    static member einsum(equation: string, [<ParamArray>]Inputs: Tensor<uint64>[]) =
        on.Einsum(equation = equation, Inputs = Inputs)
    static member einsum(equation: string, [<ParamArray>]Inputs: Tensor<int8>[]) =
        on.Einsum(equation = equation, Inputs = Inputs)
    static member einsum(equation: string, [<ParamArray>]Inputs: Tensor<int16>[]) =
        on.Einsum(equation = equation, Inputs = Inputs)
    static member einsum(equation: string, [<ParamArray>]Inputs: Tensor<int>[]) =
        on.Einsum(equation = equation, Inputs = Inputs)
    static member einsum(equation: string, [<ParamArray>]Inputs: Tensor<int64>[]) =
        on.Einsum(equation = equation, Inputs = Inputs)
    static member einsum(equation: string, [<ParamArray>]Inputs: Tensor<float32>[]) =
        on.Einsum(equation = equation, Inputs = Inputs)
    static member einsum(equation: string, [<ParamArray>]Inputs: Tensor<double>[]) =
        on.Einsum(equation = equation, Inputs = Inputs)
    static member reduce_log_sum_exp(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceLogSumExp(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_log_sum_exp(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceLogSumExp(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_log_sum_exp(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceLogSumExp(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_log_sum_exp(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceLogSumExp(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_log_sum_exp(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceLogSumExp(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_log_sum_exp(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceLogSumExp(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member sub(A: Tensor<uint8>, B: Tensor<uint8>) =
        on.Sub(A = A, B = B)
    static member sub(A: Tensor<uint16>, B: Tensor<uint16>) =
        on.Sub(A = A, B = B)
    static member sub(A: Tensor<uint32>, B: Tensor<uint32>) =
        on.Sub(A = A, B = B)
    static member sub(A: Tensor<uint64>, B: Tensor<uint64>) =
        on.Sub(A = A, B = B)
    static member sub(A: Tensor<int8>, B: Tensor<int8>) =
        on.Sub(A = A, B = B)
    static member sub(A: Tensor<int16>, B: Tensor<int16>) =
        on.Sub(A = A, B = B)
    static member sub(A: Tensor<int>, B: Tensor<int>) =
        on.Sub(A = A, B = B)
    static member sub(A: Tensor<int64>, B: Tensor<int64>) =
        on.Sub(A = A, B = B)
    static member sub(A: Tensor<float32>, B: Tensor<float32>) =
        on.Sub(A = A, B = B)
    static member sub(A: Tensor<double>, B: Tensor<double>) =
        on.Sub(A = A, B = B)
    static member floor(X: Tensor<float32>) =
        on.Floor(X = X)
    static member floor(X: Tensor<double>) =
        on.Floor(X = X)
    static member max_roi_pool(X: Tensor<float32>, rois: Tensor<float32>, pooled_shape: int64[], ?spatial_scale: float32) =
        on.MaxRoiPool(X = X, rois = rois, pooled_shape = pooled_shape, ?spatial_scale = spatial_scale)
    static member max_roi_pool(X: Tensor<double>, rois: Tensor<double>, pooled_shape: int64[], ?spatial_scale: float32) =
        on.MaxRoiPool(X = X, rois = rois, pooled_shape = pooled_shape, ?spatial_scale = spatial_scale)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<uint8>[]) =
        on.Concat(axis = axis, inputs = inputs)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<uint16>[]) =
        on.Concat(axis = axis, inputs = inputs)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<uint32>[]) =
        on.Concat(axis = axis, inputs = inputs)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<uint64>[]) =
        on.Concat(axis = axis, inputs = inputs)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<int8>[]) =
        on.Concat(axis = axis, inputs = inputs)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<int16>[]) =
        on.Concat(axis = axis, inputs = inputs)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<int>[]) =
        on.Concat(axis = axis, inputs = inputs)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<int64>[]) =
        on.Concat(axis = axis, inputs = inputs)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<float32>[]) =
        on.Concat(axis = axis, inputs = inputs)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<double>[]) =
        on.Concat(axis = axis, inputs = inputs)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<string>[]) =
        on.Concat(axis = axis, inputs = inputs)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<bool>[]) =
        on.Concat(axis = axis, inputs = inputs)
    static member concat(axis: int64, [<ParamArray>]inputs: Tensor<Complex>[]) =
        on.Concat(axis = axis, inputs = inputs)
    static member sigmoid(X: Tensor<float32>) =
        on.Sigmoid(X = X)
    static member sigmoid(X: Tensor<double>) =
        on.Sigmoid(X = X)
    static member softmax(input: Tensor<float32>, ?axis: int64) =
        on.Softmax(input = input, ?axis = axis)
    static member softmax(input: Tensor<double>, ?axis: int64) =
        on.Softmax(input = input, ?axis = axis)
    static member add(A: Tensor<uint8>, B: Tensor<uint8>) =
        on.Add(A = A, B = B)
    static member add(A: Tensor<uint16>, B: Tensor<uint16>) =
        on.Add(A = A, B = B)
    static member add(A: Tensor<uint32>, B: Tensor<uint32>) =
        on.Add(A = A, B = B)
    static member add(A: Tensor<uint64>, B: Tensor<uint64>) =
        on.Add(A = A, B = B)
    static member add(A: Tensor<int8>, B: Tensor<int8>) =
        on.Add(A = A, B = B)
    static member add(A: Tensor<int16>, B: Tensor<int16>) =
        on.Add(A = A, B = B)
    static member add(A: Tensor<int>, B: Tensor<int>) =
        on.Add(A = A, B = B)
    static member add(A: Tensor<int64>, B: Tensor<int64>) =
        on.Add(A = A, B = B)
    static member add(A: Tensor<float32>, B: Tensor<float32>) =
        on.Add(A = A, B = B)
    static member add(A: Tensor<double>, B: Tensor<double>) =
        on.Add(A = A, B = B)
    static member instance_normalization(input: Tensor<float32>, scale: Tensor<float32>, B: Tensor<float32>, ?epsilon: float32) =
        on.InstanceNormalization(input = input, scale = scale, B = B, ?epsilon = epsilon)
    static member instance_normalization(input: Tensor<double>, scale: Tensor<double>, B: Tensor<double>, ?epsilon: float32) =
        on.InstanceNormalization(input = input, scale = scale, B = B, ?epsilon = epsilon)
    static member lp_pool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?p: int64, ?pads: int64[], ?strides: int64[]) =
        on.LpPool(X = X, kernel_shape = kernel_shape, ?auto_pad = auto_pad, ?dilations = dilations, ?p = p, ?pads = pads, ?strides = strides)
    static member lp_pool(X: Tensor<double>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?p: int64, ?pads: int64[], ?strides: int64[]) =
        on.LpPool(X = X, kernel_shape = kernel_shape, ?auto_pad = auto_pad, ?dilations = dilations, ?p = p, ?pads = pads, ?strides = strides)
    static member arg_min(data: Tensor<uint8>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMin(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_min(data: Tensor<uint16>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMin(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_min(data: Tensor<uint32>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMin(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_min(data: Tensor<uint64>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMin(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_min(data: Tensor<int8>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMin(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_min(data: Tensor<int16>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMin(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_min(data: Tensor<int>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMin(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_min(data: Tensor<int64>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMin(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_min(data: Tensor<float32>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMin(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member arg_min(data: Tensor<double>, ?axis: int64, ?keepdims: int64, ?select_last_index: int64) =
        on.ArgMin(data = data, ?axis = axis, ?keepdims = keepdims, ?select_last_index = select_last_index)
    static member round(X: Tensor<float32>) =
        on.Round(X = X)
    static member round(X: Tensor<double>) =
        on.Round(X = X)
    static member bit_shift(X: Tensor<uint8>, Y: Tensor<uint8>, direction: string) =
        on.BitShift(X = X, Y = Y, direction = direction)
    static member bit_shift(X: Tensor<uint16>, Y: Tensor<uint16>, direction: string) =
        on.BitShift(X = X, Y = Y, direction = direction)
    static member bit_shift(X: Tensor<uint32>, Y: Tensor<uint32>, direction: string) =
        on.BitShift(X = X, Y = Y, direction = direction)
    static member bit_shift(X: Tensor<uint64>, Y: Tensor<uint64>, direction: string) =
        on.BitShift(X = X, Y = Y, direction = direction)
    static member average_pool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?count_include_pad: int64, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.AveragePool(X = X, kernel_shape = kernel_shape, ?auto_pad = auto_pad, ?count_include_pad = count_include_pad, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member average_pool(X: Tensor<double>, kernel_shape: int64[], ?auto_pad: string, ?count_include_pad: int64, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.AveragePool(X = X, kernel_shape = kernel_shape, ?auto_pad = auto_pad, ?count_include_pad = count_include_pad, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member exp(input: Tensor<float32>) =
        on.Exp(input = input)
    static member exp(input: Tensor<double>) =
        on.Exp(input = input)
    static member array_feature_extractor(X: Tensor<float32>, Y: Tensor<int64>) =
        on.ArrayFeatureExtractor(X = X, Y = Y)
    static member array_feature_extractor(X: Tensor<double>, Y: Tensor<int64>) =
        on.ArrayFeatureExtractor(X = X, Y = Y)
    static member array_feature_extractor(X: Tensor<int64>, Y: Tensor<int64>) =
        on.ArrayFeatureExtractor(X = X, Y = Y)
    static member array_feature_extractor(X: Tensor<int>, Y: Tensor<int64>) =
        on.ArrayFeatureExtractor(X = X, Y = Y)
    static member array_feature_extractor(X: Tensor<string>, Y: Tensor<int64>) =
        on.ArrayFeatureExtractor(X = X, Y = Y)
    static member mat_mul(A: Tensor<float32>, B: Tensor<float32>) =
        on.MatMul(A = A, B = B)
    static member mat_mul(A: Tensor<double>, B: Tensor<double>) =
        on.MatMul(A = A, B = B)
    static member mat_mul(A: Tensor<uint32>, B: Tensor<uint32>) =
        on.MatMul(A = A, B = B)
    static member mat_mul(A: Tensor<uint64>, B: Tensor<uint64>) =
        on.MatMul(A = A, B = B)
    static member mat_mul(A: Tensor<int>, B: Tensor<int>) =
        on.MatMul(A = A, B = B)
    static member mat_mul(A: Tensor<int64>, B: Tensor<int64>) =
        on.MatMul(A = A, B = B)
    static member leaky_relu(X: Tensor<float32>, ?alpha: float32) =
        on.LeakyRelu(X = X, ?alpha = alpha)
    static member leaky_relu(X: Tensor<double>, ?alpha: float32) =
        on.LeakyRelu(X = X, ?alpha = alpha)
    static member reduce_mean(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMean(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_mean(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMean(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_mean(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMean(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_mean(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMean(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_mean(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMean(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_mean(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceMean(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reverse_sequence(input: Tensor<uint8>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        on.ReverseSequence(input = input, sequence_lens = sequence_lens, ?batch_axis = batch_axis, ?time_axis = time_axis)
    static member reverse_sequence(input: Tensor<uint16>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        on.ReverseSequence(input = input, sequence_lens = sequence_lens, ?batch_axis = batch_axis, ?time_axis = time_axis)
    static member reverse_sequence(input: Tensor<uint32>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        on.ReverseSequence(input = input, sequence_lens = sequence_lens, ?batch_axis = batch_axis, ?time_axis = time_axis)
    static member reverse_sequence(input: Tensor<uint64>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        on.ReverseSequence(input = input, sequence_lens = sequence_lens, ?batch_axis = batch_axis, ?time_axis = time_axis)
    static member reverse_sequence(input: Tensor<int8>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        on.ReverseSequence(input = input, sequence_lens = sequence_lens, ?batch_axis = batch_axis, ?time_axis = time_axis)
    static member reverse_sequence(input: Tensor<int16>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        on.ReverseSequence(input = input, sequence_lens = sequence_lens, ?batch_axis = batch_axis, ?time_axis = time_axis)
    static member reverse_sequence(input: Tensor<int>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        on.ReverseSequence(input = input, sequence_lens = sequence_lens, ?batch_axis = batch_axis, ?time_axis = time_axis)
    static member reverse_sequence(input: Tensor<int64>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        on.ReverseSequence(input = input, sequence_lens = sequence_lens, ?batch_axis = batch_axis, ?time_axis = time_axis)
    static member reverse_sequence(input: Tensor<float32>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        on.ReverseSequence(input = input, sequence_lens = sequence_lens, ?batch_axis = batch_axis, ?time_axis = time_axis)
    static member reverse_sequence(input: Tensor<double>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        on.ReverseSequence(input = input, sequence_lens = sequence_lens, ?batch_axis = batch_axis, ?time_axis = time_axis)
    static member reverse_sequence(input: Tensor<string>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        on.ReverseSequence(input = input, sequence_lens = sequence_lens, ?batch_axis = batch_axis, ?time_axis = time_axis)
    static member reverse_sequence(input: Tensor<bool>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        on.ReverseSequence(input = input, sequence_lens = sequence_lens, ?batch_axis = batch_axis, ?time_axis = time_axis)
    static member reverse_sequence(input: Tensor<Complex>, sequence_lens: Tensor<int64>, ?batch_axis: int64, ?time_axis: int64) =
        on.ReverseSequence(input = input, sequence_lens = sequence_lens, ?batch_axis = batch_axis, ?time_axis = time_axis)
    static member lp_normalization(input: Tensor<float32>, ?axis: int64, ?p: int64) =
        on.LpNormalization(input = input, ?axis = axis, ?p = p)
    static member lp_normalization(input: Tensor<double>, ?axis: int64, ?p: int64) =
        on.LpNormalization(input = input, ?axis = axis, ?p = p)
    static member gemm(A: Tensor<float32>, B: Tensor<float32>, ?C: Tensor<float32>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        on.Gemm(A = A, B = B, ?C = C, ?alpha = alpha, ?beta = beta, ?transA = transA, ?transB = transB)
    static member gemm(A: Tensor<double>, B: Tensor<double>, ?C: Tensor<double>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        on.Gemm(A = A, B = B, ?C = C, ?alpha = alpha, ?beta = beta, ?transA = transA, ?transB = transB)
    static member gemm(A: Tensor<uint32>, B: Tensor<uint32>, ?C: Tensor<uint32>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        on.Gemm(A = A, B = B, ?C = C, ?alpha = alpha, ?beta = beta, ?transA = transA, ?transB = transB)
    static member gemm(A: Tensor<uint64>, B: Tensor<uint64>, ?C: Tensor<uint64>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        on.Gemm(A = A, B = B, ?C = C, ?alpha = alpha, ?beta = beta, ?transA = transA, ?transB = transB)
    static member gemm(A: Tensor<int>, B: Tensor<int>, ?C: Tensor<int>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        on.Gemm(A = A, B = B, ?C = C, ?alpha = alpha, ?beta = beta, ?transA = transA, ?transB = transB)
    static member gemm(A: Tensor<int64>, B: Tensor<int64>, ?C: Tensor<int64>, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        on.Gemm(A = A, B = B, ?C = C, ?alpha = alpha, ?beta = beta, ?transA = transA, ?transB = transB)
    static member global_lp_pool(X: Tensor<float32>, ?p: int64) =
        on.GlobalLpPool(X = X, ?p = p)
    static member global_lp_pool(X: Tensor<double>, ?p: int64) =
        on.GlobalLpPool(X = X, ?p = p)
    static member hard_swish(X: Tensor<float32>) =
        on.HardSwish(X = X)
    static member hard_swish(X: Tensor<double>) =
        on.HardSwish(X = X)
    static member mean([<ParamArray>]data_0: Tensor<float32>[]) =
        on.Mean(data_0 = data_0)
    static member mean([<ParamArray>]data_0: Tensor<double>[]) =
        on.Mean(data_0 = data_0)
    static member asin(input: Tensor<float32>) =
        on.Asin(input = input)
    static member asin(input: Tensor<double>) =
        on.Asin(input = input)
    static member one_hot_encoder(X: Tensor<string>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        on.OneHotEncoder(X = X, ?cats_int64s = cats_int64s, ?cats_strings = cats_strings, ?zeros = zeros)
    static member one_hot_encoder(X: Tensor<int64>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        on.OneHotEncoder(X = X, ?cats_int64s = cats_int64s, ?cats_strings = cats_strings, ?zeros = zeros)
    static member one_hot_encoder(X: Tensor<int>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        on.OneHotEncoder(X = X, ?cats_int64s = cats_int64s, ?cats_strings = cats_strings, ?zeros = zeros)
    static member one_hot_encoder(X: Tensor<float32>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        on.OneHotEncoder(X = X, ?cats_int64s = cats_int64s, ?cats_strings = cats_strings, ?zeros = zeros)
    static member one_hot_encoder(X: Tensor<double>, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        on.OneHotEncoder(X = X, ?cats_int64s = cats_int64s, ?cats_strings = cats_strings, ?zeros = zeros)
    static member depth_to_space(input: Tensor<uint8>, blocksize: int64, ?mode: string) =
        on.DepthToSpace(input = input, blocksize = blocksize, ?mode = mode)
    static member depth_to_space(input: Tensor<uint16>, blocksize: int64, ?mode: string) =
        on.DepthToSpace(input = input, blocksize = blocksize, ?mode = mode)
    static member depth_to_space(input: Tensor<uint32>, blocksize: int64, ?mode: string) =
        on.DepthToSpace(input = input, blocksize = blocksize, ?mode = mode)
    static member depth_to_space(input: Tensor<uint64>, blocksize: int64, ?mode: string) =
        on.DepthToSpace(input = input, blocksize = blocksize, ?mode = mode)
    static member depth_to_space(input: Tensor<int8>, blocksize: int64, ?mode: string) =
        on.DepthToSpace(input = input, blocksize = blocksize, ?mode = mode)
    static member depth_to_space(input: Tensor<int16>, blocksize: int64, ?mode: string) =
        on.DepthToSpace(input = input, blocksize = blocksize, ?mode = mode)
    static member depth_to_space(input: Tensor<int>, blocksize: int64, ?mode: string) =
        on.DepthToSpace(input = input, blocksize = blocksize, ?mode = mode)
    static member depth_to_space(input: Tensor<int64>, blocksize: int64, ?mode: string) =
        on.DepthToSpace(input = input, blocksize = blocksize, ?mode = mode)
    static member depth_to_space(input: Tensor<float32>, blocksize: int64, ?mode: string) =
        on.DepthToSpace(input = input, blocksize = blocksize, ?mode = mode)
    static member depth_to_space(input: Tensor<double>, blocksize: int64, ?mode: string) =
        on.DepthToSpace(input = input, blocksize = blocksize, ?mode = mode)
    static member depth_to_space(input: Tensor<string>, blocksize: int64, ?mode: string) =
        on.DepthToSpace(input = input, blocksize = blocksize, ?mode = mode)
    static member depth_to_space(input: Tensor<bool>, blocksize: int64, ?mode: string) =
        on.DepthToSpace(input = input, blocksize = blocksize, ?mode = mode)
    static member depth_to_space(input: Tensor<Complex>, blocksize: int64, ?mode: string) =
        on.DepthToSpace(input = input, blocksize = blocksize, ?mode = mode)
    static member div(A: Tensor<uint8>, B: Tensor<uint8>) =
        on.Div(A = A, B = B)
    static member div(A: Tensor<uint16>, B: Tensor<uint16>) =
        on.Div(A = A, B = B)
    static member div(A: Tensor<uint32>, B: Tensor<uint32>) =
        on.Div(A = A, B = B)
    static member div(A: Tensor<uint64>, B: Tensor<uint64>) =
        on.Div(A = A, B = B)
    static member div(A: Tensor<int8>, B: Tensor<int8>) =
        on.Div(A = A, B = B)
    static member div(A: Tensor<int16>, B: Tensor<int16>) =
        on.Div(A = A, B = B)
    static member div(A: Tensor<int>, B: Tensor<int>) =
        on.Div(A = A, B = B)
    static member div(A: Tensor<int64>, B: Tensor<int64>) =
        on.Div(A = A, B = B)
    static member div(A: Tensor<float32>, B: Tensor<float32>) =
        on.Div(A = A, B = B)
    static member div(A: Tensor<double>, B: Tensor<double>) =
        on.Div(A = A, B = B)
    static member softsign(input: Tensor<float32>) =
        on.Softsign(input = input)
    static member softsign(input: Tensor<double>) =
        on.Softsign(input = input)
    static member global_max_pool(X: Tensor<float32>) =
        on.GlobalMaxPool(X = X)
    static member global_max_pool(X: Tensor<double>) =
        on.GlobalMaxPool(X = X)
    static member reciprocal(X: Tensor<float32>) =
        on.Reciprocal(X = X)
    static member reciprocal(X: Tensor<double>) =
        on.Reciprocal(X = X)
    static member mean_variance_normalization(X: Tensor<float32>, ?axes: int64[]) =
        on.MeanVarianceNormalization(X = X, ?axes = axes)
    static member mean_variance_normalization(X: Tensor<double>, ?axes: int64[]) =
        on.MeanVarianceNormalization(X = X, ?axes = axes)
    static member reducel1(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceL1(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reducel1(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceL1(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reducel1(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceL1(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reducel1(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceL1(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reducel1(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceL1(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reducel1(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceL1(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member relu(X: Tensor<float32>) =
        on.Relu(X = X)
    static member relu(X: Tensor<int>) =
        on.Relu(X = X)
    static member relu(X: Tensor<int8>) =
        on.Relu(X = X)
    static member relu(X: Tensor<int16>) =
        on.Relu(X = X)
    static member relu(X: Tensor<int64>) =
        on.Relu(X = X)
    static member relu(X: Tensor<double>) =
        on.Relu(X = X)
    static member reduce_sum(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceSum(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_sum(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceSum(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_sum(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceSum(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_sum(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceSum(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_sum(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceSum(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_sum(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceSum(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member elu(X: Tensor<float32>, ?alpha: float32) =
        on.Elu(X = X, ?alpha = alpha)
    static member elu(X: Tensor<double>, ?alpha: float32) =
        on.Elu(X = X, ?alpha = alpha)
    static member reshape(data: Tensor<uint8>, shape: Tensor<int64>, ?allowzero: int64) =
        on.Reshape(data = data, shape = shape, ?allowzero = allowzero)
    static member reshape(data: Tensor<uint16>, shape: Tensor<int64>, ?allowzero: int64) =
        on.Reshape(data = data, shape = shape, ?allowzero = allowzero)
    static member reshape(data: Tensor<uint32>, shape: Tensor<int64>, ?allowzero: int64) =
        on.Reshape(data = data, shape = shape, ?allowzero = allowzero)
    static member reshape(data: Tensor<uint64>, shape: Tensor<int64>, ?allowzero: int64) =
        on.Reshape(data = data, shape = shape, ?allowzero = allowzero)
    static member reshape(data: Tensor<int8>, shape: Tensor<int64>, ?allowzero: int64) =
        on.Reshape(data = data, shape = shape, ?allowzero = allowzero)
    static member reshape(data: Tensor<int16>, shape: Tensor<int64>, ?allowzero: int64) =
        on.Reshape(data = data, shape = shape, ?allowzero = allowzero)
    static member reshape(data: Tensor<int>, shape: Tensor<int64>, ?allowzero: int64) =
        on.Reshape(data = data, shape = shape, ?allowzero = allowzero)
    static member reshape(data: Tensor<int64>, shape: Tensor<int64>, ?allowzero: int64) =
        on.Reshape(data = data, shape = shape, ?allowzero = allowzero)
    static member reshape(data: Tensor<float32>, shape: Tensor<int64>, ?allowzero: int64) =
        on.Reshape(data = data, shape = shape, ?allowzero = allowzero)
    static member reshape(data: Tensor<double>, shape: Tensor<int64>, ?allowzero: int64) =
        on.Reshape(data = data, shape = shape, ?allowzero = allowzero)
    static member reshape(data: Tensor<string>, shape: Tensor<int64>, ?allowzero: int64) =
        on.Reshape(data = data, shape = shape, ?allowzero = allowzero)
    static member reshape(data: Tensor<bool>, shape: Tensor<int64>, ?allowzero: int64) =
        on.Reshape(data = data, shape = shape, ?allowzero = allowzero)
    static member reshape(data: Tensor<Complex>, shape: Tensor<int64>, ?allowzero: int64) =
        on.Reshape(data = data, shape = shape, ?allowzero = allowzero)
    static member selu(X: Tensor<float32>, ?alpha: float32, ?gamma: float32) =
        on.Selu(X = X, ?alpha = alpha, ?gamma = gamma)
    static member selu(X: Tensor<double>, ?alpha: float32, ?gamma: float32) =
        on.Selu(X = X, ?alpha = alpha, ?gamma = gamma)
    static member global_average_pool(X: Tensor<float32>) =
        on.GlobalAveragePool(X = X)
    static member global_average_pool(X: Tensor<double>) =
        on.GlobalAveragePool(X = X)
    static member hard_sigmoid(X: Tensor<float32>, ?alpha: float32, ?beta: float32) =
        on.HardSigmoid(X = X, ?alpha = alpha, ?beta = beta)
    static member hard_sigmoid(X: Tensor<double>, ?alpha: float32, ?beta: float32) =
        on.HardSigmoid(X = X, ?alpha = alpha, ?beta = beta)
    static member log_softmax(input: Tensor<float32>, ?axis: int64) =
        on.LogSoftmax(input = input, ?axis = axis)
    static member log_softmax(input: Tensor<double>, ?axis: int64) =
        on.LogSoftmax(input = input, ?axis = axis)
    static member space_to_depth(input: Tensor<uint8>, blocksize: int64) =
        on.SpaceToDepth(input = input, blocksize = blocksize)
    static member space_to_depth(input: Tensor<uint16>, blocksize: int64) =
        on.SpaceToDepth(input = input, blocksize = blocksize)
    static member space_to_depth(input: Tensor<uint32>, blocksize: int64) =
        on.SpaceToDepth(input = input, blocksize = blocksize)
    static member space_to_depth(input: Tensor<uint64>, blocksize: int64) =
        on.SpaceToDepth(input = input, blocksize = blocksize)
    static member space_to_depth(input: Tensor<int8>, blocksize: int64) =
        on.SpaceToDepth(input = input, blocksize = blocksize)
    static member space_to_depth(input: Tensor<int16>, blocksize: int64) =
        on.SpaceToDepth(input = input, blocksize = blocksize)
    static member space_to_depth(input: Tensor<int>, blocksize: int64) =
        on.SpaceToDepth(input = input, blocksize = blocksize)
    static member space_to_depth(input: Tensor<int64>, blocksize: int64) =
        on.SpaceToDepth(input = input, blocksize = blocksize)
    static member space_to_depth(input: Tensor<float32>, blocksize: int64) =
        on.SpaceToDepth(input = input, blocksize = blocksize)
    static member space_to_depth(input: Tensor<double>, blocksize: int64) =
        on.SpaceToDepth(input = input, blocksize = blocksize)
    static member space_to_depth(input: Tensor<string>, blocksize: int64) =
        on.SpaceToDepth(input = input, blocksize = blocksize)
    static member space_to_depth(input: Tensor<bool>, blocksize: int64) =
        on.SpaceToDepth(input = input, blocksize = blocksize)
    static member space_to_depth(input: Tensor<Complex>, blocksize: int64) =
        on.SpaceToDepth(input = input, blocksize = blocksize)
    static member bitwise_not(X: Tensor<uint8>) =
        on.BitwiseNot(X = X)
    static member bitwise_not(X: Tensor<uint16>) =
        on.BitwiseNot(X = X)
    static member bitwise_not(X: Tensor<uint32>) =
        on.BitwiseNot(X = X)
    static member bitwise_not(X: Tensor<uint64>) =
        on.BitwiseNot(X = X)
    static member bitwise_not(X: Tensor<int8>) =
        on.BitwiseNot(X = X)
    static member bitwise_not(X: Tensor<int16>) =
        on.BitwiseNot(X = X)
    static member bitwise_not(X: Tensor<int>) =
        on.BitwiseNot(X = X)
    static member bitwise_not(X: Tensor<int64>) =
        on.BitwiseNot(X = X)
    static member reduce_sum_square(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceSumSquare(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_sum_square(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceSumSquare(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_sum_square(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceSumSquare(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_sum_square(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceSumSquare(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_sum_square(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceSumSquare(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_sum_square(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceSumSquare(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member sqrt(X: Tensor<float32>) =
        on.Sqrt(X = X)
    static member sqrt(X: Tensor<double>) =
        on.Sqrt(X = X)
    static member col2_im(input: Tensor<uint8>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Col2Im(input = input, image_shape = image_shape, block_shape = block_shape, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member col2_im(input: Tensor<uint16>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Col2Im(input = input, image_shape = image_shape, block_shape = block_shape, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member col2_im(input: Tensor<uint32>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Col2Im(input = input, image_shape = image_shape, block_shape = block_shape, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member col2_im(input: Tensor<uint64>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Col2Im(input = input, image_shape = image_shape, block_shape = block_shape, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member col2_im(input: Tensor<int8>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Col2Im(input = input, image_shape = image_shape, block_shape = block_shape, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member col2_im(input: Tensor<int16>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Col2Im(input = input, image_shape = image_shape, block_shape = block_shape, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member col2_im(input: Tensor<int>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Col2Im(input = input, image_shape = image_shape, block_shape = block_shape, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member col2_im(input: Tensor<int64>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Col2Im(input = input, image_shape = image_shape, block_shape = block_shape, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member col2_im(input: Tensor<float32>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Col2Im(input = input, image_shape = image_shape, block_shape = block_shape, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member col2_im(input: Tensor<double>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Col2Im(input = input, image_shape = image_shape, block_shape = block_shape, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member col2_im(input: Tensor<string>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Col2Im(input = input, image_shape = image_shape, block_shape = block_shape, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member col2_im(input: Tensor<bool>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Col2Im(input = input, image_shape = image_shape, block_shape = block_shape, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member col2_im(input: Tensor<Complex>, image_shape: Tensor<int64>, block_shape: Tensor<int64>, ?dilations: int64[], ?pads: int64[], ?strides: int64[]) =
        on.Col2Im(input = input, image_shape = image_shape, block_shape = block_shape, ?dilations = dilations, ?pads = pads, ?strides = strides)
    static member identity(input: Tensor<uint8>) =
        on.Identity(input = input)
    static member identity(input: Tensor<uint16>) =
        on.Identity(input = input)
    static member identity(input: Tensor<uint32>) =
        on.Identity(input = input)
    static member identity(input: Tensor<uint64>) =
        on.Identity(input = input)
    static member identity(input: Tensor<int8>) =
        on.Identity(input = input)
    static member identity(input: Tensor<int16>) =
        on.Identity(input = input)
    static member identity(input: Tensor<int>) =
        on.Identity(input = input)
    static member identity(input: Tensor<int64>) =
        on.Identity(input = input)
    static member identity(input: Tensor<float32>) =
        on.Identity(input = input)
    static member identity(input: Tensor<double>) =
        on.Identity(input = input)
    static member identity(input: Tensor<string>) =
        on.Identity(input = input)
    static member identity(input: Tensor<bool>) =
        on.Identity(input = input)
    static member identity(input: Tensor<Complex>) =
        on.Identity(input = input)
    static member expand(input: Tensor<uint8>, shape: Tensor<int64>) =
        on.Expand(input = input, shape = shape)
    static member expand(input: Tensor<uint16>, shape: Tensor<int64>) =
        on.Expand(input = input, shape = shape)
    static member expand(input: Tensor<uint32>, shape: Tensor<int64>) =
        on.Expand(input = input, shape = shape)
    static member expand(input: Tensor<uint64>, shape: Tensor<int64>) =
        on.Expand(input = input, shape = shape)
    static member expand(input: Tensor<int8>, shape: Tensor<int64>) =
        on.Expand(input = input, shape = shape)
    static member expand(input: Tensor<int16>, shape: Tensor<int64>) =
        on.Expand(input = input, shape = shape)
    static member expand(input: Tensor<int>, shape: Tensor<int64>) =
        on.Expand(input = input, shape = shape)
    static member expand(input: Tensor<int64>, shape: Tensor<int64>) =
        on.Expand(input = input, shape = shape)
    static member expand(input: Tensor<float32>, shape: Tensor<int64>) =
        on.Expand(input = input, shape = shape)
    static member expand(input: Tensor<double>, shape: Tensor<int64>) =
        on.Expand(input = input, shape = shape)
    static member expand(input: Tensor<string>, shape: Tensor<int64>) =
        on.Expand(input = input, shape = shape)
    static member expand(input: Tensor<bool>, shape: Tensor<int64>) =
        on.Expand(input = input, shape = shape)
    static member expand(input: Tensor<Complex>, shape: Tensor<int64>) =
        on.Expand(input = input, shape = shape)
    static member squeeze(data: Tensor<uint8>, ?axes: Tensor<int64>) =
        on.Squeeze(data = data, ?axes = axes)
    static member squeeze(data: Tensor<uint16>, ?axes: Tensor<int64>) =
        on.Squeeze(data = data, ?axes = axes)
    static member squeeze(data: Tensor<uint32>, ?axes: Tensor<int64>) =
        on.Squeeze(data = data, ?axes = axes)
    static member squeeze(data: Tensor<uint64>, ?axes: Tensor<int64>) =
        on.Squeeze(data = data, ?axes = axes)
    static member squeeze(data: Tensor<int8>, ?axes: Tensor<int64>) =
        on.Squeeze(data = data, ?axes = axes)
    static member squeeze(data: Tensor<int16>, ?axes: Tensor<int64>) =
        on.Squeeze(data = data, ?axes = axes)
    static member squeeze(data: Tensor<int>, ?axes: Tensor<int64>) =
        on.Squeeze(data = data, ?axes = axes)
    static member squeeze(data: Tensor<int64>, ?axes: Tensor<int64>) =
        on.Squeeze(data = data, ?axes = axes)
    static member squeeze(data: Tensor<float32>, ?axes: Tensor<int64>) =
        on.Squeeze(data = data, ?axes = axes)
    static member squeeze(data: Tensor<double>, ?axes: Tensor<int64>) =
        on.Squeeze(data = data, ?axes = axes)
    static member squeeze(data: Tensor<string>, ?axes: Tensor<int64>) =
        on.Squeeze(data = data, ?axes = axes)
    static member squeeze(data: Tensor<bool>, ?axes: Tensor<int64>) =
        on.Squeeze(data = data, ?axes = axes)
    static member squeeze(data: Tensor<Complex>, ?axes: Tensor<int64>) =
        on.Squeeze(data = data, ?axes = axes)
    static member sum([<ParamArray>]data_0: Tensor<float32>[]) =
        on.Sum(data_0 = data_0)
    static member sum([<ParamArray>]data_0: Tensor<double>[]) =
        on.Sum(data_0 = data_0)
    static member upsample(X: Tensor<uint8>, scales: Tensor<float32>, ?mode: string) =
        on.Upsample(X = X, scales = scales, ?mode = mode)
    static member upsample(X: Tensor<uint16>, scales: Tensor<float32>, ?mode: string) =
        on.Upsample(X = X, scales = scales, ?mode = mode)
    static member upsample(X: Tensor<uint32>, scales: Tensor<float32>, ?mode: string) =
        on.Upsample(X = X, scales = scales, ?mode = mode)
    static member upsample(X: Tensor<uint64>, scales: Tensor<float32>, ?mode: string) =
        on.Upsample(X = X, scales = scales, ?mode = mode)
    static member upsample(X: Tensor<int8>, scales: Tensor<float32>, ?mode: string) =
        on.Upsample(X = X, scales = scales, ?mode = mode)
    static member upsample(X: Tensor<int16>, scales: Tensor<float32>, ?mode: string) =
        on.Upsample(X = X, scales = scales, ?mode = mode)
    static member upsample(X: Tensor<int>, scales: Tensor<float32>, ?mode: string) =
        on.Upsample(X = X, scales = scales, ?mode = mode)
    static member upsample(X: Tensor<int64>, scales: Tensor<float32>, ?mode: string) =
        on.Upsample(X = X, scales = scales, ?mode = mode)
    static member upsample(X: Tensor<float32>, scales: Tensor<float32>, ?mode: string) =
        on.Upsample(X = X, scales = scales, ?mode = mode)
    static member upsample(X: Tensor<double>, scales: Tensor<float32>, ?mode: string) =
        on.Upsample(X = X, scales = scales, ?mode = mode)
    static member upsample(X: Tensor<string>, scales: Tensor<float32>, ?mode: string) =
        on.Upsample(X = X, scales = scales, ?mode = mode)
    static member upsample(X: Tensor<bool>, scales: Tensor<float32>, ?mode: string) =
        on.Upsample(X = X, scales = scales, ?mode = mode)
    static member upsample(X: Tensor<Complex>, scales: Tensor<float32>, ?mode: string) =
        on.Upsample(X = X, scales = scales, ?mode = mode)
    static member tanh(input: Tensor<float32>) =
        on.Tanh(input = input)
    static member tanh(input: Tensor<double>) =
        on.Tanh(input = input)
    static member lrn(X: Tensor<float32>, size: int64, ?alpha: float32, ?beta: float32, ?bias: float32) =
        on.LRN(X = X, size = size, ?alpha = alpha, ?beta = beta, ?bias = bias)
    static member lrn(X: Tensor<double>, size: int64, ?alpha: float32, ?beta: float32, ?bias: float32) =
        on.LRN(X = X, size = size, ?alpha = alpha, ?beta = beta, ?bias = bias)
    static member unsqueeze(data: Tensor<uint8>, axes: Tensor<int64>) =
        on.Unsqueeze(data = data, axes = axes)
    static member unsqueeze(data: Tensor<uint16>, axes: Tensor<int64>) =
        on.Unsqueeze(data = data, axes = axes)
    static member unsqueeze(data: Tensor<uint32>, axes: Tensor<int64>) =
        on.Unsqueeze(data = data, axes = axes)
    static member unsqueeze(data: Tensor<uint64>, axes: Tensor<int64>) =
        on.Unsqueeze(data = data, axes = axes)
    static member unsqueeze(data: Tensor<int8>, axes: Tensor<int64>) =
        on.Unsqueeze(data = data, axes = axes)
    static member unsqueeze(data: Tensor<int16>, axes: Tensor<int64>) =
        on.Unsqueeze(data = data, axes = axes)
    static member unsqueeze(data: Tensor<int>, axes: Tensor<int64>) =
        on.Unsqueeze(data = data, axes = axes)
    static member unsqueeze(data: Tensor<int64>, axes: Tensor<int64>) =
        on.Unsqueeze(data = data, axes = axes)
    static member unsqueeze(data: Tensor<float32>, axes: Tensor<int64>) =
        on.Unsqueeze(data = data, axes = axes)
    static member unsqueeze(data: Tensor<double>, axes: Tensor<int64>) =
        on.Unsqueeze(data = data, axes = axes)
    static member unsqueeze(data: Tensor<string>, axes: Tensor<int64>) =
        on.Unsqueeze(data = data, axes = axes)
    static member unsqueeze(data: Tensor<bool>, axes: Tensor<int64>) =
        on.Unsqueeze(data = data, axes = axes)
    static member unsqueeze(data: Tensor<Complex>, axes: Tensor<int64>) =
        on.Unsqueeze(data = data, axes = axes)
    static member thresholded_relu(X: Tensor<float32>, ?alpha: float32) =
        on.ThresholdedRelu(X = X, ?alpha = alpha)
    static member thresholded_relu(X: Tensor<double>, ?alpha: float32) =
        on.ThresholdedRelu(X = X, ?alpha = alpha)
    static member acos(input: Tensor<float32>) =
        on.Acos(input = input)
    static member acos(input: Tensor<double>) =
        on.Acos(input = input)
    static member atan(input: Tensor<float32>) =
        on.Atan(input = input)
    static member atan(input: Tensor<double>) =
        on.Atan(input = input)
    static member cos(input: Tensor<float32>) =
        on.Cos(input = input)
    static member cos(input: Tensor<double>) =
        on.Cos(input = input)
    static member sin(input: Tensor<float32>) =
        on.Sin(input = input)
    static member sin(input: Tensor<double>) =
        on.Sin(input = input)
    static member transpose(data: Tensor<uint8>, ?perm: int64[]) =
        on.Transpose(data = data, ?perm = perm)
    static member transpose(data: Tensor<uint16>, ?perm: int64[]) =
        on.Transpose(data = data, ?perm = perm)
    static member transpose(data: Tensor<uint32>, ?perm: int64[]) =
        on.Transpose(data = data, ?perm = perm)
    static member transpose(data: Tensor<uint64>, ?perm: int64[]) =
        on.Transpose(data = data, ?perm = perm)
    static member transpose(data: Tensor<int8>, ?perm: int64[]) =
        on.Transpose(data = data, ?perm = perm)
    static member transpose(data: Tensor<int16>, ?perm: int64[]) =
        on.Transpose(data = data, ?perm = perm)
    static member transpose(data: Tensor<int>, ?perm: int64[]) =
        on.Transpose(data = data, ?perm = perm)
    static member transpose(data: Tensor<int64>, ?perm: int64[]) =
        on.Transpose(data = data, ?perm = perm)
    static member transpose(data: Tensor<float32>, ?perm: int64[]) =
        on.Transpose(data = data, ?perm = perm)
    static member transpose(data: Tensor<double>, ?perm: int64[]) =
        on.Transpose(data = data, ?perm = perm)
    static member transpose(data: Tensor<string>, ?perm: int64[]) =
        on.Transpose(data = data, ?perm = perm)
    static member transpose(data: Tensor<bool>, ?perm: int64[]) =
        on.Transpose(data = data, ?perm = perm)
    static member transpose(data: Tensor<Complex>, ?perm: int64[]) =
        on.Transpose(data = data, ?perm = perm)
    static member reduce_prod(data: Tensor<uint32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceProd(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_prod(data: Tensor<uint64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceProd(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_prod(data: Tensor<int>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceProd(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_prod(data: Tensor<int64>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceProd(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_prod(data: Tensor<float32>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceProd(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member reduce_prod(data: Tensor<double>, ?axes: Tensor<int64>, ?keepdims: int64, ?noop_with_empty_axes: int64) =
        on.ReduceProd(data = data, ?axes = axes, ?keepdims = keepdims, ?noop_with_empty_axes = noop_with_empty_axes)
    static member sinh(input: Tensor<float32>) =
        on.Sinh(input = input)
    static member sinh(input: Tensor<double>) =
        on.Sinh(input = input)
    static member asinh(input: Tensor<float32>) =
        on.Asinh(input = input)
    static member asinh(input: Tensor<double>) =
        on.Asinh(input = input)
    static member binarizer(X: Tensor<float32>, ?threshold: float32) =
        on.Binarizer(X = X, ?threshold = threshold)
    static member binarizer(X: Tensor<double>, ?threshold: float32) =
        on.Binarizer(X = X, ?threshold = threshold)
    static member binarizer(X: Tensor<int64>, ?threshold: float32) =
        on.Binarizer(X = X, ?threshold = threshold)
    static member binarizer(X: Tensor<int>, ?threshold: float32) =
        on.Binarizer(X = X, ?threshold = threshold)
    static member trilu(input: Tensor<uint8>, ?k: Tensor<int64>, ?upper: int64) =
        on.Trilu(input = input, ?k = k, ?upper = upper)
    static member trilu(input: Tensor<uint16>, ?k: Tensor<int64>, ?upper: int64) =
        on.Trilu(input = input, ?k = k, ?upper = upper)
    static member trilu(input: Tensor<uint32>, ?k: Tensor<int64>, ?upper: int64) =
        on.Trilu(input = input, ?k = k, ?upper = upper)
    static member trilu(input: Tensor<uint64>, ?k: Tensor<int64>, ?upper: int64) =
        on.Trilu(input = input, ?k = k, ?upper = upper)
    static member trilu(input: Tensor<int8>, ?k: Tensor<int64>, ?upper: int64) =
        on.Trilu(input = input, ?k = k, ?upper = upper)
    static member trilu(input: Tensor<int16>, ?k: Tensor<int64>, ?upper: int64) =
        on.Trilu(input = input, ?k = k, ?upper = upper)
    static member trilu(input: Tensor<int>, ?k: Tensor<int64>, ?upper: int64) =
        on.Trilu(input = input, ?k = k, ?upper = upper)
    static member trilu(input: Tensor<int64>, ?k: Tensor<int64>, ?upper: int64) =
        on.Trilu(input = input, ?k = k, ?upper = upper)
    static member trilu(input: Tensor<float32>, ?k: Tensor<int64>, ?upper: int64) =
        on.Trilu(input = input, ?k = k, ?upper = upper)
    static member trilu(input: Tensor<double>, ?k: Tensor<int64>, ?upper: int64) =
        on.Trilu(input = input, ?k = k, ?upper = upper)
    static member trilu(input: Tensor<string>, ?k: Tensor<int64>, ?upper: int64) =
        on.Trilu(input = input, ?k = k, ?upper = upper)
    static member trilu(input: Tensor<bool>, ?k: Tensor<int64>, ?upper: int64) =
        on.Trilu(input = input, ?k = k, ?upper = upper)
    static member trilu(input: Tensor<Complex>, ?k: Tensor<int64>, ?upper: int64) =
        on.Trilu(input = input, ?k = k, ?upper = upper)
    static member acosh(input: Tensor<float32>) =
        on.Acosh(input = input)
    static member acosh(input: Tensor<double>) =
        on.Acosh(input = input)
    static member cosh(input: Tensor<float32>) =
        on.Cosh(input = input)
    static member cosh(input: Tensor<double>) =
        on.Cosh(input = input)
    static member atanh(input: Tensor<float32>) =
        on.Atanh(input = input)
    static member atanh(input: Tensor<double>) =
        on.Atanh(input = input)
    static member shrink(input: Tensor<uint8>, ?bias: float32, ?lambd: float32) =
        on.Shrink(input = input, ?bias = bias, ?lambd = lambd)
    static member shrink(input: Tensor<uint16>, ?bias: float32, ?lambd: float32) =
        on.Shrink(input = input, ?bias = bias, ?lambd = lambd)
    static member shrink(input: Tensor<uint32>, ?bias: float32, ?lambd: float32) =
        on.Shrink(input = input, ?bias = bias, ?lambd = lambd)
    static member shrink(input: Tensor<uint64>, ?bias: float32, ?lambd: float32) =
        on.Shrink(input = input, ?bias = bias, ?lambd = lambd)
    static member shrink(input: Tensor<int8>, ?bias: float32, ?lambd: float32) =
        on.Shrink(input = input, ?bias = bias, ?lambd = lambd)
    static member shrink(input: Tensor<int16>, ?bias: float32, ?lambd: float32) =
        on.Shrink(input = input, ?bias = bias, ?lambd = lambd)
    static member shrink(input: Tensor<int>, ?bias: float32, ?lambd: float32) =
        on.Shrink(input = input, ?bias = bias, ?lambd = lambd)
    static member shrink(input: Tensor<int64>, ?bias: float32, ?lambd: float32) =
        on.Shrink(input = input, ?bias = bias, ?lambd = lambd)
    static member shrink(input: Tensor<float32>, ?bias: float32, ?lambd: float32) =
        on.Shrink(input = input, ?bias = bias, ?lambd = lambd)
    static member shrink(input: Tensor<double>, ?bias: float32, ?lambd: float32) =
        on.Shrink(input = input, ?bias = bias, ?lambd = lambd)
    static member hardmax(input: Tensor<float32>, ?axis: int64) =
        on.Hardmax(input = input, ?axis = axis)
    static member hardmax(input: Tensor<double>, ?axis: int64) =
        on.Hardmax(input = input, ?axis = axis)
    static member erf(input: Tensor<uint8>) =
        on.Erf(input = input)
    static member erf(input: Tensor<uint16>) =
        on.Erf(input = input)
    static member erf(input: Tensor<uint32>) =
        on.Erf(input = input)
    static member erf(input: Tensor<uint64>) =
        on.Erf(input = input)
    static member erf(input: Tensor<int8>) =
        on.Erf(input = input)
    static member erf(input: Tensor<int16>) =
        on.Erf(input = input)
    static member erf(input: Tensor<int>) =
        on.Erf(input = input)
    static member erf(input: Tensor<int64>) =
        on.Erf(input = input)
    static member erf(input: Tensor<float32>) =
        on.Erf(input = input)
    static member erf(input: Tensor<double>) =
        on.Erf(input = input)
    static member optional_has_element(?input: Tensor<uint8>) =
        on.OptionalHasElement(?input = input)
    static member optional_has_element(?input: Tensor<uint16>) =
        on.OptionalHasElement(?input = input)
    static member optional_has_element(?input: Tensor<uint32>) =
        on.OptionalHasElement(?input = input)
    static member optional_has_element(?input: Tensor<uint64>) =
        on.OptionalHasElement(?input = input)
    static member optional_has_element(?input: Tensor<int8>) =
        on.OptionalHasElement(?input = input)
    static member optional_has_element(?input: Tensor<int16>) =
        on.OptionalHasElement(?input = input)
    static member optional_has_element(?input: Tensor<int>) =
        on.OptionalHasElement(?input = input)
    static member optional_has_element(?input: Tensor<int64>) =
        on.OptionalHasElement(?input = input)
    static member optional_has_element(?input: Tensor<float32>) =
        on.OptionalHasElement(?input = input)
    static member optional_has_element(?input: Tensor<double>) =
        on.OptionalHasElement(?input = input)
    static member optional_has_element(?input: Tensor<string>) =
        on.OptionalHasElement(?input = input)
    static member optional_has_element(?input: Tensor<bool>) =
        on.OptionalHasElement(?input = input)
    static member optional_has_element(?input: Tensor<Complex>) =
        on.OptionalHasElement(?input = input)
    static member greater_or_equal(A: Tensor<uint8>, B: Tensor<uint8>) =
        on.GreaterOrEqual(A = A, B = B)
    static member greater_or_equal(A: Tensor<uint16>, B: Tensor<uint16>) =
        on.GreaterOrEqual(A = A, B = B)
    static member greater_or_equal(A: Tensor<uint32>, B: Tensor<uint32>) =
        on.GreaterOrEqual(A = A, B = B)
    static member greater_or_equal(A: Tensor<uint64>, B: Tensor<uint64>) =
        on.GreaterOrEqual(A = A, B = B)
    static member greater_or_equal(A: Tensor<int8>, B: Tensor<int8>) =
        on.GreaterOrEqual(A = A, B = B)
    static member greater_or_equal(A: Tensor<int16>, B: Tensor<int16>) =
        on.GreaterOrEqual(A = A, B = B)
    static member greater_or_equal(A: Tensor<int>, B: Tensor<int>) =
        on.GreaterOrEqual(A = A, B = B)
    static member greater_or_equal(A: Tensor<int64>, B: Tensor<int64>) =
        on.GreaterOrEqual(A = A, B = B)
    static member greater_or_equal(A: Tensor<float32>, B: Tensor<float32>) =
        on.GreaterOrEqual(A = A, B = B)
    static member greater_or_equal(A: Tensor<double>, B: Tensor<double>) =
        on.GreaterOrEqual(A = A, B = B)
    static member less_or_equal(A: Tensor<uint8>, B: Tensor<uint8>) =
        on.LessOrEqual(A = A, B = B)
    static member less_or_equal(A: Tensor<uint16>, B: Tensor<uint16>) =
        on.LessOrEqual(A = A, B = B)
    static member less_or_equal(A: Tensor<uint32>, B: Tensor<uint32>) =
        on.LessOrEqual(A = A, B = B)
    static member less_or_equal(A: Tensor<uint64>, B: Tensor<uint64>) =
        on.LessOrEqual(A = A, B = B)
    static member less_or_equal(A: Tensor<int8>, B: Tensor<int8>) =
        on.LessOrEqual(A = A, B = B)
    static member less_or_equal(A: Tensor<int16>, B: Tensor<int16>) =
        on.LessOrEqual(A = A, B = B)
    static member less_or_equal(A: Tensor<int>, B: Tensor<int>) =
        on.LessOrEqual(A = A, B = B)
    static member less_or_equal(A: Tensor<int64>, B: Tensor<int64>) =
        on.LessOrEqual(A = A, B = B)
    static member less_or_equal(A: Tensor<float32>, B: Tensor<float32>) =
        on.LessOrEqual(A = A, B = B)
    static member less_or_equal(A: Tensor<double>, B: Tensor<double>) =
        on.LessOrEqual(A = A, B = B)
    static member is_inf(X: Tensor<float32>, ?detect_negative: int64, ?detect_positive: int64) =
        on.IsInf(X = X, ?detect_negative = detect_negative, ?detect_positive = detect_positive)
    static member is_inf(X: Tensor<double>, ?detect_negative: int64, ?detect_positive: int64) =
        on.IsInf(X = X, ?detect_negative = detect_negative, ?detect_positive = detect_positive)
    static member or_(A: Tensor<bool>, B: Tensor<bool>) =
        on.Or(A = A, B = B)
    static member tf_idf_vectorizer(X: Tensor<string>, max_gram_length: int64, max_skip_count: int64, min_gram_length: int64, mode: string, ngram_counts: int64[], ngram_indexes: int64[], ?pool_int64s: int64[], ?pool_strings: string[], ?weights: float32[]) =
        on.TfIdfVectorizer(X = X, max_gram_length = max_gram_length, max_skip_count = max_skip_count, min_gram_length = min_gram_length, mode = mode, ngram_counts = ngram_counts, ngram_indexes = ngram_indexes, ?pool_int64s = pool_int64s, ?pool_strings = pool_strings, ?weights = weights)
    static member tf_idf_vectorizer(X: Tensor<int>, max_gram_length: int64, max_skip_count: int64, min_gram_length: int64, mode: string, ngram_counts: int64[], ngram_indexes: int64[], ?pool_int64s: int64[], ?pool_strings: string[], ?weights: float32[]) =
        on.TfIdfVectorizer(X = X, max_gram_length = max_gram_length, max_skip_count = max_skip_count, min_gram_length = min_gram_length, mode = mode, ngram_counts = ngram_counts, ngram_indexes = ngram_indexes, ?pool_int64s = pool_int64s, ?pool_strings = pool_strings, ?weights = weights)
    static member tf_idf_vectorizer(X: Tensor<int64>, max_gram_length: int64, max_skip_count: int64, min_gram_length: int64, mode: string, ngram_counts: int64[], ngram_indexes: int64[], ?pool_int64s: int64[], ?pool_strings: string[], ?weights: float32[]) =
        on.TfIdfVectorizer(X = X, max_gram_length = max_gram_length, max_skip_count = max_skip_count, min_gram_length = min_gram_length, mode = mode, ngram_counts = ngram_counts, ngram_indexes = ngram_indexes, ?pool_int64s = pool_int64s, ?pool_strings = pool_strings, ?weights = weights)
    static member and_(A: Tensor<bool>, B: Tensor<bool>) =
        on.And(A = A, B = B)
    static member less(A: Tensor<uint8>, B: Tensor<uint8>) =
        on.Less(A = A, B = B)
    static member less(A: Tensor<uint16>, B: Tensor<uint16>) =
        on.Less(A = A, B = B)
    static member less(A: Tensor<uint32>, B: Tensor<uint32>) =
        on.Less(A = A, B = B)
    static member less(A: Tensor<uint64>, B: Tensor<uint64>) =
        on.Less(A = A, B = B)
    static member less(A: Tensor<int8>, B: Tensor<int8>) =
        on.Less(A = A, B = B)
    static member less(A: Tensor<int16>, B: Tensor<int16>) =
        on.Less(A = A, B = B)
    static member less(A: Tensor<int>, B: Tensor<int>) =
        on.Less(A = A, B = B)
    static member less(A: Tensor<int64>, B: Tensor<int64>) =
        on.Less(A = A, B = B)
    static member less(A: Tensor<float32>, B: Tensor<float32>) =
        on.Less(A = A, B = B)
    static member less(A: Tensor<double>, B: Tensor<double>) =
        on.Less(A = A, B = B)
    static member equal(A: Tensor<bool>, B: Tensor<bool>) =
        on.Equal(A = A, B = B)
    static member equal(A: Tensor<uint8>, B: Tensor<uint8>) =
        on.Equal(A = A, B = B)
    static member equal(A: Tensor<uint16>, B: Tensor<uint16>) =
        on.Equal(A = A, B = B)
    static member equal(A: Tensor<uint32>, B: Tensor<uint32>) =
        on.Equal(A = A, B = B)
    static member equal(A: Tensor<uint64>, B: Tensor<uint64>) =
        on.Equal(A = A, B = B)
    static member equal(A: Tensor<int8>, B: Tensor<int8>) =
        on.Equal(A = A, B = B)
    static member equal(A: Tensor<int16>, B: Tensor<int16>) =
        on.Equal(A = A, B = B)
    static member equal(A: Tensor<int>, B: Tensor<int>) =
        on.Equal(A = A, B = B)
    static member equal(A: Tensor<int64>, B: Tensor<int64>) =
        on.Equal(A = A, B = B)
    static member equal(A: Tensor<float32>, B: Tensor<float32>) =
        on.Equal(A = A, B = B)
    static member equal(A: Tensor<double>, B: Tensor<double>) =
        on.Equal(A = A, B = B)
    static member equal(A: Tensor<string>, B: Tensor<string>) =
        on.Equal(A = A, B = B)
    static member greater(A: Tensor<uint8>, B: Tensor<uint8>) =
        on.Greater(A = A, B = B)
    static member greater(A: Tensor<uint16>, B: Tensor<uint16>) =
        on.Greater(A = A, B = B)
    static member greater(A: Tensor<uint32>, B: Tensor<uint32>) =
        on.Greater(A = A, B = B)
    static member greater(A: Tensor<uint64>, B: Tensor<uint64>) =
        on.Greater(A = A, B = B)
    static member greater(A: Tensor<int8>, B: Tensor<int8>) =
        on.Greater(A = A, B = B)
    static member greater(A: Tensor<int16>, B: Tensor<int16>) =
        on.Greater(A = A, B = B)
    static member greater(A: Tensor<int>, B: Tensor<int>) =
        on.Greater(A = A, B = B)
    static member greater(A: Tensor<int64>, B: Tensor<int64>) =
        on.Greater(A = A, B = B)
    static member greater(A: Tensor<float32>, B: Tensor<float32>) =
        on.Greater(A = A, B = B)
    static member greater(A: Tensor<double>, B: Tensor<double>) =
        on.Greater(A = A, B = B)
    static member is_nan(X: Tensor<float32>) =
        on.IsNaN(X = X)
    static member is_nan(X: Tensor<double>) =
        on.IsNaN(X = X)
    static member shape(data: Tensor<uint8>, ?end: int64, ?start: int64) =
        on.Shape(data = data, ?end = end, ?start = start)
    static member shape(data: Tensor<uint16>, ?end: int64, ?start: int64) =
        on.Shape(data = data, ?end = end, ?start = start)
    static member shape(data: Tensor<uint32>, ?end: int64, ?start: int64) =
        on.Shape(data = data, ?end = end, ?start = start)
    static member shape(data: Tensor<uint64>, ?end: int64, ?start: int64) =
        on.Shape(data = data, ?end = end, ?start = start)
    static member shape(data: Tensor<int8>, ?end: int64, ?start: int64) =
        on.Shape(data = data, ?end = end, ?start = start)
    static member shape(data: Tensor<int16>, ?end: int64, ?start: int64) =
        on.Shape(data = data, ?end = end, ?start = start)
    static member shape(data: Tensor<int>, ?end: int64, ?start: int64) =
        on.Shape(data = data, ?end = end, ?start = start)
    static member shape(data: Tensor<int64>, ?end: int64, ?start: int64) =
        on.Shape(data = data, ?end = end, ?start = start)
    static member shape(data: Tensor<float32>, ?end: int64, ?start: int64) =
        on.Shape(data = data, ?end = end, ?start = start)
    static member shape(data: Tensor<double>, ?end: int64, ?start: int64) =
        on.Shape(data = data, ?end = end, ?start = start)
    static member shape(data: Tensor<string>, ?end: int64, ?start: int64) =
        on.Shape(data = data, ?end = end, ?start = start)
    static member shape(data: Tensor<bool>, ?end: int64, ?start: int64) =
        on.Shape(data = data, ?end = end, ?start = start)
    static member shape(data: Tensor<Complex>, ?end: int64, ?start: int64) =
        on.Shape(data = data, ?end = end, ?start = start)
    static member size(data: Tensor<uint8>) =
        on.Size(data = data)
    static member size(data: Tensor<uint16>) =
        on.Size(data = data)
    static member size(data: Tensor<uint32>) =
        on.Size(data = data)
    static member size(data: Tensor<uint64>) =
        on.Size(data = data)
    static member size(data: Tensor<int8>) =
        on.Size(data = data)
    static member size(data: Tensor<int16>) =
        on.Size(data = data)
    static member size(data: Tensor<int>) =
        on.Size(data = data)
    static member size(data: Tensor<int64>) =
        on.Size(data = data)
    static member size(data: Tensor<float32>) =
        on.Size(data = data)
    static member size(data: Tensor<double>) =
        on.Size(data = data)
    static member size(data: Tensor<string>) =
        on.Size(data = data)
    static member size(data: Tensor<bool>) =
        on.Size(data = data)
    static member size(data: Tensor<Complex>) =
        on.Size(data = data)
    static member xor(A: Tensor<bool>, B: Tensor<bool>) =
        on.Xor(A = A, B = B)
    static member adam(R: Tensor<float32>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<float32>[], ?alpha: float32, ?beta: float32, ?epsilon: float32, ?norm_coefficient: float32, ?norm_coefficient_post: float32) =
        on.Adam(R = R, T = T, inputs = inputs, ?alpha = alpha, ?beta = beta, ?epsilon = epsilon, ?norm_coefficient = norm_coefficient, ?norm_coefficient_post = norm_coefficient_post)
    static member adam(R: Tensor<float32>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<double>[], ?alpha: float32, ?beta: float32, ?epsilon: float32, ?norm_coefficient: float32, ?norm_coefficient_post: float32) =
        on.Adam(R = R, T = T, inputs = inputs, ?alpha = alpha, ?beta = beta, ?epsilon = epsilon, ?norm_coefficient = norm_coefficient, ?norm_coefficient_post = norm_coefficient_post)
    static member adam(R: Tensor<double>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<float32>[], ?alpha: float32, ?beta: float32, ?epsilon: float32, ?norm_coefficient: float32, ?norm_coefficient_post: float32) =
        on.Adam(R = R, T = T, inputs = inputs, ?alpha = alpha, ?beta = beta, ?epsilon = epsilon, ?norm_coefficient = norm_coefficient, ?norm_coefficient_post = norm_coefficient_post)
    static member adam(R: Tensor<double>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<double>[], ?alpha: float32, ?beta: float32, ?epsilon: float32, ?norm_coefficient: float32, ?norm_coefficient_post: float32) =
        on.Adam(R = R, T = T, inputs = inputs, ?alpha = alpha, ?beta = beta, ?epsilon = epsilon, ?norm_coefficient = norm_coefficient, ?norm_coefficient_post = norm_coefficient_post)
    static member adagrad(R: Tensor<float32>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<float32>[], ?decay_factor: float32, ?epsilon: float32, ?norm_coefficient: float32) =
        on.Adagrad(R = R, T = T, inputs = inputs, ?decay_factor = decay_factor, ?epsilon = epsilon, ?norm_coefficient = norm_coefficient)
    static member adagrad(R: Tensor<float32>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<double>[], ?decay_factor: float32, ?epsilon: float32, ?norm_coefficient: float32) =
        on.Adagrad(R = R, T = T, inputs = inputs, ?decay_factor = decay_factor, ?epsilon = epsilon, ?norm_coefficient = norm_coefficient)
    static member adagrad(R: Tensor<double>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<float32>[], ?decay_factor: float32, ?epsilon: float32, ?norm_coefficient: float32) =
        on.Adagrad(R = R, T = T, inputs = inputs, ?decay_factor = decay_factor, ?epsilon = epsilon, ?norm_coefficient = norm_coefficient)
    static member adagrad(R: Tensor<double>, T: Tensor<int64>, [<ParamArray>]inputs: Tensor<double>[], ?decay_factor: float32, ?epsilon: float32, ?norm_coefficient: float32) =
        on.Adagrad(R = R, T = T, inputs = inputs, ?decay_factor = decay_factor, ?epsilon = epsilon, ?norm_coefficient = norm_coefficient)
    static member momentum(R: Tensor<float32>, T: Tensor<int64>, alpha: float32, beta: float32, mode: string, norm_coefficient: float32, [<ParamArray>]inputs: Tensor<float32>[]) =
        on.Momentum(R = R, T = T, alpha = alpha, beta = beta, mode = mode, norm_coefficient = norm_coefficient, inputs = inputs)
    static member momentum(R: Tensor<float32>, T: Tensor<int64>, alpha: float32, beta: float32, mode: string, norm_coefficient: float32, [<ParamArray>]inputs: Tensor<double>[]) =
        on.Momentum(R = R, T = T, alpha = alpha, beta = beta, mode = mode, norm_coefficient = norm_coefficient, inputs = inputs)
    static member momentum(R: Tensor<double>, T: Tensor<int64>, alpha: float32, beta: float32, mode: string, norm_coefficient: float32, [<ParamArray>]inputs: Tensor<float32>[]) =
        on.Momentum(R = R, T = T, alpha = alpha, beta = beta, mode = mode, norm_coefficient = norm_coefficient, inputs = inputs)
    static member momentum(R: Tensor<double>, T: Tensor<int64>, alpha: float32, beta: float32, mode: string, norm_coefficient: float32, [<ParamArray>]inputs: Tensor<double>[]) =
        on.Momentum(R = R, T = T, alpha = alpha, beta = beta, mode = mode, norm_coefficient = norm_coefficient, inputs = inputs)
    static member grid_sample(X: Tensor<uint8>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<uint8>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<uint16>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<uint16>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<uint32>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<uint32>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<uint64>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<uint64>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<int8>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<int8>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<int16>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<int16>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<int>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<int>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<int64>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<int64>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<float32>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<float32>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<double>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<double>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<string>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<string>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<bool>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<bool>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<Complex>, grid: Tensor<float32>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member grid_sample(X: Tensor<Complex>, grid: Tensor<double>, ?align_corners: int64, ?mode: string, ?padding_mode: string) =
        on.GridSample(X = X, grid = grid, ?align_corners = align_corners, ?mode = mode, ?padding_mode = padding_mode)
    static member cast_like(input: Tensor<float32>, target_type: Tensor<float32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<float32>, target_type: Tensor<double>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<float32>, target_type: Tensor<int8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<float32>, target_type: Tensor<int16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<float32>, target_type: Tensor<int>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<float32>, target_type: Tensor<int64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<float32>, target_type: Tensor<uint8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<float32>, target_type: Tensor<uint16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<float32>, target_type: Tensor<uint32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<float32>, target_type: Tensor<uint64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<float32>, target_type: Tensor<bool>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<float32>, target_type: Tensor<string>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<double>, target_type: Tensor<float32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<double>, target_type: Tensor<double>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<double>, target_type: Tensor<int8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<double>, target_type: Tensor<int16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<double>, target_type: Tensor<int>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<double>, target_type: Tensor<int64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<double>, target_type: Tensor<uint8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<double>, target_type: Tensor<uint16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<double>, target_type: Tensor<uint32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<double>, target_type: Tensor<uint64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<double>, target_type: Tensor<bool>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<double>, target_type: Tensor<string>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int8>, target_type: Tensor<float32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int8>, target_type: Tensor<double>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int8>, target_type: Tensor<int8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int8>, target_type: Tensor<int16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int8>, target_type: Tensor<int>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int8>, target_type: Tensor<int64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int8>, target_type: Tensor<uint8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int8>, target_type: Tensor<uint16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int8>, target_type: Tensor<uint32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int8>, target_type: Tensor<uint64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int8>, target_type: Tensor<bool>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int8>, target_type: Tensor<string>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int16>, target_type: Tensor<float32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int16>, target_type: Tensor<double>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int16>, target_type: Tensor<int8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int16>, target_type: Tensor<int16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int16>, target_type: Tensor<int>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int16>, target_type: Tensor<int64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int16>, target_type: Tensor<uint8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int16>, target_type: Tensor<uint16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int16>, target_type: Tensor<uint32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int16>, target_type: Tensor<uint64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int16>, target_type: Tensor<bool>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int16>, target_type: Tensor<string>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int>, target_type: Tensor<float32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int>, target_type: Tensor<double>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int>, target_type: Tensor<int8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int>, target_type: Tensor<int16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int>, target_type: Tensor<int>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int>, target_type: Tensor<int64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int>, target_type: Tensor<uint8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int>, target_type: Tensor<uint16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int>, target_type: Tensor<uint32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int>, target_type: Tensor<uint64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int>, target_type: Tensor<bool>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int>, target_type: Tensor<string>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int64>, target_type: Tensor<float32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int64>, target_type: Tensor<double>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int64>, target_type: Tensor<int8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int64>, target_type: Tensor<int16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int64>, target_type: Tensor<int>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int64>, target_type: Tensor<int64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int64>, target_type: Tensor<uint8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int64>, target_type: Tensor<uint16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int64>, target_type: Tensor<uint32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int64>, target_type: Tensor<uint64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int64>, target_type: Tensor<bool>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<int64>, target_type: Tensor<string>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint8>, target_type: Tensor<float32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint8>, target_type: Tensor<double>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint8>, target_type: Tensor<int8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint8>, target_type: Tensor<int16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint8>, target_type: Tensor<int>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint8>, target_type: Tensor<int64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint8>, target_type: Tensor<uint8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint8>, target_type: Tensor<uint16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint8>, target_type: Tensor<uint32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint8>, target_type: Tensor<uint64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint8>, target_type: Tensor<bool>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint8>, target_type: Tensor<string>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint16>, target_type: Tensor<float32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint16>, target_type: Tensor<double>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint16>, target_type: Tensor<int8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint16>, target_type: Tensor<int16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint16>, target_type: Tensor<int>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint16>, target_type: Tensor<int64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint16>, target_type: Tensor<uint8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint16>, target_type: Tensor<uint16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint16>, target_type: Tensor<uint32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint16>, target_type: Tensor<uint64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint16>, target_type: Tensor<bool>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint16>, target_type: Tensor<string>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint32>, target_type: Tensor<float32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint32>, target_type: Tensor<double>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint32>, target_type: Tensor<int8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint32>, target_type: Tensor<int16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint32>, target_type: Tensor<int>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint32>, target_type: Tensor<int64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint32>, target_type: Tensor<uint8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint32>, target_type: Tensor<uint16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint32>, target_type: Tensor<uint32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint32>, target_type: Tensor<uint64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint32>, target_type: Tensor<bool>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint32>, target_type: Tensor<string>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint64>, target_type: Tensor<float32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint64>, target_type: Tensor<double>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint64>, target_type: Tensor<int8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint64>, target_type: Tensor<int16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint64>, target_type: Tensor<int>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint64>, target_type: Tensor<int64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint64>, target_type: Tensor<uint8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint64>, target_type: Tensor<uint16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint64>, target_type: Tensor<uint32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint64>, target_type: Tensor<uint64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint64>, target_type: Tensor<bool>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<uint64>, target_type: Tensor<string>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<bool>, target_type: Tensor<float32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<bool>, target_type: Tensor<double>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<bool>, target_type: Tensor<int8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<bool>, target_type: Tensor<int16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<bool>, target_type: Tensor<int>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<bool>, target_type: Tensor<int64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<bool>, target_type: Tensor<uint8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<bool>, target_type: Tensor<uint16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<bool>, target_type: Tensor<uint32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<bool>, target_type: Tensor<uint64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<bool>, target_type: Tensor<bool>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<bool>, target_type: Tensor<string>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<string>, target_type: Tensor<float32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<string>, target_type: Tensor<double>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<string>, target_type: Tensor<int8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<string>, target_type: Tensor<int16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<string>, target_type: Tensor<int>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<string>, target_type: Tensor<int64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<string>, target_type: Tensor<uint8>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<string>, target_type: Tensor<uint16>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<string>, target_type: Tensor<uint32>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<string>, target_type: Tensor<uint64>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<string>, target_type: Tensor<bool>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member cast_like(input: Tensor<string>, target_type: Tensor<string>, ?saturate: int64) =
        on.CastLike(input = input, target_type = target_type, ?saturate = saturate)
    static member softmax_cross_entropy_loss(scores: Tensor<float32>, labels: Tensor<int>, ?weights: Tensor<float32>, ?ignore_index: int64, ?reduction: string) =
        on.SoftmaxCrossEntropyLoss(scores = scores, labels = labels, ?weights = weights, ?ignore_index = ignore_index, ?reduction = reduction)
    static member softmax_cross_entropy_loss(scores: Tensor<float32>, labels: Tensor<int64>, ?weights: Tensor<float32>, ?ignore_index: int64, ?reduction: string) =
        on.SoftmaxCrossEntropyLoss(scores = scores, labels = labels, ?weights = weights, ?ignore_index = ignore_index, ?reduction = reduction)
    static member softmax_cross_entropy_loss(scores: Tensor<double>, labels: Tensor<int>, ?weights: Tensor<double>, ?ignore_index: int64, ?reduction: string) =
        on.SoftmaxCrossEntropyLoss(scores = scores, labels = labels, ?weights = weights, ?ignore_index = ignore_index, ?reduction = reduction)
    static member softmax_cross_entropy_loss(scores: Tensor<double>, labels: Tensor<int64>, ?weights: Tensor<double>, ?ignore_index: int64, ?reduction: string) =
        on.SoftmaxCrossEntropyLoss(scores = scores, labels = labels, ?weights = weights, ?ignore_index = ignore_index, ?reduction = reduction)
    static member gather_elements(data: Tensor<uint8>, indices: Tensor<int>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<uint8>, indices: Tensor<int64>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<uint16>, indices: Tensor<int>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<uint16>, indices: Tensor<int64>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<uint32>, indices: Tensor<int>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<uint32>, indices: Tensor<int64>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<uint64>, indices: Tensor<int>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<uint64>, indices: Tensor<int64>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<int8>, indices: Tensor<int>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<int8>, indices: Tensor<int64>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<int16>, indices: Tensor<int>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<int16>, indices: Tensor<int64>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<int>, indices: Tensor<int>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<int>, indices: Tensor<int64>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<int64>, indices: Tensor<int>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<int64>, indices: Tensor<int64>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<float32>, indices: Tensor<int>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<float32>, indices: Tensor<int64>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<double>, indices: Tensor<int>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<double>, indices: Tensor<int64>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<string>, indices: Tensor<int>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<string>, indices: Tensor<int64>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<bool>, indices: Tensor<int>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<bool>, indices: Tensor<int64>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<Complex>, indices: Tensor<int>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member gather_elements(data: Tensor<Complex>, indices: Tensor<int64>, ?axis: int64) =
        on.GatherElements(data = data, indices = indices, ?axis = axis)
    static member cum_sum(x: Tensor<uint32>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        on.CumSum(x = x, axis = axis, ?exclusive = exclusive, ?reverse = reverse)
    static member cum_sum(x: Tensor<uint32>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        on.CumSum(x = x, axis = axis, ?exclusive = exclusive, ?reverse = reverse)
    static member cum_sum(x: Tensor<uint64>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        on.CumSum(x = x, axis = axis, ?exclusive = exclusive, ?reverse = reverse)
    static member cum_sum(x: Tensor<uint64>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        on.CumSum(x = x, axis = axis, ?exclusive = exclusive, ?reverse = reverse)
    static member cum_sum(x: Tensor<int>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        on.CumSum(x = x, axis = axis, ?exclusive = exclusive, ?reverse = reverse)
    static member cum_sum(x: Tensor<int>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        on.CumSum(x = x, axis = axis, ?exclusive = exclusive, ?reverse = reverse)
    static member cum_sum(x: Tensor<int64>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        on.CumSum(x = x, axis = axis, ?exclusive = exclusive, ?reverse = reverse)
    static member cum_sum(x: Tensor<int64>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        on.CumSum(x = x, axis = axis, ?exclusive = exclusive, ?reverse = reverse)
    static member cum_sum(x: Tensor<float32>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        on.CumSum(x = x, axis = axis, ?exclusive = exclusive, ?reverse = reverse)
    static member cum_sum(x: Tensor<float32>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        on.CumSum(x = x, axis = axis, ?exclusive = exclusive, ?reverse = reverse)
    static member cum_sum(x: Tensor<double>, axis: Tensor<int>, ?exclusive: int64, ?reverse: int64) =
        on.CumSum(x = x, axis = axis, ?exclusive = exclusive, ?reverse = reverse)
    static member cum_sum(x: Tensor<double>, axis: Tensor<int64>, ?exclusive: int64, ?reverse: int64) =
        on.CumSum(x = x, axis = axis, ?exclusive = exclusive, ?reverse = reverse)
    static member roi_align(X: Tensor<float32>, rois: Tensor<float32>, batch_indices: Tensor<int64>, ?coordinate_transformation_mode: string, ?mode: string, ?output_height: int64, ?output_width: int64, ?sampling_ratio: int64, ?spatial_scale: float32) =
        on.RoiAlign(X = X, rois = rois, batch_indices = batch_indices, ?coordinate_transformation_mode = coordinate_transformation_mode, ?mode = mode, ?output_height = output_height, ?output_width = output_width, ?sampling_ratio = sampling_ratio, ?spatial_scale = spatial_scale)
    static member roi_align(X: Tensor<double>, rois: Tensor<double>, batch_indices: Tensor<int64>, ?coordinate_transformation_mode: string, ?mode: string, ?output_height: int64, ?output_width: int64, ?sampling_ratio: int64, ?spatial_scale: float32) =
        on.RoiAlign(X = X, rois = rois, batch_indices = batch_indices, ?coordinate_transformation_mode = coordinate_transformation_mode, ?mode = mode, ?output_height = output_height, ?output_width = output_width, ?sampling_ratio = sampling_ratio, ?spatial_scale = spatial_scale)
    static member dequantize_linear(x: Tensor<int8>, x_scale: Tensor<float32>, ?x_zero_point: Tensor<int8>, ?axis: int64) =
        on.DequantizeLinear(x = x, x_scale = x_scale, ?x_zero_point = x_zero_point, ?axis = axis)
    static member dequantize_linear(x: Tensor<uint8>, x_scale: Tensor<float32>, ?x_zero_point: Tensor<uint8>, ?axis: int64) =
        on.DequantizeLinear(x = x, x_scale = x_scale, ?x_zero_point = x_zero_point, ?axis = axis)
    static member dequantize_linear(x: Tensor<int>, x_scale: Tensor<float32>, ?x_zero_point: Tensor<int>, ?axis: int64) =
        on.DequantizeLinear(x = x, x_scale = x_scale, ?x_zero_point = x_zero_point, ?axis = axis)
    static member dft(input: Tensor<float32>, ?dft_length: Tensor<int>, ?axis: int64, ?inverse: int64, ?onesided: int64) =
        on.DFT(input = input, ?dft_length = dft_length, ?axis = axis, ?inverse = inverse, ?onesided = onesided)
    static member dft(input: Tensor<float32>, ?dft_length: Tensor<int64>, ?axis: int64, ?inverse: int64, ?onesided: int64) =
        on.DFT(input = input, ?dft_length = dft_length, ?axis = axis, ?inverse = inverse, ?onesided = onesided)
    static member dft(input: Tensor<double>, ?dft_length: Tensor<int>, ?axis: int64, ?inverse: int64, ?onesided: int64) =
        on.DFT(input = input, ?dft_length = dft_length, ?axis = axis, ?inverse = inverse, ?onesided = onesided)
    static member dft(input: Tensor<double>, ?dft_length: Tensor<int64>, ?axis: int64, ?inverse: int64, ?onesided: int64) =
        on.DFT(input = input, ?dft_length = dft_length, ?axis = axis, ?inverse = inverse, ?onesided = onesided)
    static member quantize_linear(x: Tensor<float32>, y_scale: Tensor<float32>, ?y_zero_point: Tensor<int8>, ?axis: int64, ?saturate: int64) =
        on.QuantizeLinear(x = x, y_scale = y_scale, ?y_zero_point = y_zero_point, ?axis = axis, ?saturate = saturate)
    static member quantize_linear(x: Tensor<float32>, y_scale: Tensor<float32>, ?y_zero_point: Tensor<uint8>, ?axis: int64, ?saturate: int64) =
        on.QuantizeLinear(x = x, y_scale = y_scale, ?y_zero_point = y_zero_point, ?axis = axis, ?saturate = saturate)
    static member quantize_linear(x: Tensor<int>, y_scale: Tensor<int>, ?y_zero_point: Tensor<int8>, ?axis: int64, ?saturate: int64) =
        on.QuantizeLinear(x = x, y_scale = y_scale, ?y_zero_point = y_zero_point, ?axis = axis, ?saturate = saturate)
    static member quantize_linear(x: Tensor<int>, y_scale: Tensor<int>, ?y_zero_point: Tensor<uint8>, ?axis: int64, ?saturate: int64) =
        on.QuantizeLinear(x = x, y_scale = y_scale, ?y_zero_point = y_zero_point, ?axis = axis, ?saturate = saturate)
    static member q_linear_conv(x: Tensor<int8>, x_scale: Tensor<float32>, x_zero_point: Tensor<int8>, w: Tensor<int8>, w_scale: Tensor<float32>, w_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.QLinearConv(x = x, x_scale = x_scale, x_zero_point = x_zero_point, w = w, w_scale = w_scale, w_zero_point = w_zero_point, y_scale = y_scale, y_zero_point = y_zero_point, ?B = B, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member q_linear_conv(x: Tensor<int8>, x_scale: Tensor<float32>, x_zero_point: Tensor<int8>, w: Tensor<int8>, w_scale: Tensor<float32>, w_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.QLinearConv(x = x, x_scale = x_scale, x_zero_point = x_zero_point, w = w, w_scale = w_scale, w_zero_point = w_zero_point, y_scale = y_scale, y_zero_point = y_zero_point, ?B = B, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member q_linear_conv(x: Tensor<int8>, x_scale: Tensor<float32>, x_zero_point: Tensor<int8>, w: Tensor<uint8>, w_scale: Tensor<float32>, w_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.QLinearConv(x = x, x_scale = x_scale, x_zero_point = x_zero_point, w = w, w_scale = w_scale, w_zero_point = w_zero_point, y_scale = y_scale, y_zero_point = y_zero_point, ?B = B, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member q_linear_conv(x: Tensor<int8>, x_scale: Tensor<float32>, x_zero_point: Tensor<int8>, w: Tensor<uint8>, w_scale: Tensor<float32>, w_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.QLinearConv(x = x, x_scale = x_scale, x_zero_point = x_zero_point, w = w, w_scale = w_scale, w_zero_point = w_zero_point, y_scale = y_scale, y_zero_point = y_zero_point, ?B = B, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member q_linear_conv(x: Tensor<uint8>, x_scale: Tensor<float32>, x_zero_point: Tensor<uint8>, w: Tensor<int8>, w_scale: Tensor<float32>, w_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.QLinearConv(x = x, x_scale = x_scale, x_zero_point = x_zero_point, w = w, w_scale = w_scale, w_zero_point = w_zero_point, y_scale = y_scale, y_zero_point = y_zero_point, ?B = B, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member q_linear_conv(x: Tensor<uint8>, x_scale: Tensor<float32>, x_zero_point: Tensor<uint8>, w: Tensor<int8>, w_scale: Tensor<float32>, w_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.QLinearConv(x = x, x_scale = x_scale, x_zero_point = x_zero_point, w = w, w_scale = w_scale, w_zero_point = w_zero_point, y_scale = y_scale, y_zero_point = y_zero_point, ?B = B, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member q_linear_conv(x: Tensor<uint8>, x_scale: Tensor<float32>, x_zero_point: Tensor<uint8>, w: Tensor<uint8>, w_scale: Tensor<float32>, w_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.QLinearConv(x = x, x_scale = x_scale, x_zero_point = x_zero_point, w = w, w_scale = w_scale, w_zero_point = w_zero_point, y_scale = y_scale, y_zero_point = y_zero_point, ?B = B, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member q_linear_conv(x: Tensor<uint8>, x_scale: Tensor<float32>, x_zero_point: Tensor<uint8>, w: Tensor<uint8>, w_scale: Tensor<float32>, w_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>, ?B: Tensor<int>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.QLinearConv(x = x, x_scale = x_scale, x_zero_point = x_zero_point, w = w, w_scale = w_scale, w_zero_point = w_zero_point, y_scale = y_scale, y_zero_point = y_zero_point, ?B = B, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member conv_integer(x: Tensor<int8>, w: Tensor<int8>, ?x_zero_point: Tensor<int8>, ?w_zero_point: Tensor<int8>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.ConvInteger(x = x, w = w, ?x_zero_point = x_zero_point, ?w_zero_point = w_zero_point, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member conv_integer(x: Tensor<int8>, w: Tensor<uint8>, ?x_zero_point: Tensor<int8>, ?w_zero_point: Tensor<uint8>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.ConvInteger(x = x, w = w, ?x_zero_point = x_zero_point, ?w_zero_point = w_zero_point, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member conv_integer(x: Tensor<uint8>, w: Tensor<int8>, ?x_zero_point: Tensor<uint8>, ?w_zero_point: Tensor<int8>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.ConvInteger(x = x, w = w, ?x_zero_point = x_zero_point, ?w_zero_point = w_zero_point, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member conv_integer(x: Tensor<uint8>, w: Tensor<uint8>, ?x_zero_point: Tensor<uint8>, ?w_zero_point: Tensor<uint8>, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        on.ConvInteger(x = x, w = w, ?x_zero_point = x_zero_point, ?w_zero_point = w_zero_point, ?auto_pad = auto_pad, ?dilations = dilations, ?group = group, ?kernel_shape = kernel_shape, ?pads = pads, ?strides = strides)
    static member q_linear_mat_mul(a: Tensor<int8>, a_scale: Tensor<float32>, a_zero_point: Tensor<int8>, b: Tensor<int8>, b_scale: Tensor<float32>, b_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>) =
        on.QLinearMatMul(a = a, a_scale = a_scale, a_zero_point = a_zero_point, b = b, b_scale = b_scale, b_zero_point = b_zero_point, y_scale = y_scale, y_zero_point = y_zero_point)
    static member q_linear_mat_mul(a: Tensor<int8>, a_scale: Tensor<float32>, a_zero_point: Tensor<int8>, b: Tensor<int8>, b_scale: Tensor<float32>, b_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>) =
        on.QLinearMatMul(a = a, a_scale = a_scale, a_zero_point = a_zero_point, b = b, b_scale = b_scale, b_zero_point = b_zero_point, y_scale = y_scale, y_zero_point = y_zero_point)
    static member q_linear_mat_mul(a: Tensor<int8>, a_scale: Tensor<float32>, a_zero_point: Tensor<int8>, b: Tensor<uint8>, b_scale: Tensor<float32>, b_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>) =
        on.QLinearMatMul(a = a, a_scale = a_scale, a_zero_point = a_zero_point, b = b, b_scale = b_scale, b_zero_point = b_zero_point, y_scale = y_scale, y_zero_point = y_zero_point)
    static member q_linear_mat_mul(a: Tensor<int8>, a_scale: Tensor<float32>, a_zero_point: Tensor<int8>, b: Tensor<uint8>, b_scale: Tensor<float32>, b_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>) =
        on.QLinearMatMul(a = a, a_scale = a_scale, a_zero_point = a_zero_point, b = b, b_scale = b_scale, b_zero_point = b_zero_point, y_scale = y_scale, y_zero_point = y_zero_point)
    static member q_linear_mat_mul(a: Tensor<uint8>, a_scale: Tensor<float32>, a_zero_point: Tensor<uint8>, b: Tensor<int8>, b_scale: Tensor<float32>, b_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>) =
        on.QLinearMatMul(a = a, a_scale = a_scale, a_zero_point = a_zero_point, b = b, b_scale = b_scale, b_zero_point = b_zero_point, y_scale = y_scale, y_zero_point = y_zero_point)
    static member q_linear_mat_mul(a: Tensor<uint8>, a_scale: Tensor<float32>, a_zero_point: Tensor<uint8>, b: Tensor<int8>, b_scale: Tensor<float32>, b_zero_point: Tensor<int8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>) =
        on.QLinearMatMul(a = a, a_scale = a_scale, a_zero_point = a_zero_point, b = b, b_scale = b_scale, b_zero_point = b_zero_point, y_scale = y_scale, y_zero_point = y_zero_point)
    static member q_linear_mat_mul(a: Tensor<uint8>, a_scale: Tensor<float32>, a_zero_point: Tensor<uint8>, b: Tensor<uint8>, b_scale: Tensor<float32>, b_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<int8>) =
        on.QLinearMatMul(a = a, a_scale = a_scale, a_zero_point = a_zero_point, b = b, b_scale = b_scale, b_zero_point = b_zero_point, y_scale = y_scale, y_zero_point = y_zero_point)
    static member q_linear_mat_mul(a: Tensor<uint8>, a_scale: Tensor<float32>, a_zero_point: Tensor<uint8>, b: Tensor<uint8>, b_scale: Tensor<float32>, b_zero_point: Tensor<uint8>, y_scale: Tensor<float32>, y_zero_point: Tensor<uint8>) =
        on.QLinearMatMul(a = a, a_scale = a_scale, a_zero_point = a_zero_point, b = b, b_scale = b_scale, b_zero_point = b_zero_point, y_scale = y_scale, y_zero_point = y_zero_point)
    static member mat_mul_integer(A: Tensor<int8>, B: Tensor<int8>, ?a_zero_point: Tensor<int8>, ?b_zero_point: Tensor<int8>) =
        on.MatMulInteger(A = A, B = B, ?a_zero_point = a_zero_point, ?b_zero_point = b_zero_point)
    static member mat_mul_integer(A: Tensor<int8>, B: Tensor<uint8>, ?a_zero_point: Tensor<int8>, ?b_zero_point: Tensor<uint8>) =
        on.MatMulInteger(A = A, B = B, ?a_zero_point = a_zero_point, ?b_zero_point = b_zero_point)
    static member mat_mul_integer(A: Tensor<uint8>, B: Tensor<int8>, ?a_zero_point: Tensor<uint8>, ?b_zero_point: Tensor<int8>) =
        on.MatMulInteger(A = A, B = B, ?a_zero_point = a_zero_point, ?b_zero_point = b_zero_point)
    static member mat_mul_integer(A: Tensor<uint8>, B: Tensor<uint8>, ?a_zero_point: Tensor<uint8>, ?b_zero_point: Tensor<uint8>) =
        on.MatMulInteger(A = A, B = B, ?a_zero_point = a_zero_point, ?b_zero_point = b_zero_point)
    static member pad(data: Tensor<uint8>, pads: Tensor<int64>, ?constant_value: Tensor<uint8>, ?axes: Tensor<int>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<uint8>, pads: Tensor<int64>, ?constant_value: Tensor<uint8>, ?axes: Tensor<int64>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<uint16>, pads: Tensor<int64>, ?constant_value: Tensor<uint16>, ?axes: Tensor<int>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<uint16>, pads: Tensor<int64>, ?constant_value: Tensor<uint16>, ?axes: Tensor<int64>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<uint32>, pads: Tensor<int64>, ?constant_value: Tensor<uint32>, ?axes: Tensor<int>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<uint32>, pads: Tensor<int64>, ?constant_value: Tensor<uint32>, ?axes: Tensor<int64>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<uint64>, pads: Tensor<int64>, ?constant_value: Tensor<uint64>, ?axes: Tensor<int>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<uint64>, pads: Tensor<int64>, ?constant_value: Tensor<uint64>, ?axes: Tensor<int64>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<int8>, pads: Tensor<int64>, ?constant_value: Tensor<int8>, ?axes: Tensor<int>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<int8>, pads: Tensor<int64>, ?constant_value: Tensor<int8>, ?axes: Tensor<int64>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<int16>, pads: Tensor<int64>, ?constant_value: Tensor<int16>, ?axes: Tensor<int>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<int16>, pads: Tensor<int64>, ?constant_value: Tensor<int16>, ?axes: Tensor<int64>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<int>, pads: Tensor<int64>, ?constant_value: Tensor<int>, ?axes: Tensor<int>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<int>, pads: Tensor<int64>, ?constant_value: Tensor<int>, ?axes: Tensor<int64>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<int64>, pads: Tensor<int64>, ?constant_value: Tensor<int64>, ?axes: Tensor<int>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<int64>, pads: Tensor<int64>, ?constant_value: Tensor<int64>, ?axes: Tensor<int64>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<float32>, pads: Tensor<int64>, ?constant_value: Tensor<float32>, ?axes: Tensor<int>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<float32>, pads: Tensor<int64>, ?constant_value: Tensor<float32>, ?axes: Tensor<int64>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<double>, pads: Tensor<int64>, ?constant_value: Tensor<double>, ?axes: Tensor<int>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<double>, pads: Tensor<int64>, ?constant_value: Tensor<double>, ?axes: Tensor<int64>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<string>, pads: Tensor<int64>, ?constant_value: Tensor<string>, ?axes: Tensor<int>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<string>, pads: Tensor<int64>, ?constant_value: Tensor<string>, ?axes: Tensor<int64>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<bool>, pads: Tensor<int64>, ?constant_value: Tensor<bool>, ?axes: Tensor<int>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<bool>, pads: Tensor<int64>, ?constant_value: Tensor<bool>, ?axes: Tensor<int64>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<Complex>, pads: Tensor<int64>, ?constant_value: Tensor<Complex>, ?axes: Tensor<int>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member pad(data: Tensor<Complex>, pads: Tensor<int64>, ?constant_value: Tensor<Complex>, ?axes: Tensor<int64>, ?mode: string) =
        on.Pad(data = data, pads = pads, ?constant_value = constant_value, ?axes = axes, ?mode = mode)
    static member max_unpool(X: Tensor<float32>, I: Tensor<int64>, kernel_shape: int64[], ?output_shape: Tensor<int64>, ?pads: int64[], ?strides: int64[]) =
        on.MaxUnpool(X = X, I = I, kernel_shape = kernel_shape, ?output_shape = output_shape, ?pads = pads, ?strides = strides)
    static member max_unpool(X: Tensor<double>, I: Tensor<int64>, kernel_shape: int64[], ?output_shape: Tensor<int64>, ?pads: int64[], ?strides: int64[]) =
        on.MaxUnpool(X = X, I = I, kernel_shape = kernel_shape, ?output_shape = output_shape, ?pads = pads, ?strides = strides)
    static member slice(data: Tensor<uint8>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<uint8>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<uint16>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<uint16>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<uint32>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<uint32>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<uint64>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<uint64>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<int8>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<int8>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<int16>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<int16>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<int>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<int>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<int64>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<int64>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<float32>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<float32>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<double>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<double>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<string>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<string>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<bool>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<bool>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<Complex>, starts: Tensor<int>, ends: Tensor<int>, ?axes: Tensor<int>, ?steps: Tensor<int>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member slice(data: Tensor<Complex>, starts: Tensor<int64>, ends: Tensor<int64>, ?axes: Tensor<int64>, ?steps: Tensor<int64>) =
        on.Slice(data = data, starts = starts, ends = ends, ?axes = axes, ?steps = steps)
    static member negative_log_likelihood_loss(input: Tensor<float32>, target: Tensor<int>, ?weight: Tensor<float32>, ?ignore_index: int64, ?reduction: string) =
        on.NegativeLogLikelihoodLoss(input = input, target = target, ?weight = weight, ?ignore_index = ignore_index, ?reduction = reduction)
    static member negative_log_likelihood_loss(input: Tensor<float32>, target: Tensor<int64>, ?weight: Tensor<float32>, ?ignore_index: int64, ?reduction: string) =
        on.NegativeLogLikelihoodLoss(input = input, target = target, ?weight = weight, ?ignore_index = ignore_index, ?reduction = reduction)
    static member negative_log_likelihood_loss(input: Tensor<double>, target: Tensor<int>, ?weight: Tensor<double>, ?ignore_index: int64, ?reduction: string) =
        on.NegativeLogLikelihoodLoss(input = input, target = target, ?weight = weight, ?ignore_index = ignore_index, ?reduction = reduction)
    static member negative_log_likelihood_loss(input: Tensor<double>, target: Tensor<int64>, ?weight: Tensor<double>, ?ignore_index: int64, ?reduction: string) =
        on.NegativeLogLikelihoodLoss(input = input, target = target, ?weight = weight, ?ignore_index = ignore_index, ?reduction = reduction)
    static member gather(data: Tensor<uint8>, indices: Tensor<int>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<uint8>, indices: Tensor<int64>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<uint16>, indices: Tensor<int>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<uint16>, indices: Tensor<int64>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<uint32>, indices: Tensor<int>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<uint32>, indices: Tensor<int64>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<uint64>, indices: Tensor<int>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<uint64>, indices: Tensor<int64>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<int8>, indices: Tensor<int>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<int8>, indices: Tensor<int64>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<int16>, indices: Tensor<int>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<int16>, indices: Tensor<int64>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<int>, indices: Tensor<int>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<int>, indices: Tensor<int64>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<int64>, indices: Tensor<int>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<int64>, indices: Tensor<int64>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<float32>, indices: Tensor<int>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<float32>, indices: Tensor<int64>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<double>, indices: Tensor<int>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<double>, indices: Tensor<int64>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<string>, indices: Tensor<int>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<string>, indices: Tensor<int64>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<bool>, indices: Tensor<int>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<bool>, indices: Tensor<int64>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<Complex>, indices: Tensor<int>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member gather(data: Tensor<Complex>, indices: Tensor<int64>, ?axis: int64) =
        on.Gather(data = data, indices = indices, ?axis = axis)
    static member scatter_elements(data: Tensor<uint8>, indices: Tensor<int>, updates: Tensor<uint8>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<uint8>, indices: Tensor<int64>, updates: Tensor<uint8>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<uint16>, indices: Tensor<int>, updates: Tensor<uint16>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<uint16>, indices: Tensor<int64>, updates: Tensor<uint16>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<uint32>, indices: Tensor<int>, updates: Tensor<uint32>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<uint32>, indices: Tensor<int64>, updates: Tensor<uint32>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<uint64>, indices: Tensor<int>, updates: Tensor<uint64>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<uint64>, indices: Tensor<int64>, updates: Tensor<uint64>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<int8>, indices: Tensor<int>, updates: Tensor<int8>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<int8>, indices: Tensor<int64>, updates: Tensor<int8>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<int16>, indices: Tensor<int>, updates: Tensor<int16>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<int16>, indices: Tensor<int64>, updates: Tensor<int16>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<int>, indices: Tensor<int>, updates: Tensor<int>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<int>, indices: Tensor<int64>, updates: Tensor<int>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<int64>, indices: Tensor<int>, updates: Tensor<int64>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<int64>, indices: Tensor<int64>, updates: Tensor<int64>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<float32>, indices: Tensor<int>, updates: Tensor<float32>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<float32>, indices: Tensor<int64>, updates: Tensor<float32>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<double>, indices: Tensor<int>, updates: Tensor<double>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<double>, indices: Tensor<int64>, updates: Tensor<double>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<string>, indices: Tensor<int>, updates: Tensor<string>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<string>, indices: Tensor<int64>, updates: Tensor<string>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<bool>, indices: Tensor<int>, updates: Tensor<bool>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<bool>, indices: Tensor<int64>, updates: Tensor<bool>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<Complex>, indices: Tensor<int>, updates: Tensor<Complex>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member scatter_elements(data: Tensor<Complex>, indices: Tensor<int64>, updates: Tensor<Complex>, ?axis: int64, ?reduction: string) =
        on.ScatterElements(data = data, indices = indices, updates = updates, ?axis = axis, ?reduction = reduction)
    static member stft(signal: Tensor<float32>, frame_step: Tensor<int>, ?window: Tensor<float32>, ?frame_length: Tensor<int>, ?onesided: int64) =
        on.STFT(signal = signal, frame_step = frame_step, ?window = window, ?frame_length = frame_length, ?onesided = onesided)
    static member stft(signal: Tensor<float32>, frame_step: Tensor<int64>, ?window: Tensor<float32>, ?frame_length: Tensor<int64>, ?onesided: int64) =
        on.STFT(signal = signal, frame_step = frame_step, ?window = window, ?frame_length = frame_length, ?onesided = onesided)
    static member stft(signal: Tensor<double>, frame_step: Tensor<int>, ?window: Tensor<double>, ?frame_length: Tensor<int>, ?onesided: int64) =
        on.STFT(signal = signal, frame_step = frame_step, ?window = window, ?frame_length = frame_length, ?onesided = onesided)
    static member stft(signal: Tensor<double>, frame_step: Tensor<int64>, ?window: Tensor<double>, ?frame_length: Tensor<int64>, ?onesided: int64) =
        on.STFT(signal = signal, frame_step = frame_step, ?window = window, ?frame_length = frame_length, ?onesided = onesided)
    static member tile(input: Tensor<uint8>, repeats: Tensor<int64>) =
        on.Tile(input = input, repeats = repeats)
    static member tile(input: Tensor<uint16>, repeats: Tensor<int64>) =
        on.Tile(input = input, repeats = repeats)
    static member tile(input: Tensor<uint32>, repeats: Tensor<int64>) =
        on.Tile(input = input, repeats = repeats)
    static member tile(input: Tensor<uint64>, repeats: Tensor<int64>) =
        on.Tile(input = input, repeats = repeats)
    static member tile(input: Tensor<int8>, repeats: Tensor<int64>) =
        on.Tile(input = input, repeats = repeats)
    static member tile(input: Tensor<int16>, repeats: Tensor<int64>) =
        on.Tile(input = input, repeats = repeats)
    static member tile(input: Tensor<int>, repeats: Tensor<int64>) =
        on.Tile(input = input, repeats = repeats)
    static member tile(input: Tensor<int64>, repeats: Tensor<int64>) =
        on.Tile(input = input, repeats = repeats)
    static member tile(input: Tensor<float32>, repeats: Tensor<int64>) =
        on.Tile(input = input, repeats = repeats)
    static member tile(input: Tensor<double>, repeats: Tensor<int64>) =
        on.Tile(input = input, repeats = repeats)
    static member tile(input: Tensor<string>, repeats: Tensor<int64>) =
        on.Tile(input = input, repeats = repeats)
    static member tile(input: Tensor<bool>, repeats: Tensor<int64>) =
        on.Tile(input = input, repeats = repeats)
    static member tile(input: Tensor<Complex>, repeats: Tensor<int64>) =
        on.Tile(input = input, repeats = repeats)
    static member pow(X: Tensor<int>, Y: Tensor<uint8>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int>, Y: Tensor<uint16>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int>, Y: Tensor<uint32>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int>, Y: Tensor<uint64>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int>, Y: Tensor<int8>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int>, Y: Tensor<int16>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int>, Y: Tensor<int>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int>, Y: Tensor<int64>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int>, Y: Tensor<float32>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int>, Y: Tensor<double>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int64>, Y: Tensor<uint8>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int64>, Y: Tensor<uint16>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int64>, Y: Tensor<uint32>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int64>, Y: Tensor<uint64>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int64>, Y: Tensor<int8>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int64>, Y: Tensor<int16>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int64>, Y: Tensor<int>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int64>, Y: Tensor<int64>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int64>, Y: Tensor<float32>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<int64>, Y: Tensor<double>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<float32>, Y: Tensor<uint8>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<float32>, Y: Tensor<uint16>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<float32>, Y: Tensor<uint32>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<float32>, Y: Tensor<uint64>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<float32>, Y: Tensor<int8>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<float32>, Y: Tensor<int16>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<float32>, Y: Tensor<int>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<float32>, Y: Tensor<int64>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<float32>, Y: Tensor<float32>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<float32>, Y: Tensor<double>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<double>, Y: Tensor<uint8>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<double>, Y: Tensor<uint16>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<double>, Y: Tensor<uint32>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<double>, Y: Tensor<uint64>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<double>, Y: Tensor<int8>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<double>, Y: Tensor<int16>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<double>, Y: Tensor<int>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<double>, Y: Tensor<int64>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<double>, Y: Tensor<float32>) =
        on.Pow(X = X, Y = Y)
    static member pow(X: Tensor<double>, Y: Tensor<double>) =
        on.Pow(X = X, Y = Y)
    static member compress(input: Tensor<uint8>, condition: Tensor<bool>, ?axis: int64) =
        on.Compress(input = input, condition = condition, ?axis = axis)
    static member compress(input: Tensor<uint16>, condition: Tensor<bool>, ?axis: int64) =
        on.Compress(input = input, condition = condition, ?axis = axis)
    static member compress(input: Tensor<uint32>, condition: Tensor<bool>, ?axis: int64) =
        on.Compress(input = input, condition = condition, ?axis = axis)
    static member compress(input: Tensor<uint64>, condition: Tensor<bool>, ?axis: int64) =
        on.Compress(input = input, condition = condition, ?axis = axis)
    static member compress(input: Tensor<int8>, condition: Tensor<bool>, ?axis: int64) =
        on.Compress(input = input, condition = condition, ?axis = axis)
    static member compress(input: Tensor<int16>, condition: Tensor<bool>, ?axis: int64) =
        on.Compress(input = input, condition = condition, ?axis = axis)
    static member compress(input: Tensor<int>, condition: Tensor<bool>, ?axis: int64) =
        on.Compress(input = input, condition = condition, ?axis = axis)
    static member compress(input: Tensor<int64>, condition: Tensor<bool>, ?axis: int64) =
        on.Compress(input = input, condition = condition, ?axis = axis)
    static member compress(input: Tensor<float32>, condition: Tensor<bool>, ?axis: int64) =
        on.Compress(input = input, condition = condition, ?axis = axis)
    static member compress(input: Tensor<double>, condition: Tensor<bool>, ?axis: int64) =
        on.Compress(input = input, condition = condition, ?axis = axis)
    static member compress(input: Tensor<string>, condition: Tensor<bool>, ?axis: int64) =
        on.Compress(input = input, condition = condition, ?axis = axis)
    static member compress(input: Tensor<bool>, condition: Tensor<bool>, ?axis: int64) =
        on.Compress(input = input, condition = condition, ?axis = axis)
    static member compress(input: Tensor<Complex>, condition: Tensor<bool>, ?axis: int64) =
        on.Compress(input = input, condition = condition, ?axis = axis)
    static member scatter(data: Tensor<uint8>, indices: Tensor<int>, updates: Tensor<uint8>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<uint8>, indices: Tensor<int64>, updates: Tensor<uint8>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<uint16>, indices: Tensor<int>, updates: Tensor<uint16>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<uint16>, indices: Tensor<int64>, updates: Tensor<uint16>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<uint32>, indices: Tensor<int>, updates: Tensor<uint32>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<uint32>, indices: Tensor<int64>, updates: Tensor<uint32>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<uint64>, indices: Tensor<int>, updates: Tensor<uint64>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<uint64>, indices: Tensor<int64>, updates: Tensor<uint64>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<int8>, indices: Tensor<int>, updates: Tensor<int8>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<int8>, indices: Tensor<int64>, updates: Tensor<int8>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<int16>, indices: Tensor<int>, updates: Tensor<int16>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<int16>, indices: Tensor<int64>, updates: Tensor<int16>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<int>, indices: Tensor<int>, updates: Tensor<int>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<int>, indices: Tensor<int64>, updates: Tensor<int>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<int64>, indices: Tensor<int>, updates: Tensor<int64>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<int64>, indices: Tensor<int64>, updates: Tensor<int64>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<float32>, indices: Tensor<int>, updates: Tensor<float32>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<float32>, indices: Tensor<int64>, updates: Tensor<float32>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<double>, indices: Tensor<int>, updates: Tensor<double>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<double>, indices: Tensor<int64>, updates: Tensor<double>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<string>, indices: Tensor<int>, updates: Tensor<string>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<string>, indices: Tensor<int64>, updates: Tensor<string>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<bool>, indices: Tensor<int>, updates: Tensor<bool>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<bool>, indices: Tensor<int64>, updates: Tensor<bool>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<Complex>, indices: Tensor<int>, updates: Tensor<Complex>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member scatter(data: Tensor<Complex>, indices: Tensor<int64>, updates: Tensor<Complex>, ?axis: int64) =
        on.Scatter(data = data, indices = indices, updates = updates, ?axis = axis)
    static member center_crop_pad(input_data: Tensor<uint8>, shape: Tensor<int>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<uint8>, shape: Tensor<int64>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<uint16>, shape: Tensor<int>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<uint16>, shape: Tensor<int64>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<uint32>, shape: Tensor<int>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<uint32>, shape: Tensor<int64>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<uint64>, shape: Tensor<int>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<uint64>, shape: Tensor<int64>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<int8>, shape: Tensor<int>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<int8>, shape: Tensor<int64>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<int16>, shape: Tensor<int>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<int16>, shape: Tensor<int64>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<int>, shape: Tensor<int>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<int>, shape: Tensor<int64>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<int64>, shape: Tensor<int>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<int64>, shape: Tensor<int64>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<float32>, shape: Tensor<int>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<float32>, shape: Tensor<int64>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<double>, shape: Tensor<int>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<double>, shape: Tensor<int64>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<string>, shape: Tensor<int>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<string>, shape: Tensor<int64>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<bool>, shape: Tensor<int>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<bool>, shape: Tensor<int64>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<Complex>, shape: Tensor<int>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member center_crop_pad(input_data: Tensor<Complex>, shape: Tensor<int64>, ?axes: int64[]) =
        on.CenterCropPad(input_data = input_data, shape = shape, ?axes = axes)
    static member where(condition: Tensor<bool>, X: Tensor<uint8>, Y: Tensor<uint8>) =
        on.Where(condition = condition, X = X, Y = Y)
    static member where(condition: Tensor<bool>, X: Tensor<uint16>, Y: Tensor<uint16>) =
        on.Where(condition = condition, X = X, Y = Y)
    static member where(condition: Tensor<bool>, X: Tensor<uint32>, Y: Tensor<uint32>) =
        on.Where(condition = condition, X = X, Y = Y)
    static member where(condition: Tensor<bool>, X: Tensor<uint64>, Y: Tensor<uint64>) =
        on.Where(condition = condition, X = X, Y = Y)
    static member where(condition: Tensor<bool>, X: Tensor<int8>, Y: Tensor<int8>) =
        on.Where(condition = condition, X = X, Y = Y)
    static member where(condition: Tensor<bool>, X: Tensor<int16>, Y: Tensor<int16>) =
        on.Where(condition = condition, X = X, Y = Y)
    static member where(condition: Tensor<bool>, X: Tensor<int>, Y: Tensor<int>) =
        on.Where(condition = condition, X = X, Y = Y)
    static member where(condition: Tensor<bool>, X: Tensor<int64>, Y: Tensor<int64>) =
        on.Where(condition = condition, X = X, Y = Y)
    static member where(condition: Tensor<bool>, X: Tensor<float32>, Y: Tensor<float32>) =
        on.Where(condition = condition, X = X, Y = Y)
    static member where(condition: Tensor<bool>, X: Tensor<double>, Y: Tensor<double>) =
        on.Where(condition = condition, X = X, Y = Y)
    static member where(condition: Tensor<bool>, X: Tensor<string>, Y: Tensor<string>) =
        on.Where(condition = condition, X = X, Y = Y)
    static member where(condition: Tensor<bool>, X: Tensor<bool>, Y: Tensor<bool>) =
        on.Where(condition = condition, X = X, Y = Y)
    static member where(condition: Tensor<bool>, X: Tensor<Complex>, Y: Tensor<Complex>) =
        on.Where(condition = condition, X = X, Y = Y)
    static member resize(X: Tensor<uint8>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<uint8>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<uint16>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<uint16>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<uint32>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<uint32>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<uint64>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<uint64>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<int8>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<int8>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<int16>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<int16>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<int>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<int>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<int64>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<int64>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<float32>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<float32>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<double>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<double>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<string>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<string>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<bool>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<bool>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<Complex>, ?roi: Tensor<float32>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member resize(X: Tensor<Complex>, ?roi: Tensor<double>, ?scales: Tensor<float32>, ?sizes: Tensor<int64>, ?antialias: int64, ?axes: int64[], ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?keep_aspect_ratio_policy: string, ?mode: string, ?nearest_mode: string) =
        on.Resize(X = X, ?roi = roi, ?scales = scales, ?sizes = sizes, ?antialias = antialias, ?axes = axes, ?coordinate_transformation_mode = coordinate_transformation_mode, ?cubic_coeff_a = cubic_coeff_a, ?exclude_outside = exclude_outside, ?extrapolation_value = extrapolation_value, ?keep_aspect_ratio_policy = keep_aspect_ratio_policy, ?mode = mode, ?nearest_mode = nearest_mode)
    static member non_max_suppression(boxes: Tensor<float32>, scores: Tensor<float32>, ?max_output_boxes_per_class: Tensor<int64>, ?iou_threshold: Tensor<float32>, ?score_threshold: Tensor<float32>, ?center_point_box: int64) =
        on.NonMaxSuppression(boxes = boxes, scores = scores, ?max_output_boxes_per_class = max_output_boxes_per_class, ?iou_threshold = iou_threshold, ?score_threshold = score_threshold, ?center_point_box = center_point_box)
    static member string_normalizer(X: Tensor<string>, ?case_change_action: string, ?is_case_sensitive: int64, ?locale: string, ?stopwords: string[]) =
        on.StringNormalizer(X = X, ?case_change_action = case_change_action, ?is_case_sensitive = is_case_sensitive, ?locale = locale, ?stopwords = stopwords)
    static member label_encoder(X: Tensor<string>, ?default_float: float32, ?default_int64: int64, ?default_string: string, ?keys_floats: float32[], ?keys_int64s: int64[], ?keys_strings: string[], ?values_floats: float32[], ?values_int64s: int64[], ?values_strings: string[]) =
        on.LabelEncoder(X = X, ?default_float = default_float, ?default_int64 = default_int64, ?default_string = default_string, ?keys_floats = keys_floats, ?keys_int64s = keys_int64s, ?keys_strings = keys_strings, ?values_floats = values_floats, ?values_int64s = values_int64s, ?values_strings = values_strings)
    static member label_encoder(X: Tensor<int64>, ?default_float: float32, ?default_int64: int64, ?default_string: string, ?keys_floats: float32[], ?keys_int64s: int64[], ?keys_strings: string[], ?values_floats: float32[], ?values_int64s: int64[], ?values_strings: string[]) =
        on.LabelEncoder(X = X, ?default_float = default_float, ?default_int64 = default_int64, ?default_string = default_string, ?keys_floats = keys_floats, ?keys_int64s = keys_int64s, ?keys_strings = keys_strings, ?values_floats = values_floats, ?values_int64s = values_int64s, ?values_strings = values_strings)
    static member label_encoder(X: Tensor<float32>, ?default_float: float32, ?default_int64: int64, ?default_string: string, ?keys_floats: float32[], ?keys_int64s: int64[], ?keys_strings: string[], ?values_floats: float32[], ?values_int64s: int64[], ?values_strings: string[]) =
        on.LabelEncoder(X = X, ?default_float = default_float, ?default_int64 = default_int64, ?default_string = default_string, ?keys_floats = keys_floats, ?keys_int64s = keys_int64s, ?keys_strings = keys_strings, ?values_floats = values_floats, ?values_int64s = values_int64s, ?values_strings = values_strings)
    static member category_mapper(X: Tensor<string>, ?cats_int64s: int64[], ?cats_strings: string[], ?default_int64: int64, ?default_string: string) =
        on.CategoryMapper(X = X, ?cats_int64s = cats_int64s, ?cats_strings = cats_strings, ?default_int64 = default_int64, ?default_string = default_string)
    static member category_mapper(X: Tensor<int64>, ?cats_int64s: int64[], ?cats_strings: string[], ?default_int64: int64, ?default_string: string) =
        on.CategoryMapper(X = X, ?cats_int64s = cats_int64s, ?cats_strings = cats_strings, ?default_int64 = default_int64, ?default_string = default_string)
    static member sequence_empty<'a>() =
        on.SequenceEmpty<'a>()
    static member eye_like<'a>(input: Tensor<uint8>, ?k: int64) =
        on.EyeLike<'a>(input = input, ?k = k)
    static member eye_like(input: Tensor<uint8>, ?k: int64) =
        on.EyeLike(input = input, ?k = k)
    static member eye_like<'a>(input: Tensor<int16>, ?k: int64) =
        on.EyeLike<'a>(input = input, ?k = k)
    static member eye_like(input: Tensor<int16>, ?k: int64) =
        on.EyeLike(input = input, ?k = k)
    static member eye_like<'a>(input: Tensor<float32>, ?k: int64) =
        on.EyeLike<'a>(input = input, ?k = k)
    static member eye_like(input: Tensor<float32>, ?k: int64) =
        on.EyeLike(input = input, ?k = k)
    static member eye_like<'a>(input: Tensor<uint32>, ?k: int64) =
        on.EyeLike<'a>(input = input, ?k = k)
    static member eye_like(input: Tensor<uint32>, ?k: int64) =
        on.EyeLike(input = input, ?k = k)
    static member eye_like<'a>(input: Tensor<int8>, ?k: int64) =
        on.EyeLike<'a>(input = input, ?k = k)
    static member eye_like(input: Tensor<int8>, ?k: int64) =
        on.EyeLike(input = input, ?k = k)
    static member eye_like<'a>(input: Tensor<uint64>, ?k: int64) =
        on.EyeLike<'a>(input = input, ?k = k)
    static member eye_like(input: Tensor<uint64>, ?k: int64) =
        on.EyeLike(input = input, ?k = k)
    static member eye_like<'a>(input: Tensor<int>, ?k: int64) =
        on.EyeLike<'a>(input = input, ?k = k)
    static member eye_like(input: Tensor<int>, ?k: int64) =
        on.EyeLike(input = input, ?k = k)
    static member eye_like<'a>(input: Tensor<int64>, ?k: int64) =
        on.EyeLike<'a>(input = input, ?k = k)
    static member eye_like(input: Tensor<int64>, ?k: int64) =
        on.EyeLike(input = input, ?k = k)
    static member eye_like<'a>(input: Tensor<uint16>, ?k: int64) =
        on.EyeLike<'a>(input = input, ?k = k)
    static member eye_like(input: Tensor<uint16>, ?k: int64) =
        on.EyeLike(input = input, ?k = k)
    static member eye_like<'a>(input: Tensor<bool>, ?k: int64) =
        on.EyeLike<'a>(input = input, ?k = k)
    static member eye_like(input: Tensor<bool>, ?k: int64) =
        on.EyeLike(input = input, ?k = k)
    static member eye_like<'a>(input: Tensor<double>, ?k: int64) =
        on.EyeLike<'a>(input = input, ?k = k)
    static member eye_like(input: Tensor<double>, ?k: int64) =
        on.EyeLike(input = input, ?k = k)
    static member multinomial<'a>(input: Tensor<float32>, ?sample_size: int64, ?seed: float32) =
        on.Multinomial<'a>(input = input, ?sample_size = sample_size, ?seed = seed)
    static member multinomial(input: Tensor<float32>, ?sample_size: int64, ?seed: float32) =
        on.Multinomial(input = input, ?sample_size = sample_size, ?seed = seed)
    static member multinomial<'a>(input: Tensor<double>, ?sample_size: int64, ?seed: float32) =
        on.Multinomial<'a>(input = input, ?sample_size = sample_size, ?seed = seed)
    static member multinomial(input: Tensor<double>, ?sample_size: int64, ?seed: float32) =
        on.Multinomial(input = input, ?sample_size = sample_size, ?seed = seed)
    static member random_uniform_like<'a>(input: Tensor<uint8>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike<'a>(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like(input: Tensor<uint8>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like<'a>(input: Tensor<int16>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike<'a>(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like(input: Tensor<int16>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like<'a>(input: Tensor<int64>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike<'a>(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like(input: Tensor<int64>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like<'a>(input: Tensor<int8>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike<'a>(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like(input: Tensor<int8>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like<'a>(input: Tensor<float32>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike<'a>(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like(input: Tensor<float32>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like<'a>(input: Tensor<string>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike<'a>(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like(input: Tensor<string>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like<'a>(input: Tensor<uint64>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike<'a>(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like(input: Tensor<uint64>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like<'a>(input: Tensor<int>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike<'a>(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like(input: Tensor<int>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like<'a>(input: Tensor<Complex>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike<'a>(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like(input: Tensor<Complex>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like<'a>(input: Tensor<uint32>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike<'a>(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like(input: Tensor<uint32>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like<'a>(input: Tensor<uint16>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike<'a>(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like(input: Tensor<uint16>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like<'a>(input: Tensor<bool>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike<'a>(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like(input: Tensor<bool>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like<'a>(input: Tensor<double>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike<'a>(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_uniform_like(input: Tensor<double>, ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniformLike(input = input, ?high = high, ?low = low, ?seed = seed)
    static member random_normal_like<'a>(input: Tensor<uint8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike<'a>(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like(input: Tensor<uint8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like<'a>(input: Tensor<int16>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike<'a>(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like(input: Tensor<int16>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like<'a>(input: Tensor<int64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike<'a>(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like(input: Tensor<int64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like<'a>(input: Tensor<int8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike<'a>(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like(input: Tensor<int8>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like<'a>(input: Tensor<float32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike<'a>(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like(input: Tensor<float32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like<'a>(input: Tensor<string>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike<'a>(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like(input: Tensor<string>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like<'a>(input: Tensor<uint64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike<'a>(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like(input: Tensor<uint64>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like<'a>(input: Tensor<int>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike<'a>(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like(input: Tensor<int>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like<'a>(input: Tensor<Complex>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike<'a>(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like(input: Tensor<Complex>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like<'a>(input: Tensor<uint32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike<'a>(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like(input: Tensor<uint32>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like<'a>(input: Tensor<uint16>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike<'a>(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like(input: Tensor<uint16>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like<'a>(input: Tensor<bool>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike<'a>(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like(input: Tensor<bool>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like<'a>(input: Tensor<double>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike<'a>(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal_like(input: Tensor<double>, ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormalLike(input = input, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_normal<'a>(shape: int64[], ?mean: float32, ?scale: float32, ?seed: float32) =
        on.RandomNormal<'a>(shape = shape, ?mean = mean, ?scale = scale, ?seed = seed)
    static member random_uniform<'a>(shape: int64[], ?high: float32, ?low: float32, ?seed: float32) =
        on.RandomUniform<'a>(shape = shape, ?high = high, ?low = low, ?seed = seed)
    static member cast<'a>(input: Tensor<uint8>, ?saturate: int64) =
        on.Cast<'a>(input = input, ?saturate = saturate)
    static member cast<'a>(input: Tensor<int16>, ?saturate: int64) =
        on.Cast<'a>(input = input, ?saturate = saturate)
    static member cast<'a>(input: Tensor<int64>, ?saturate: int64) =
        on.Cast<'a>(input = input, ?saturate = saturate)
    static member cast<'a>(input: Tensor<int8>, ?saturate: int64) =
        on.Cast<'a>(input = input, ?saturate = saturate)
    static member cast<'a>(input: Tensor<uint64>, ?saturate: int64) =
        on.Cast<'a>(input = input, ?saturate = saturate)
    static member cast<'a>(input: Tensor<float32>, ?saturate: int64) =
        on.Cast<'a>(input = input, ?saturate = saturate)
    static member cast<'a>(input: Tensor<int>, ?saturate: int64) =
        on.Cast<'a>(input = input, ?saturate = saturate)
    static member cast<'a>(input: Tensor<string>, ?saturate: int64) =
        on.Cast<'a>(input = input, ?saturate = saturate)
    static member cast<'a>(input: Tensor<uint32>, ?saturate: int64) =
        on.Cast<'a>(input = input, ?saturate = saturate)
    static member cast<'a>(input: Tensor<uint16>, ?saturate: int64) =
        on.Cast<'a>(input = input, ?saturate = saturate)
    static member cast<'a>(input: Tensor<bool>, ?saturate: int64) =
        on.Cast<'a>(input = input, ?saturate = saturate)
    static member cast<'a>(input: Tensor<double>, ?saturate: int64) =
        on.Cast<'a>(input = input, ?saturate = saturate)
    static member lstm(X: Tensor<float32>, W: Tensor<float32>, R: Tensor<float32>, ?B: Tensor<float32>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<float32>, ?initial_c: Tensor<float32>, ?P: Tensor<float32>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?input_forget: int64, ?layout: int64) =
        on.LSTM(X = X, W = W, R = R, ?B = B, ?sequence_lens = sequence_lens, ?initial_h = initial_h, ?initial_c = initial_c, ?P = P, ?activation_alpha = activation_alpha, ?activation_beta = activation_beta, ?activations = activations, ?clip = clip, ?direction = direction, ?hidden_size = hidden_size, ?input_forget = input_forget, ?layout = layout)
    static member lstm(X: Tensor<double>, W: Tensor<double>, R: Tensor<double>, ?B: Tensor<double>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<double>, ?initial_c: Tensor<double>, ?P: Tensor<double>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?input_forget: int64, ?layout: int64) =
        on.LSTM(X = X, W = W, R = R, ?B = B, ?sequence_lens = sequence_lens, ?initial_h = initial_h, ?initial_c = initial_c, ?P = P, ?activation_alpha = activation_alpha, ?activation_beta = activation_beta, ?activations = activations, ?clip = clip, ?direction = direction, ?hidden_size = hidden_size, ?input_forget = input_forget, ?layout = layout)
    static member linear_classifier(X: Tensor<float32>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        on.LinearClassifier(X = X, coefficients = coefficients, ?classlabels_ints = classlabels_ints, ?classlabels_strings = classlabels_strings, ?intercepts = intercepts, ?multi_class = multi_class, ?post_transform = post_transform)
    static member linear_classifier(X: Tensor<double>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        on.LinearClassifier(X = X, coefficients = coefficients, ?classlabels_ints = classlabels_ints, ?classlabels_strings = classlabels_strings, ?intercepts = intercepts, ?multi_class = multi_class, ?post_transform = post_transform)
    static member linear_classifier(X: Tensor<int64>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        on.LinearClassifier(X = X, coefficients = coefficients, ?classlabels_ints = classlabels_ints, ?classlabels_strings = classlabels_strings, ?intercepts = intercepts, ?multi_class = multi_class, ?post_transform = post_transform)
    static member linear_classifier(X: Tensor<int>, coefficients: float32[], ?classlabels_ints: int64[], ?classlabels_strings: string[], ?intercepts: float32[], ?multi_class: int64, ?post_transform: string) =
        on.LinearClassifier(X = X, coefficients = coefficients, ?classlabels_ints = classlabels_ints, ?classlabels_strings = classlabels_strings, ?intercepts = intercepts, ?multi_class = multi_class, ?post_transform = post_transform)
    static member svm_classifier(X: Tensor<float32>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        on.SVMClassifier(X = X, ?classlabels_ints = classlabels_ints, ?classlabels_strings = classlabels_strings, ?coefficients = coefficients, ?kernel_params = kernel_params, ?kernel_type = kernel_type, ?post_transform = post_transform, ?prob_a = prob_a, ?prob_b = prob_b, ?rho = rho, ?support_vectors = support_vectors, ?vectors_per_class = vectors_per_class)
    static member svm_classifier(X: Tensor<double>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        on.SVMClassifier(X = X, ?classlabels_ints = classlabels_ints, ?classlabels_strings = classlabels_strings, ?coefficients = coefficients, ?kernel_params = kernel_params, ?kernel_type = kernel_type, ?post_transform = post_transform, ?prob_a = prob_a, ?prob_b = prob_b, ?rho = rho, ?support_vectors = support_vectors, ?vectors_per_class = vectors_per_class)
    static member svm_classifier(X: Tensor<int64>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        on.SVMClassifier(X = X, ?classlabels_ints = classlabels_ints, ?classlabels_strings = classlabels_strings, ?coefficients = coefficients, ?kernel_params = kernel_params, ?kernel_type = kernel_type, ?post_transform = post_transform, ?prob_a = prob_a, ?prob_b = prob_b, ?rho = rho, ?support_vectors = support_vectors, ?vectors_per_class = vectors_per_class)
    static member svm_classifier(X: Tensor<int>, ?classlabels_ints: int64[], ?classlabels_strings: string[], ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?post_transform: string, ?prob_a: float32[], ?prob_b: float32[], ?rho: float32[], ?support_vectors: float32[], ?vectors_per_class: int64[]) =
        on.SVMClassifier(X = X, ?classlabels_ints = classlabels_ints, ?classlabels_strings = classlabels_strings, ?coefficients = coefficients, ?kernel_params = kernel_params, ?kernel_type = kernel_type, ?post_transform = post_transform, ?prob_a = prob_a, ?prob_b = prob_b, ?rho = rho, ?support_vectors = support_vectors, ?vectors_per_class = vectors_per_class)
    static member max_pool(X: Tensor<float32>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?pads: int64[], ?storage_order: int64, ?strides: int64[]) =
        on.MaxPool(X = X, kernel_shape = kernel_shape, ?auto_pad = auto_pad, ?dilations = dilations, ?pads = pads, ?storage_order = storage_order, ?strides = strides)
    static member max_pool(X: Tensor<double>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?pads: int64[], ?storage_order: int64, ?strides: int64[]) =
        on.MaxPool(X = X, kernel_shape = kernel_shape, ?auto_pad = auto_pad, ?dilations = dilations, ?pads = pads, ?storage_order = storage_order, ?strides = strides)
    static member max_pool(X: Tensor<int8>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?pads: int64[], ?storage_order: int64, ?strides: int64[]) =
        on.MaxPool(X = X, kernel_shape = kernel_shape, ?auto_pad = auto_pad, ?dilations = dilations, ?pads = pads, ?storage_order = storage_order, ?strides = strides)
    static member max_pool(X: Tensor<uint8>, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?pads: int64[], ?storage_order: int64, ?strides: int64[]) =
        on.MaxPool(X = X, kernel_shape = kernel_shape, ?auto_pad = auto_pad, ?dilations = dilations, ?pads = pads, ?storage_order = storage_order, ?strides = strides)
    static member gru(X: Tensor<float32>, W: Tensor<float32>, R: Tensor<float32>, ?B: Tensor<float32>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<float32>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?layout: int64, ?linear_before_reset: int64) =
        on.GRU(X = X, W = W, R = R, ?B = B, ?sequence_lens = sequence_lens, ?initial_h = initial_h, ?activation_alpha = activation_alpha, ?activation_beta = activation_beta, ?activations = activations, ?clip = clip, ?direction = direction, ?hidden_size = hidden_size, ?layout = layout, ?linear_before_reset = linear_before_reset)
    static member gru(X: Tensor<double>, W: Tensor<double>, R: Tensor<double>, ?B: Tensor<double>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<double>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?layout: int64, ?linear_before_reset: int64) =
        on.GRU(X = X, W = W, R = R, ?B = B, ?sequence_lens = sequence_lens, ?initial_h = initial_h, ?activation_alpha = activation_alpha, ?activation_beta = activation_beta, ?activations = activations, ?clip = clip, ?direction = direction, ?hidden_size = hidden_size, ?layout = layout, ?linear_before_reset = linear_before_reset)
    static member topk(X: Tensor<uint8>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        on.TopK(X = X, K = K, ?axis = axis, ?largest = largest, ?sorted = sorted)
    static member topk(X: Tensor<uint16>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        on.TopK(X = X, K = K, ?axis = axis, ?largest = largest, ?sorted = sorted)
    static member topk(X: Tensor<uint32>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        on.TopK(X = X, K = K, ?axis = axis, ?largest = largest, ?sorted = sorted)
    static member topk(X: Tensor<uint64>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        on.TopK(X = X, K = K, ?axis = axis, ?largest = largest, ?sorted = sorted)
    static member topk(X: Tensor<int8>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        on.TopK(X = X, K = K, ?axis = axis, ?largest = largest, ?sorted = sorted)
    static member topk(X: Tensor<int16>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        on.TopK(X = X, K = K, ?axis = axis, ?largest = largest, ?sorted = sorted)
    static member topk(X: Tensor<int>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        on.TopK(X = X, K = K, ?axis = axis, ?largest = largest, ?sorted = sorted)
    static member topk(X: Tensor<int64>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        on.TopK(X = X, K = K, ?axis = axis, ?largest = largest, ?sorted = sorted)
    static member topk(X: Tensor<float32>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        on.TopK(X = X, K = K, ?axis = axis, ?largest = largest, ?sorted = sorted)
    static member topk(X: Tensor<double>, K: Tensor<int64>, ?axis: int64, ?largest: int64, ?sorted: int64) =
        on.TopK(X = X, K = K, ?axis = axis, ?largest = largest, ?sorted = sorted)
    static member dropout(data: Tensor<float32>, ?ratio: Tensor<float32>, ?training_mode: Tensor<bool>, ?seed: int64) =
        on.Dropout(data = data, ?ratio = ratio, ?training_mode = training_mode, ?seed = seed)
    static member dropout(data: Tensor<float32>, ?ratio: Tensor<double>, ?training_mode: Tensor<bool>, ?seed: int64) =
        on.Dropout(data = data, ?ratio = ratio, ?training_mode = training_mode, ?seed = seed)
    static member dropout(data: Tensor<double>, ?ratio: Tensor<float32>, ?training_mode: Tensor<bool>, ?seed: int64) =
        on.Dropout(data = data, ?ratio = ratio, ?training_mode = training_mode, ?seed = seed)
    static member dropout(data: Tensor<double>, ?ratio: Tensor<double>, ?training_mode: Tensor<bool>, ?seed: int64) =
        on.Dropout(data = data, ?ratio = ratio, ?training_mode = training_mode, ?seed = seed)
    static member unique(X: Tensor<uint8>, ?axis: int64, ?sorted: int64) =
        on.Unique(X = X, ?axis = axis, ?sorted = sorted)
    static member unique(X: Tensor<uint16>, ?axis: int64, ?sorted: int64) =
        on.Unique(X = X, ?axis = axis, ?sorted = sorted)
    static member unique(X: Tensor<uint32>, ?axis: int64, ?sorted: int64) =
        on.Unique(X = X, ?axis = axis, ?sorted = sorted)
    static member unique(X: Tensor<uint64>, ?axis: int64, ?sorted: int64) =
        on.Unique(X = X, ?axis = axis, ?sorted = sorted)
    static member unique(X: Tensor<int8>, ?axis: int64, ?sorted: int64) =
        on.Unique(X = X, ?axis = axis, ?sorted = sorted)
    static member unique(X: Tensor<int16>, ?axis: int64, ?sorted: int64) =
        on.Unique(X = X, ?axis = axis, ?sorted = sorted)
    static member unique(X: Tensor<int>, ?axis: int64, ?sorted: int64) =
        on.Unique(X = X, ?axis = axis, ?sorted = sorted)
    static member unique(X: Tensor<int64>, ?axis: int64, ?sorted: int64) =
        on.Unique(X = X, ?axis = axis, ?sorted = sorted)
    static member unique(X: Tensor<float32>, ?axis: int64, ?sorted: int64) =
        on.Unique(X = X, ?axis = axis, ?sorted = sorted)
    static member unique(X: Tensor<double>, ?axis: int64, ?sorted: int64) =
        on.Unique(X = X, ?axis = axis, ?sorted = sorted)
    static member unique(X: Tensor<string>, ?axis: int64, ?sorted: int64) =
        on.Unique(X = X, ?axis = axis, ?sorted = sorted)
    static member unique(X: Tensor<bool>, ?axis: int64, ?sorted: int64) =
        on.Unique(X = X, ?axis = axis, ?sorted = sorted)
    static member unique(X: Tensor<Complex>, ?axis: int64, ?sorted: int64) =
        on.Unique(X = X, ?axis = axis, ?sorted = sorted)
    static member dynamic_quantize_linear(x: Tensor<float32>) =
        on.DynamicQuantizeLinear(x = x)
    static member rnn(X: Tensor<float32>, W: Tensor<float32>, R: Tensor<float32>, ?B: Tensor<float32>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<float32>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?layout: int64) =
        on.RNN(X = X, W = W, R = R, ?B = B, ?sequence_lens = sequence_lens, ?initial_h = initial_h, ?activation_alpha = activation_alpha, ?activation_beta = activation_beta, ?activations = activations, ?clip = clip, ?direction = direction, ?hidden_size = hidden_size, ?layout = layout)
    static member rnn(X: Tensor<double>, W: Tensor<double>, R: Tensor<double>, ?B: Tensor<double>, ?sequence_lens: Tensor<int>, ?initial_h: Tensor<double>, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?layout: int64) =
        on.RNN(X = X, W = W, R = R, ?B = B, ?sequence_lens = sequence_lens, ?initial_h = initial_h, ?activation_alpha = activation_alpha, ?activation_beta = activation_beta, ?activations = activations, ?clip = clip, ?direction = direction, ?hidden_size = hidden_size, ?layout = layout)
    static member batch_normalization(X: Tensor<float32>, scale: Tensor<float32>, B: Tensor<float32>, input_mean: Tensor<float32>, input_var: Tensor<float32>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        on.BatchNormalization(X = X, scale = scale, B = B, input_mean = input_mean, input_var = input_var, ?epsilon = epsilon, ?momentum = momentum, ?training_mode = training_mode)
    static member batch_normalization(X: Tensor<float32>, scale: Tensor<float32>, B: Tensor<float32>, input_mean: Tensor<double>, input_var: Tensor<double>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        on.BatchNormalization(X = X, scale = scale, B = B, input_mean = input_mean, input_var = input_var, ?epsilon = epsilon, ?momentum = momentum, ?training_mode = training_mode)
    static member batch_normalization(X: Tensor<float32>, scale: Tensor<double>, B: Tensor<double>, input_mean: Tensor<float32>, input_var: Tensor<float32>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        on.BatchNormalization(X = X, scale = scale, B = B, input_mean = input_mean, input_var = input_var, ?epsilon = epsilon, ?momentum = momentum, ?training_mode = training_mode)
    static member batch_normalization(X: Tensor<float32>, scale: Tensor<double>, B: Tensor<double>, input_mean: Tensor<double>, input_var: Tensor<double>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        on.BatchNormalization(X = X, scale = scale, B = B, input_mean = input_mean, input_var = input_var, ?epsilon = epsilon, ?momentum = momentum, ?training_mode = training_mode)
    static member batch_normalization(X: Tensor<double>, scale: Tensor<float32>, B: Tensor<float32>, input_mean: Tensor<float32>, input_var: Tensor<float32>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        on.BatchNormalization(X = X, scale = scale, B = B, input_mean = input_mean, input_var = input_var, ?epsilon = epsilon, ?momentum = momentum, ?training_mode = training_mode)
    static member batch_normalization(X: Tensor<double>, scale: Tensor<float32>, B: Tensor<float32>, input_mean: Tensor<double>, input_var: Tensor<double>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        on.BatchNormalization(X = X, scale = scale, B = B, input_mean = input_mean, input_var = input_var, ?epsilon = epsilon, ?momentum = momentum, ?training_mode = training_mode)
    static member batch_normalization(X: Tensor<double>, scale: Tensor<double>, B: Tensor<double>, input_mean: Tensor<float32>, input_var: Tensor<float32>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        on.BatchNormalization(X = X, scale = scale, B = B, input_mean = input_mean, input_var = input_var, ?epsilon = epsilon, ?momentum = momentum, ?training_mode = training_mode)
    static member batch_normalization(X: Tensor<double>, scale: Tensor<double>, B: Tensor<double>, input_mean: Tensor<double>, input_var: Tensor<double>, ?epsilon: float32, ?momentum: float32, ?training_mode: int64) =
        on.BatchNormalization(X = X, scale = scale, B = B, input_mean = input_mean, input_var = input_var, ?epsilon = epsilon, ?momentum = momentum, ?training_mode = training_mode)
