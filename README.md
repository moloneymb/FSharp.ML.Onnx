# ONNX Backend
This repository is for experimenting with building an ONNX Backend. The API will be code generated from the ONNX operator definitions. The API will be a simple eager execution. F# quotations will be used to turn this API into an ONNX graph to improve execution performance. 

## Phase 1 Eager Execution
*	Create and execute MNist (done)
*	Create and execute single op graphs (done)
*	Create and execute MNist composed out of single op graphs (paused)
*	Protobuf interface for F# code generator (abandoned)
*	C++ wrapper interface (in-progress)
*	Python code generator for F# API ops (done - mostly)

## Phase 2 Graph Execution
*	Object model for ONNX metadata (done)
*	Quotation transform (done)
*	Simple structual types (done)

## Phase 3 Analyzer
*	Limit F# code at design time to ensure possilbe ONNX conversion

## Other
*	Use the ONNX library to resolve shapes for shape checking


