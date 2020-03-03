Runtime data is likely needed for the quotation transforms in Phase 2

The three alternatives approaches for this
(1) A python code generated F# object model
(2) A C++ wrapper for ONNX that exposes (see defn.psi) 
(3) A protobuf object model created from Python and loaded in F#

The quickest approach is (1) or (3) but (2) opens up the possibility to re-use the same wrapper for shape analysis


