#load "Base.fsx"
//
//// TODO Evaulting sub expressions may not work if there are missing variable declarations
////      For example 'shape' var is missing in the tested sub expression
////      Reducing the sub expression will often fix this as it will push the variable assignment into 
////      the sub expression.
////      In general it's probably OK not to worry about it because the expression will still contain the Var
//
//
//let eagerToGraph (f: Expr<'a -> 'b>) =
//    f
//    |> Expr.applyTransform ExprTransforms.expandWithReflectedDefinition 
//    |> Expr.applyTransform (ExprTransforms.reduceApplicationsAndLambdas true) 
//    |> processExpr <@ Graph.Default() @>
