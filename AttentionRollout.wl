(* Native Wolfram Language attention rollout. No external framework is required. *)
BeginPackage["WolframAttention`"];

AttentionRollout::usage = "AttentionRollout[attention] computes mean-head attention rollout. Input axes are {layer, head, query, key}, or {layer, query, key} for already fused heads. Returns raw, residual-adjusted and cumulative matrices.";
AttentionPatchMap::usage = "AttentionPatchMap[matrix, grid, token] extracts a query token's attention to spatial patches, excluding that token. grid is {rows, columns}; token defaults to 1.";

Begin["`Private`"];

Options[AttentionRollout] = {"ResidualWeight" -> 1.};

AttentionRollout[input_, OptionsPattern[]] := Module[
  {dims = Dimensions[input], rank, data, valid, bounds, layers, heads, tokens, weight = OptionValue["ResidualWeight"], fused, residual, joint, rowError},
  rank = Length[dims];
  data = Developer`ToPackedArray[input];
  (* Avoid millions of interpreted scalar predicates for B/16 attention tensors.
     Packed real/integer arrays are rectangular and contain finite machine numbers.
     Retain the original validation path for exact and other unpacked input. *)
  valid = If[Developer`PackedArrayQ[data, Real] || Developer`PackedArrayQ[data, Integer],
    bounds = MinMax[Flatten[data]]; TrueQ[0 <= First[bounds] && Last[bounds] < Infinity],
    ArrayQ[input, rank, (NumberQ[#] && TrueQ[0 <= # < Infinity]) &]];
  If[!MemberQ[{3, 4}, rank] || !valid || Min[dims] < 1,
    Return[Failure["InvalidAttention", <|"MessageTemplate" -> "Expected a nonempty rectangular array of finite, nonnegative real attention weights with rank 3 or 4."|>]]];
  If[dims[[-1]] != dims[[-2]],
    Return[Failure["NonSquareAttention", <|"MessageTemplate" -> "Rollout requires square self-attention matrices with a fixed token layout."|>]]];
  If[!NumberQ[weight] || !TrueQ[0 <= weight < Infinity],
    Return[Failure["InvalidResidualWeight", <|"MessageTemplate" -> "ResidualWeight must be a finite, nonnegative real number."|>]]];
  {layers, tokens} = {First[dims], Last[dims]};
  heads = If[rank == 4, dims[[2]], 1];
  data = N[data];
  rowError = Max[Abs[(Total /@ Flatten[data, rank - 2]) - 1.]];
  If[rowError > 10^-5,
    Return[Failure["NonStochasticAttention", <|"MessageTemplate" -> "Attention must sum to one over keys for each query; check axis order and inference mode.", "MaximumRowSumError" -> rowError|>]]];
  fused = If[rank == 4, Mean /@ data, data];
  residual = Map[Function[a, (#/Total[#] & /@ (a + weight IdentityMatrix[tokens]))], fused];
  (* Rows are queries, columns are keys. New layers multiply on the LEFT. *)
  joint = Rest@FoldList[#2 . #1 &, N@IdentityMatrix[tokens], residual];
  <|"HeadMeanAttention" -> fused, "ResidualAttention" -> residual,
    "RolloutByLayer" -> joint, "Rollout" -> Last[joint],
    "LayerCount" -> layers, "HeadCount" -> heads, "TokenCount" -> tokens,
    "ResidualWeight" -> weight, "InputRowSumError" -> rowError|>
];

AttentionPatchMap[matrix_, grid_, token_: 1] := Module[{n, indices},
  If[!MatrixQ[matrix, NumericQ] || Length[Dimensions[matrix]] != 2 || Length[matrix] != Length[First[matrix]],
    Return[Failure["InvalidMatrix", <|"MessageTemplate" -> "Expected a square query-by-key matrix."|>]]];
  n = Length[matrix];
  If[!MatchQ[grid, {_Integer?Positive, _Integer?Positive}] || Times @@ grid != n - 1 || !IntegerQ[token] || !Between[token, {1, n}],
    Return[Failure["InvalidPatchLayout", <|"MessageTemplate" -> "This adapter requires one special token and a grid containing all other tokens."|>]]];
  indices = DeleteCases[Range[n], token];
  ArrayReshape[matrix[[token, indices]], grid]
];

End[];
EndPackage[];
