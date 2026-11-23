Get[FileNameJoin[{DirectoryName[$InputFileName], "HeadLayerAnalysis.wl"}]];
BeginPackage["WolframAttention`"];
ReproduceAttention::usage = "ReproduceAttention[id] recomputes head maps, rollout and layer metrics from a cached attention tensor. The default id is samoyed.";
Begin["`Private`"];
reproductionRoot = DirectoryName[$InputFileName];

ReproduceAttention[id_String : "samoyed"] := Module[
  {directory, record, path, digest, attention, analysis, expected, error, keys},
  If[!MemberQ[{"samoyed", "samoyed-option-4", "samoyed-option-5", "samoyed-option-1"}, id],
    Return[Failure["UnknownExample", <|"ID" -> id|>]]];
  directory = FileNameJoin[{reproductionRoot, "results", id <> "-native-001"}];
  path = FileNameJoin[{directory, "attention.f32"}];
  If[!FileExistsQ[path], Return[Failure["MissingTensor", <|"Path" -> path|>]]];
  record = Import[FileNameJoin[{directory, "result.json"}], "RawJSON"];
  digest = IntegerString[FileHash[path, "SHA256"], 16, 64];
  If[digest =!= record["AttentionSHA256"], Return[Failure["TensorHash", <|"Path" -> path|>]]];
  attention = ArrayReshape[BinaryReadList[path, "Real32", ByteOrdering -> -1], record["AttentionShape"]];
  analysis = AttentionHeadLayerAnalysis[attention, {14, 14}];
  If[FailureQ[analysis], Return[analysis]];
  expected = Import[FileNameJoin[{directory, "analysis.json"}], "RawJSON"];
  keys = {"HeadPatchMaps", "MeanPatchMaps", "RolloutPatchMaps", "PreviousCLSFootprint"};
  error = Max[(Max[Abs[Flatten[analysis[#] - expected[#]]]] &) /@ keys];
  If[error >= 10^-12, Return[Failure["MapMismatch", <|"MaximumError" -> error|>]]];
  Join[analysis, <|"ID" -> id, "HeadMean" -> Last[analysis["MeanPatchMaps"]],
    "Rollout" -> Last[analysis["RolloutPatchMaps"]],
    "ModelCrop" -> Import[FileNameJoin[{directory, "model-crop.png"}]],
    "Top3" -> record["Top3"], "MaximumMapError" -> error|>]
];

End[];
EndPackage[];
