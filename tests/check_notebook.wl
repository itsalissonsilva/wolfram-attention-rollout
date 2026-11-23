root = DirectoryName[DirectoryName[$InputFileName]];
out = Environment["ATTENTION_ROLLOUT_OUTPUT"];
$AllowInternet = False;
$Messages = {OpenWrite[FileNameJoin[{out, "messages.txt"}]]};
notebook = Get[FileNameJoin[{root, "AttentionHeadsAndRollout.nb"}]];
boxes = DeleteCases[Cases[notebook, Cell[BoxData[b_], "Input", ___] :> b, Infinity], ""];
If[!MemberQ[{2, 3}, Length[boxes]],
  Export[FileNameJoin[{out, "result.json"}], <|"Success" -> False, "InputCellCount" -> Length[boxes]|>, "RawJSON"]; Quit[1]];
held = ToExpression[#, StandardForm, HoldComplete] & /@ boxes;
structureOK = And @@ (MatchQ[#, HoldComplete[_]] && FreeQ[#, _Defer] & /@ held);
ReleaseHold[held[[1]]];
toy = ReleaseHold[held[[2]]];
attention = ArrayReshape[BinaryReadList[FileNameJoin[{root, "results", "samoyed-native-001", "attention.f32"}],
  "Real32", ByteOrdering -> -1], {12, 12, 197, 197}];
headMean = Mean /@ attention;
rollout = attentionRollout[attention];
maps = {ArrayReshape[headMean[[-1, 1, 2 ;;]], {14, 14}], ArrayReshape[rollout[[1, 2 ;;]], {14, 14}]};
cachedCellOK = If[Length[held] == 3,
  Max[Abs[Flatten[maps - ReleaseHold[held[[3]] /. HoldPattern[NotebookDirectory[]] -> root]]]] < 10^-12,
  True];
expected = Import[FileNameJoin[{root, "results", "samoyed-native-001", "analysis.json"}], "RawJSON"];
error = Max[Abs[Flatten[maps - {Last[expected["MeanPatchMaps"]], Last[expected["RolloutPatchMaps"]]}]]];
passed = structureOK && cachedCellOK && toy === {{0, 1, 0}, {0, 0, 1}, 10} && Dimensions[maps] === {2, 14, 14} && error < 10^-12;
Export[FileNameJoin[{out, "diagnostics.txt"}], ToString[Short[{held, structureOK, toy, Dimensions[maps], error}, 6], InputForm], "Text"];
Export[FileNameJoin[{out, "result.json"}], <|"Success" -> TrueQ[passed], "Version" -> $Version,
  "InputCellCount" -> Length[boxes], "ExecutableCellStructure" -> structureOK,
  "ToyResult" -> ToString[toy, InputForm], "MaximumMapError" -> If[NumberQ[error], error, Null],
  "CachedCellPresent" -> (Length[held] == 3)|>, "RawJSON"];
Quit[If[passed, 0, 1]];
