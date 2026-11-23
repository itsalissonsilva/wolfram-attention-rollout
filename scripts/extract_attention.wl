root = DirectoryName[DirectoryName[$InputFileName]];
out = Environment["ATTENTION_ROLLOUT_OUTPUT"];
$Messages = {OpenWrite[FileNameJoin[{out, "messages.txt"}]]};
$AllowInternet = False;
Get[FileNameJoin[{root, "HeadLayerAnalysis.wl"}]];
progress[stage_] := Export[FileNameJoin[{out, "progress.json"}], <|"Stage" -> stage|>, "RawJSON"];
fail[stage_] := (Export[FileNameJoin[{out, "result.json"}], <|"Success" -> False, "Stage" -> stage|>, "RawJSON"]; Quit[1]);
dump[name_, data_] := Module[{s, path = FileNameJoin[{out, name}]},
  s = OpenWrite[path, BinaryFormat -> True]; BinaryWrite[s, Flatten[data], "Real32", ByteOrdering -> -1]; Close[s];
  <|"Shape" -> Dimensions[data], "SHA256" -> IntegerString[FileHash[path, "SHA256"], 16, 64]|>
];
imageID = Environment["ATTENTION_ROLLOUT_IMAGE"];
If[imageID === $Failed, imageID = "samoyed"];
If[!MemberQ[{"samoyed", "samoyed-option-1", "samoyed-option-4", "samoyed-option-5"}, imageID], fail["UnknownImage"]];
item = Import[FileNameJoin[{root, "experiments", imageID <> ".json"}], "RawJSON"];
sourceImage = FileNameJoin[{root, item["LocalFile"]}];
If[IntegerString[FileHash[sourceImage, "SHA256"], 16, 64] != item["SHA256"], fail["ImageHash"]];
progress["Loading Base/16"];
model = Import[FileNameJoin[{root, "experiments", "model.json"}], "RawJSON"];
modelPath = FileNameJoin[{root, model["LocalFile"]}];
If[IntegerString[FileHash[modelPath, "SHA256"], 16, 64] != model["SHA256"], fail["ModelHash"]];
net = Import[modelPath, "WLNet"];
net = NetReplacePart[net, {"Input", "VarianceImage"} -> {.229, .224, .225}^2];
split = NetPort[{"patch_embeddings", "Input"}];
progress["Preparing model crop"];
pixels = Normal[NetTake[net, split][Import[sourceImage], TargetDevice -> "CPU", NetEvaluationMode -> "Test"]];
inputInfo = dump["input.f32", pixels];
crop = Image[Clip[MapThread[#1 #2 + #3 &, {pixels, {.229, .224, .225}, {.485, .456, .406}}], {0, 1}], "Real", Interleaving -> False];
Export[FileNameJoin[{out, "model-crop.png"}], crop];
suffix = NetTake[net, {split, NetPort["Output"]}];
ports = NetPort[{"encoder", #, "attention", "attention", "AttentionWeights"}] & /@ Range[12];
logitPort = NetPort[{"predict", "Output"}];
progress["Extracting all heads and layers"];
evaluated = suffix[pixels, Join[{logitPort}, ports], TargetDevice -> "CPU", NetEvaluationMode -> "Test"];
If[!AssociationQ[evaluated], fail["Evaluation"]];
attention = Transpose[Normal[evaluated[#]], {2, 1, 3}] & /@ ports;
logits = Normal[evaluated[logitPort]];
attentionInfo = dump["attention.f32", attention];
progress["Analyzing attention"];
analysis = AttentionHeadLayerAnalysis[attention, {14, 14}];
If[FailureQ[analysis], fail["Analysis"]];
labels = NetExtract[net, {"Output", "Labels"}];
labelName[label_] := Replace[label, Entity[_, name_String] :>
  StringReplace[First[StringSplit[name, "::"]], RegularExpression["([a-z])([A-Z])"] -> "$1 $2"]];
probabilities = Exp[logits - Max[logits]]; probabilities /= Total[probabilities];
indices = Reverse[Ordering[probabilities, -3]];
top3 = (<|"Class" -> labelName[labels[[#]]], "Index" -> (# - 1), "Probability" -> probabilities[[#]]|> &) /@ indices;
analysis = Join[analysis, <|"Model" -> "B16", "Case" -> imageID, "Normalization" -> "StandardDeviation",
  "PatchPixels" -> 16, "Top3" -> top3, "ImageSource" -> item,
  "Scope" -> "Single image, standard mean-head residual-aware rollout, unchanged model weights."|>];
Export[FileNameJoin[{out, "analysis.json"}], analysis, "RawJSON"];
Export[FileNameJoin[{out, "logits.json"}], logits, "RawJSON"];
Export[FileNameJoin[{out, "exploration.wxf"}], Join[analysis, <|"ModelCrop" -> crop|>], "WXF"];
rowError = Max[Abs[Flatten[Map[Total, attention, {3}]] - 1]];
Export[FileNameJoin[{out, "result.json"}], <|
  "Success" -> (Dimensions[attention] === {12, 12, 197, 197} && rowError < 10^-5 && analysis["FinalMixture"]["MaximumError"] < 10^-12),
  "Model" -> "B16", "Image" -> imageID, "Top3" -> top3,
  "AttentionShape" -> attentionInfo["Shape"], "AttentionSHA256" -> attentionInfo["SHA256"],
  "InputSHA256" -> inputInfo["SHA256"], "ImageSHA256" -> item["SHA256"], "AttentionRowSumError" -> rowError,
  "FinalLayer" -> Last[analysis["LayerMetrics"]], "FinalMixture" -> KeyDrop[analysis["FinalMixture"], "SpatialMixtureWeights"],
  "InferenceCalls" -> 1, "WolframVersion" -> $Version
|>, "RawJSON"];
progress["Complete"];
Quit[];
