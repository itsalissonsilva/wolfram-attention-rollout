Get[FileNameJoin[{DirectoryName[$InputFileName], "AttentionRollout.wl"}]];
BeginPackage["WolframAttention`"];

AttentionHeadLayerAnalysis::usage = "AttentionHeadLayerAnalysis[attention, grid] computes every CLS head map, standard mean-head rollout, head diversity and token-history mixing. Axes are layer, head, query, key; one leading CLS token is required.";
AttentionHeadLayerPlot::usage = "AttentionHeadLayerPlot[result, layer, head] compares a selected head, the head mean and cumulative rollout on the model crop.";
AttentionHeadGrid::usage = "AttentionHeadGrid[result, layer] displays all heads, their mean and rollout. Head order is unchanged.";
AttentionHeadLayerExplorer::usage = "AttentionHeadLayerExplorer[models] provides model, layer, head, view and color-scale controls for cached analysis associations.";
AttentionRolloutToy::usage = "AttentionRolloutToy[a,b,r] evaluates a fixed two-layer scalar mixing model with input {0,a,b} and residual weight r. No learned projections, MLPs or normalization layers are present.";
AttentionRolloutToyExplorer::usage = "AttentionRolloutToyExplorer[] displays the exact toy model with input-value and residual controls.";

Begin["`Private`"];
spatialNormalize[x_] := With[{mass = Total[Flatten[x]]}, If[mass > 0, x/mass, 0. x]];
spatialEntropy[x_] := Module[{p, mass = Total[Flatten[x]]},
  If[mass <= 0, Return[Missing["ZeroSpatialMass"]]];
  p = Select[Flatten[x]/mass, # > 0 &];
  -Total[p Log[p]]/Log[Length[Flatten[x]]]
];
displayNumber[x_?NumericQ] := StringTrim@ToString[NumberForm[x, {5, 3}, NumberPadding -> {"", "0"}], OutputForm];
displayNumber[_] := "n/a";

AttentionHeadLayerAnalysis[input_, grid_] := Module[
  {dims = Dimensions[input], a, rolled, fused, residual, joints, l, h, n, headRows, headMaps, metrics,
   mass, ent, weights, weighted, meanEntropy, rows, tv, previous, previousMass, previousRows,
   finalWeights, finalSpatial, selected, footprints, mixtureError, maxPreviousTV},
  If[Length[dims] != 4, Return[Failure["ExpectedHeads", <|"MessageTemplate" -> "Expected layer, head, query, key axes."|>]]];
  n = Last[dims];
  If[!MatchQ[grid, {_Integer?Positive, _Integer?Positive}] || Times @@ grid != n - 1 || n < 3,
    Return[Failure["PatchGrid", <|"MessageTemplate" -> "Expected at least two patches and one leading CLS token."|>]]];
  a = Developer`ToPackedArray[N[input]];
  rolled = AttentionRollout[a];
  If[FailureQ[rolled], Return[rolled]];
  {l, h} = Take[dims, 2];
  fused = rolled["HeadMeanAttention"]; residual = rolled["ResidualAttention"]; joints = rolled["RolloutByLayer"];
  headRows = a[[All, All, 1, 2 ;;]];
  headMaps = Map[ArrayReshape[#, grid] &, headRows, {2}];
  metrics = Table[
    mass = Total /@ headRows[[i]];
    ent = spatialEntropy /@ headRows[[i]];
    weights = If[Total[mass] > 0, mass/Total[mass], 0. mass];
    weighted = If[Total[mass] > 0, Total[MapThread[If[#1 > 0, #1 #2, 0.] &, {weights, ent}]], Missing["ZeroSpatialMass"]];
    meanEntropy = spatialEntropy[fused[[i, 1, 2 ;;]]];
    rows = spatialNormalize /@ joints[[i, All, 2 ;;]];
    tv = If[Total[joints[[i, 1, 2 ;;]]] > 0, (Total[Abs[# - First[rows]]]/2 &) /@ rows, {Missing["ZeroSpatialMass"]}];
    <|"Layer" -> i, "HeadEntropies" -> ent, "HeadPatchMasses" -> mass, "SpatialHeadMixtureWeights" -> weights,
      "WeightedHeadEntropy" -> weighted, "MeanHeadEntropy" -> meanEntropy,
      "HeadJensenShannon" -> If[NumberQ[meanEntropy], meanEntropy - weighted, Missing["ZeroSpatialMass"]],
      "RolloutEntropy" -> spatialEntropy[joints[[i, 1, 2 ;;]]], "RolloutPatchMass" -> Total[joints[[i, 1, 2 ;;]]],
      "MaxRowTVFromCLS" -> If[VectorQ[tv, NumericQ], Max[tv], First[tv]],
      "MeanRowTVFromCLS" -> If[VectorQ[tv, NumericQ], Mean[tv], First[tv]]|>,
    {i, l}];
  previous = If[l > 1, joints[[-2]], N@IdentityMatrix[n]];
  previousMass = Total /@ previous[[All, 2 ;;]];
  previousRows = spatialNormalize /@ previous[[All, 2 ;;]];
  finalWeights = spatialNormalize[Last[residual][[1]] previousMass];
  finalSpatial = spatialNormalize[Last[joints][[1, 2 ;;]]];
  mixtureError = Max[Abs[finalWeights . previousRows - finalSpatial]];
  maxPreviousTV = If[First[previousMass] > 0, Max[(Total[Abs[# - First[previousRows]]]/2 &) /@ previousRows], Missing["ZeroSpatialMass"]];
  selected = Ordering[-Last[fused][[1, 2 ;;]], Min[2, n - 1]] + 1;
  footprints = Table[<|"Token" -> j, "PatchRow" -> Quotient[j - 2, grid[[2]]] + 1,
    "PatchColumn" -> Mod[j - 2, grid[[2]]] + 1, "LastMeanAttention" -> Last[fused][[1, j]],
    "FinalSpatialMixtureWeight" -> finalWeights[[j]], "PreviousRolloutEntropy" -> spatialEntropy[previous[[j, 2 ;;]]],
    "Map" -> ArrayReshape[previousRows[[j]], grid]|>, {j, selected}];
  <|"LayerCount" -> l, "HeadCount" -> h, "PatchGrid" -> grid, "HeadPatchMaps" -> headMaps,
    "MeanPatchMaps" -> (AttentionPatchMap[#, grid] & /@ fused),
    "RolloutPatchMaps" -> (AttentionPatchMap[#, grid] & /@ joints), "LayerMetrics" -> metrics,
    "PreviousCLSFootprint" -> ArrayReshape[First[previousRows], grid], "SelectedFootprints" -> footprints,
    "FinalMixture" -> <|"MaximumError" -> mixtureError, "WeightSum" -> Total[finalWeights],
      "MaxPreviousRowTVFromCLS" -> maxPreviousTV,
      "FinalTVFromPreviousCLS" -> If[First[previousMass] > 0, Total[Abs[finalSpatial - First[previousRows]]]/2, Missing["ZeroSpatialMass"]],
      "SpatialMixtureWeights" -> finalWeights|>|>
];

headOverlay[crop_Image, map_, range_] := Module[{scaled, large, colors},
  scaled = If[range[[1]] == range[[2]], 0. map, Clip[Rescale[map, range], {0, 1}]];
  large = ImageData@ImageResize[Image[scaled, "Real"], ImageDimensions[crop], Resampling -> "Linear"];
  colors = Image[Map[(List @@ ColorData["TemperatureMap"][#]) &, large, {2}], "Real"];
  ImageCompose[ColorConvert[crop, "RGB"], {colors, .45}]
];
scaledHeadMaps[maps_, scaling_] := Module[{shown, ranges},
  shown = If[scaling == "Conditional", spatialNormalize /@ maps, maps];
  ranges = If[scaling == "Independent", MinMax[Flatten[#]] & /@ shown, ConstantArray[{0., Max[Flatten[shown]]}, Length[shown]]];
  {shown, ranges}
];
scaleCaption["Conditional"] := "Shared scale after conditioning on spatial patches; compare concentration. Patch mass is shown separately.";
scaleCaption["Raw"] := "Shared scale for original weights; CLS is omitted from the picture, without renormalizing patch weights.";
scaleCaption["Independent"] := "Contrast stretched separately in each panel; colors do not compare absolute weights.";

Options[AttentionHeadLayerPlot] = {"Scaling" -> "Conditional"};
AttentionHeadLayerPlot[result_Association, layer_Integer, head_Integer, OptionsPattern[]] := Module[
  {maps, shown, ranges, labels, metric, scaling = OptionValue["Scaling"], crop},
  If[!Between[layer, {1, result["LayerCount"]}] || !Between[head, {1, result["HeadCount"]}], Return[Failure["Index", <||>]]];
  If[!MemberQ[{"Conditional", "Raw", "Independent"}, scaling], Return[Failure["Scaling", <||>]]];
  crop = Lookup[result, "ModelCrop", Missing[]];
  If[!ImageQ[crop], Return[Failure["ModelCrop", <||>]]];
  maps = {result["HeadPatchMaps"][[layer, head]], result["MeanPatchMaps"][[layer]], result["RolloutPatchMaps"][[layer]]};
  {shown, ranges} = scaledHeadMaps[maps, scaling];
  metric = result["LayerMetrics"][[layer]];
  labels = {"Head " <> ToString[head], "Mean of all heads", "Rollout through layer " <> ToString[layer]};
  Column[{
    Style[Lookup[result, "Model", "ViT"] <> " | layer " <> ToString[layer], 16, Bold],
    Grid[{Prepend[labels, "Model crop"], Prepend[MapThread[ImageResize[headOverlay[crop, #1, #2], 230] &, {shown, ranges}], ImageResize[crop, 230]],
      Prepend[("H = " <> displayNumber[spatialEntropy[#]] <> "; patch mass = " <> displayNumber[Total[Flatten[#]]]) & /@ maps, ""]}, Alignment -> {Center, Top}],
    Style[scaleCaption[scaling], 11],
    Style["H is normalized spatial entropy (1 = uniform), not explanation quality. Head indices are local to each block.", 10]
  }]
];

Options[AttentionHeadGrid] = {"Scaling" -> "Conditional"};
AttentionHeadGrid[result_Association, layer_Integer, OptionsPattern[]] := Module[
  {maps, shown, ranges, panels, crop, scaling = OptionValue["Scaling"], heads},
  If[!Between[layer, {1, result["LayerCount"]}], Return[Failure["Index", <||>]]];
  If[!MemberQ[{"Conditional", "Raw", "Independent"}, scaling], Return[Failure["Scaling", <||>]]];
  crop = Lookup[result, "ModelCrop", Missing[]]; If[!ImageQ[crop], Return[Failure["ModelCrop", <||>]]];
  heads = result["HeadCount"];
  maps = Join[result["HeadPatchMaps"][[layer]], {result["MeanPatchMaps"][[layer]], result["RolloutPatchMaps"][[layer]]}];
  {shown, ranges} = scaledHeadMaps[maps, scaling];
  panels = Table[Column[{
    Style[If[i <= heads, "Head " <> ToString[i], If[i == heads + 1, "Head mean", "Rollout"]], 11, Bold],
    ImageResize[headOverlay[crop, shown[[i]], ranges[[i]]], 185],
    Style["H " <> displayNumber[spatialEntropy[maps[[i]]]] <> " | mass " <> displayNumber[Total[Flatten[maps[[i]]]]], 10]
  }, Alignment -> Center], {i, Length[maps]}];
  panels = Join[Take[panels, heads], {Column[{"Model crop", ImageResize[crop, 185]}]}, Drop[panels, heads]];
  Column[{Style[Lookup[result, "Model", "ViT"] <> " | every head in layer " <> ToString[layer], 16, Bold],
    Grid[Partition[PadRight[panels, 4 Ceiling[Length[panels]/4], ""], 4], Alignment -> {Center, Top}], Style[scaleCaption[scaling], 11]}]
];

AttentionHeadLayerExplorer[models_Association] := With[{saved = models, choices = Keys[models],
  layerLimit = Min[Lookup[Values[models], "LayerCount"]], headLimit = Min[Lookup[Values[models], "HeadCount"]]},
  Manipulate[If[view == "Selected head", AttentionHeadLayerPlot[saved[model], layer, head, "Scaling" -> scaling],
    AttentionHeadGrid[saved[model], layer, "Scaling" -> scaling]],
    {{model, First[choices], "Model"}, choices},
    {{layer, layerLimit, "Layer"}, 1, layerLimit, 1}, {{head, 1, "Head"}, 1, headLimit, 1},
    {{view, "Selected head", "View"}, {"Selected head", "All heads"}},
    {{scaling, "Conditional", "Color scale"}, {"Conditional", "Raw", "Independent"}}, SaveDefinitions -> True]
];

AttentionRolloutToy[a_: 2, b_: 10, r_: 0] := Module[{weights, adjusted, states, rolled, coefficients},
  If[!And @@ (NumberQ[#] && Head[#] =!= Complex && TrueQ[-Infinity < # < Infinity] & /@ {a, b, r}) || r < 0,
    Return[Failure["ToyParameters", <||>]]];
  weights = {{{1, 0, 0}, {0, 0, 1}, {0, 0, 1}}, {{0, 1, 0}, {0, 1, 0}, {0, 0, 1}}};
  adjusted = (# + r IdentityMatrix[3])/(1 + r) & /@ weights;
  states = FoldList[#2 . #1 &, {0, a, b}, adjusted];
  rolled = AttentionRollout[weights, "ResidualWeight" -> r];
  coefficients = First[rolled["Rollout"]];
  <|"AttentionWeights" -> weights, "AdjustedWeights" -> adjusted, "States" -> states,
    "ResidualWeight" -> r, "FinalRawCLSWeights" -> First[Last[weights]],
    "InputCoefficients" -> coefficients, "ValueContributions" -> coefficients {0, a, b},
    "FinalCLS" -> Last[states][[1]], "ProductError" -> Max[Abs[Last[adjusted] . First[adjusted] - rolled["Rollout"]]]|>
];

AttentionRolloutToyExplorer[] := Manipulate[
  Module[{d = AttentionRolloutToy[a, b, r]}, Column[{
    Style["A two-hop route: input B to intermediate A to final CLS", 16, Bold],
    Grid[Prepend[MapThread[Prepend, {Transpose[d["States"]], {"CLS", "A", "B"}}], {"Token", "Input", "After layer 1", "After layer 2"}], Frame -> All],
    Row[{"Final attention to intermediate tokens: ", d["FinalRawCLSWeights"]}],
    Row[{"Rollout coefficients on original {CLS,A,B}: ", d["InputCoefficients"]}],
    Row[{"Coefficient times input value: ", d["ValueContributions"], "; sum = ", d["FinalCLS"]}],
    Style["The matrices are fixed. Value projections are identity; there are no MLPs or normalization layers. Residual mixing is (A + r I)/(1+r).", 11]
  }]], {{a, 2, "Input A"}, 0, 12, 1}, {{b, 10, "Input B"}, 0, 12, 1}, {{r, 0, "Residual weight"}, {0, 1, 2}}, SaveDefinitions -> True
];

End[];
EndPackage[];
