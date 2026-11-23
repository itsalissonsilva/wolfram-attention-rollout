root = DirectoryName[DirectoryName[$InputFileName]];
out = Environment["ATTENTION_ROLLOUT_OUTPUT"];
$Messages = {OpenWrite[FileNameJoin[{out, "messages.txt"}]]};
Get[FileNameJoin[{root, "AttentionRollout.wl"}]];
tests = {};
record[name_, result_] := AppendTo[tests, <|"Name" -> name, "Passed" -> TrueQ[result]|>];
close[a_, b_] := Dimensions[a] === Dimensions[b] && Max[Abs[Flatten[N[a - b]]]] < 10^-12;

(* Noncommuting matrices distinguish correct layer composition from reversed order. *)
a = {{1., 0., 0.}, {0.25, 0.75, 0.}, {0., 0.5, 0.5}};
b = {{0.2, 0.8, 0.}, {0., 1., 0.}, {0.6, 0., 0.4}};
r = AttentionRollout[{a, b}];
record["Residual-aware two-layer composition", close[r["Rollout"], ((b + IdentityMatrix[3])/2) . ((a + IdentityMatrix[3])/2)]];
record["Fixture detects reversed multiplication", !close[r["Rollout"], ((a + IdentityMatrix[3])/2) . ((b + IdentityMatrix[3])/2)]];
record["No-residual composition", close[AttentionRollout[{a, b}, "ResidualWeight" -> 0]["Rollout"], b . a]];
record["Identity remains identity", close[AttentionRollout[ConstantArray[IdentityMatrix[3], {4, 2}]]["Rollout"], IdentityMatrix[3]]];
record["Uniform attention remains uniform without residuals", close[AttentionRollout[ConstantArray[ConstantArray[1/3, {3, 3}], 3], "ResidualWeight" -> 0]["Rollout"], ConstantArray[1/3, {3, 3}]]];
record["Mean-head fusion", close[AttentionRollout[{{a, b}}]["HeadMeanAttention"], {(a + b)/2}]];
record["One layer equals adjusted attention", close[First[r["RolloutByLayer"]], First[r["ResidualAttention"]]]];
record["Every rollout row remains normalized", Max[Abs[Flatten[Map[Total, r["RolloutByLayer"], {2}]] - 1]] < 10^-12];
record["Reject transposed stochastic matrices", FailureQ[AttentionRollout[{Transpose[a]}]]];
record["Reject negative weights", FailureQ[AttentionRollout[{{{1.1, -0.1}, {0., 1.}}}]]];
record["Reject non-square attention", FailureQ[AttentionRollout[{{{1., 0., 0.}, {0., 1., 0.}}}]]];
record["Reject empty input", FailureQ[AttentionRollout[{}]]];
record["Reject invalid residual weight", FailureQ[AttentionRollout[{a}, "ResidualWeight" -> -1]]];
record["Patch order and special-token exclusion", AttentionPatchMap[Partition[Range[25], 5], {2, 2}] === {{2, 3}, {4, 5}}];
record["Reject mismatched patch grid", FailureQ[AttentionPatchMap[a, {2, 2}]]];
record["Reject complex attention", FailureQ[AttentionRollout[{{{1. + I, -I}, {0., 1.}}}]]];
record["Reject nonfinite attention", FailureQ[AttentionRollout[{{{Infinity, 0.}, {0., 1.}}}]]];
record["Reject ragged attention", FailureQ[AttentionRollout[{{{1., 0.}, {1.}}}]]];
large = Developer`ToPackedArray[ConstantArray[1./197, {12, 12, 197, 197}]];
timed = AbsoluteTiming[largeResult = AttentionRollout[large]];
record["B16-sized packed uniform attention", AssociationQ[largeResult] && close[largeResult["Rollout"], ConstantArray[(1. - 2.^-12)/197, {197, 197}] + 2.^-12 IdentityMatrix[197]]];
Export[FileNameJoin[{out, "performance.json"}], <|"Shape" -> Dimensions[large], "RolloutSeconds" -> First[timed]|>, "RawJSON"];

Export[FileNameJoin[{out, "result.json"}], <|"Success" -> And @@ Lookup[tests, "Passed"], "Version" -> $Version, "Passed" -> Count[Lookup[tests, "Passed"], True], "Total" -> Length[tests], "Tests" -> tests|>, "RawJSON"];
Quit[If[And @@ Lookup[tests, "Passed"], 0, 1]];
