root = DirectoryName[DirectoryName[$InputFileName]];
out = Environment["ATTENTION_ROLLOUT_OUTPUT"];
$AllowInternet = False;
Get[FileNameJoin[{root, "Reproduce.wl"}]];
cases = {"samoyed", "samoyed-option-4", "samoyed-option-5", "samoyed-option-1"};
results = Table[
  result = ReproduceAttention[id];
  If[FailureQ[result], <|"ID" -> id, "Passed" -> False, "Failure" -> ToString[result, InputForm]|>,
    <|"ID" -> id, "Passed" -> (Dimensions[result["HeadMean"]] === {14, 14} &&
      Dimensions[result["Rollout"]] === {14, 14} && result["MaximumMapError"] < 10^-12),
      "MaximumMapError" -> result["MaximumMapError"], "Top3" -> result["Top3"]|>],
  {id, cases}];
passed = And @@ Lookup[results, "Passed"];
Export[FileNameJoin[{out, "result.json"}], <|"Success" -> passed, "Version" -> $Version, "Examples" -> results|>, "RawJSON"];
Quit[If[passed, 0, 1]];
