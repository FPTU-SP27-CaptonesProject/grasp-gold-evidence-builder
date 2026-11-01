param(
    [Parameter(Mandatory = $true)][string]$QaFile,
    [string]$RetrievalRoot = "../hf-artifacts/grasp-retrieval",
    [string]$RetrievalRevision,
    [string]$RawRoot = "raw",
    [string]$OutputDir = "work/exact-grpo",
    [string]$PythonExecutable,
    [int]$ExpectedTrainRows = 9000,
    [switch]$Download
)
$ErrorActionPreference = "Stop"
$BuilderRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $BuilderRoot
if (-not $PythonExecutable) {
    $LocalPython = Join-Path $BuilderRoot ".venv/Scripts/python.exe"
    $PythonExecutable = if (Test-Path -LiteralPath $LocalPython) { $LocalPython } else { "python" }
}
$QaPath = (Resolve-Path -LiteralPath $QaFile).Path
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program (exit $LASTEXITCODE)" }
}
if ($Download) {
    foreach ($Dataset in @("hotpotqa", "2wiki", "nq")) {
        Invoke-Checked $PythonExecutable @("-m", "grasp_gold_evidence_builder.download", "--dataset", $Dataset, "--split", "all", "--raw-root", $RawRoot)
    }
}
$PipelineArguments = @(
    "-m", "grasp_gold_evidence_builder.pipeline", "--qa", $QaPath,
    "--raw-root", $RawRoot, "--retrieval-root", $RetrievalRoot,
    "--output-dir", $OutputDir, "--expected-train-rows", "$ExpectedTrainRows"
)
if ($RetrievalRevision) { $PipelineArguments += @("--retrieval-revision", $RetrievalRevision) }
Invoke-Checked $PythonExecutable $PipelineArguments
Write-Host "Validated evidence files: $((Resolve-Path -LiteralPath $OutputDir).Path)"
Write-Host "Next: grasp-gold-package. Real SFT rollout smoke is still required before primary GRPO."
