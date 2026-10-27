[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RunRoot,
    [Parameter(Mandatory = $true)][string]$PuzzleId,
    [string]$Captures,
    [string]$ExistingStage1,
    [ValidateSet('flagged', 'all', 'none')][string]$Review = 'flagged',
    [ValidateSet('cuda', 'cpu')][string]$Stage3Device = 'cuda',
    [string]$Stage4Config,
    [switch]$Resume,
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$desktopRoot = Split-Path -Parent $workspaceRoot
$stage1Root = Join-Path $desktopRoot 'paper-segmentation-stage'
$stage2Root = Join-Path $desktopRoot 'paper-reconstruction-stage2'
$stage3Root = Join-Path $desktopRoot 'paper-reconstruction-stage3'
$resolvedRunRoot = [IO.Path]::GetFullPath($RunRoot)
if (-not $Stage4Config) { $Stage4Config = Join-Path $workspaceRoot 'configs\hybrid_development.json' }
if ((-not $ExistingStage1) -eq (-not $Captures)) {
    throw 'Choose exactly one of -Captures (one page of physical pieces) or -ExistingStage1 (accepted output for one page).'
}
if ($ExistingStage1) {
    $stage1Output = (Resolve-Path -LiteralPath $ExistingStage1).Path
    if (-not (Test-Path -LiteralPath (Join-Path $stage1Output 'segmentation_manifest.jsonl'))) {
        throw 'ExistingStage1 has no segmentation_manifest.jsonl.'
    }
} else {
    $Captures = (Resolve-Path -LiteralPath $Captures).Path
    $stage1Output = Join-Path $resolvedRunRoot 'stage1'
}
$stage2Output = Join-Path $resolvedRunRoot 'stage2'
$stage3Output = Join-Path $resolvedRunRoot 'stage3'
$stage4Output = Join-Path $resolvedRunRoot 'stage4'
$commands = [System.Collections.Generic.List[object]]::new()
if (-not $ExistingStage1) {
    $commands.Add(@{Name='Stage 1 segmentation'; Python=(Join-Path $stage1Root '.venv\Scripts\python.exe'); Arguments=@('-B', (Join-Path $stage1Root 'scripts\segment_folder.py'), $Captures, $stage1Output, '--review', $Review)})
}
$stage2Arguments = @('-B', (Join-Path $stage2Root 'scripts\run_stage2_inference.py'), '--input', $stage1Output, '--output', $stage2Output, '--release', (Join-Path $stage2Root 'models\release.json'))
if ($Resume) { $stage2Arguments += '--resume' }
$commands.Add(@{Name='Stage 2 boundary evidence'; Python=(Join-Path $stage2Root '.venv\Scripts\python.exe'); Arguments=$stage2Arguments})
$commands.Add(@{Name='Stage 2 verification'; Python=(Join-Path $stage2Root '.venv\Scripts\python.exe'); Arguments=@('-B', (Join-Path $stage2Root 'scripts\verify_stage2_output.py'), '--input', $stage2Output, '--release', (Join-Path $stage2Root 'models\release.json'))})
$commands.Add(@{Name='Stage 3 frozen alternatives'; Python=(Join-Path $stage3Root '.venv\Scripts\python.exe'); Arguments=@('-B', (Join-Path $stage3Root 'scripts\stage3.py'), 'infer', '--stage1', $stage1Output, '--stage2', $stage2Output, '--output', $stage3Output, '--models', (Join-Path $stage3Root 'models'), '--device', $Stage3Device, '--puzzle-id', $PuzzleId)})
$stage4Arguments = @('-m', 'paper_reconstruction_stage4', 'solve', '--bundle', $stage3Output, '--config', $Stage4Config, '--output', $stage4Output)
if ($Resume) { $stage4Arguments += '--resume' }
$commands.Add(@{Name='Stage 4 global assembly'; Python=(Join-Path $workspaceRoot '.venv\Scripts\python.exe'); Arguments=$stage4Arguments})
foreach ($command in $commands) {
    if (-not (Test-Path -LiteralPath $command.Python)) { throw ('Missing Python runtime: ' + $command.Python) }
}
$request = @{Schema='stage4.local-pipeline-request.v1'; PuzzleId=$PuzzleId; Captures=$Captures; ExistingStage1=$ExistingStage1; Commands=$commands}
if ($PlanOnly) {
    $request | ConvertTo-Json -Depth 7
    return
}
if ((Test-Path -LiteralPath $resolvedRunRoot) -and -not $Resume) {
    throw 'RunRoot already exists; choose a fresh directory or use -Resume with unchanged inputs/configuration.'
}
New-Item -ItemType Directory -Path $resolvedRunRoot -Force | Out-Null
$requestPath = Join-Path $resolvedRunRoot 'pipeline_request.json'
$request.Stage3Device = $Stage3Device
$request.Stage4ConfigurationSha256 = (Get-FileHash -LiteralPath $Stage4Config -Algorithm SHA256).Hash.ToLowerInvariant()
$request.CaptureFiles = @()
if ($Captures) {
    $request.CaptureFiles = @(Get-ChildItem -LiteralPath $Captures -File -Recurse | Sort-Object FullName | ForEach-Object {
        [ordered]@{Path=$_.FullName; Sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
    })
}
$captureIdentity = $request.CaptureFiles | ConvertTo-Json -Depth 4 -Compress
$captureHasher = [Security.Cryptography.SHA256]::Create()
try {
    $request.CaptureFilesIdentity = ([BitConverter]::ToString($captureHasher.ComputeHash([Text.Encoding]::UTF8.GetBytes([string]$captureIdentity)))).Replace('-', '').ToLowerInvariant()
} finally {
    $captureHasher.Dispose()
}
if (Test-Path -LiteralPath $requestPath) {
    $previousRequest = Get-Content -LiteralPath $requestPath -Raw | ConvertFrom-Json
    foreach ($identityField in @('PuzzleId', 'Captures', 'ExistingStage1', 'Stage3Device', 'Stage4ConfigurationSha256', 'CaptureFilesIdentity')) {
        if ([string]$previousRequest.$identityField -cne [string]$request[$identityField]) {
            throw ('Pipeline resume identity changed: ' + $identityField + '. Choose a new run directory.')
        }
    }
}
if (-not (Test-Path -LiteralPath $requestPath)) {
    $request | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $requestPath -Encoding utf8
}
foreach ($command in $commands) {
    Write-Host $command.Name
    $skipInference = $false
    if ($command.Name -eq 'Stage 1 segmentation' -and $Resume) {
        $stage1Commit = Join-Path $resolvedRunRoot 'stage1_commit.json'
        if (Test-Path -LiteralPath $stage1Commit) {
            $committed = Get-Content -LiteralPath $stage1Commit -Raw | ConvertFrom-Json
            $actual = (Get-FileHash -LiteralPath (Join-Path $stage1Output 'segmentation_manifest.jsonl') -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -cne $committed.ManifestSha256) { throw 'Completed Stage 1 manifest changed. Use a new run directory.' }
            $skipInference = $true
        }
    }
    if ($command.Name -eq 'Stage 2 boundary evidence' -and $Resume) {
        $stage2Manifest = Join-Path $stage2Output 'edge_hypotheses_manifest.json'
        if (Test-Path -LiteralPath $stage2Manifest) {
            $completed = Get-Content -LiteralPath $stage2Manifest -Raw | ConvertFrom-Json
            $stage1Hash = (Get-FileHash -LiteralPath (Join-Path $stage1Output 'segmentation_manifest.jsonl') -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($completed.status -cne 'complete' -or $completed.stage1_manifest_sha256 -cne $stage1Hash) { throw 'Completed Stage 2 output has a different Stage 1 binding.' }
            $stage2Commit = Join-Path $resolvedRunRoot 'stage2_commit.json'
            if (Test-Path -LiteralPath $stage2Commit) {
                $committed = Get-Content -LiteralPath $stage2Commit -Raw | ConvertFrom-Json
                $actual = (Get-FileHash -LiteralPath $stage2Manifest -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($actual -cne $committed.ManifestSha256) { throw 'Completed Stage 2 manifest changed. Use a new run directory.' }
            }
            $skipInference = $true
        }
    }
    if ($skipInference) {
        Write-Host 'Reusing completed inference to preserve downstream receipt identity.'
        continue
    }
    $pythonExecutable = $command.Python
    $argumentsForStage = $command.Arguments
    & $pythonExecutable @argumentsForStage
    if ($LASTEXITCODE -ne 0) { throw ($command.Name + ' stopped with exit code ' + $LASTEXITCODE + '. Published outputs and committed upstream caches are preserved.') }
    if ($command.Name -eq 'Stage 1 segmentation') {
        @{ManifestSha256=(Get-FileHash -LiteralPath (Join-Path $stage1Output 'segmentation_manifest.jsonl') -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resolvedRunRoot 'stage1_commit.json') -Encoding utf8
    }
    if ($command.Name -eq 'Stage 2 verification') {
        @{ManifestSha256=(Get-FileHash -LiteralPath (Join-Path $stage2Output 'edge_hypotheses_manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resolvedRunRoot 'stage2_commit.json') -Encoding utf8
    }
}
Write-Host ('Completed observable pipeline. Inspect: ' + (Join-Path $stage4Output 'preview.png'))
Write-Host 'Accuracy evaluation is a separate operation after prediction commitment.'
