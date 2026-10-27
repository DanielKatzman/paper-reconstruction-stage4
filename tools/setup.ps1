param(
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cpu',
    [string]$TorchIndexUrl = '',
    [string]$TorchPackageVersion = ''
)
$ErrorActionPreference = 'Stop'
if (-not $TorchIndexUrl) {
    $TorchIndexUrl = if ($Device -eq 'cpu') { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu118' }
}
if (-not $TorchPackageVersion) {
    $TorchPackageVersion = if ($Device -eq 'cpu') { '2.7.1+cpu' } else { '2.7.1+cu118' }
}
$Repo = Split-Path -Parent $PSScriptRoot
$Runtimes = @{}
Push-Location $Repo
try {
    foreach ($Stage in @('stage1', 'stage2', 'stage3', 'stage4')) {
        $Version = if ($Stage -in @('stage3','stage4')) { '-3.12' } else { '-3.14' }
        $EnvironmentPath = Join-Path $Repo "local/envs/$Stage"
        $Executable = Join-Path $EnvironmentPath 'Scripts/python.exe'
        if (-not (Test-Path -LiteralPath $Executable)) {
            & py $Version -m venv $EnvironmentPath
            if ($LASTEXITCODE -ne 0) { throw "Creating $Stage environment failed" }
        }
        & $Executable -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed' }
        if ($Stage -eq 'stage1') { $Package = './stages/stage1[runtime]' }
        elseif ($Stage -eq 'stage2') { $Package = './stages/stage2[inference]' }
        elseif ($Stage -eq 'stage3') {
            & $Executable -m pip install "torch==$TorchPackageVersion" --index-url $TorchIndexUrl
            if ($LASTEXITCODE -ne 0) { throw 'Torch install failed' }
            & $Executable -m pip install -e ./contracts/sdk -e './stages/stage3[test]' -c ./stages/stage3/constraints.txt
            if ($LASTEXITCODE -ne 0) { throw 'Stage 3 install failed' }
            $Runtimes[$Stage] = $Executable
            continue
        } else {
            & $Executable -m pip install -e ./contracts/sdk -e ./stages/stage4 -c ./stages/stage4/constraints.txt
            if ($LASTEXITCODE -ne 0) { throw 'Stage 4 install failed' }
            $Runtimes[$Stage] = $Executable
            continue
        }
        & $Executable -m pip install -e $Package
        if ($LASTEXITCODE -ne 0) { throw "$Stage install failed" }
        $Runtimes[$Stage] = $Executable
    }
    if ($Device -eq 'cpu') {
        & $Runtimes['stage3'] -B (Join-Path $PSScriptRoot 'check_cpu_runtime.py')
        if ($LASTEXITCODE -ne 0) { throw 'CPU runtime verification failed' }
    }
    $Runtimes | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $Repo 'local/runtimes.json')
} finally { Pop-Location }
