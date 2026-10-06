# Start only this project's services in hidden background windows.
# Press Enter to stop the two processes owned by this script.
param([ValidateSet('demo', 'video', 'real')][string]$Mode = 'demo')
$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$pythonExecutable = Join-Path $projectDirectory '.venv\Scripts\python.exe'
$frontendDirectory = Join-Path $projectDirectory 'frontend'
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw 'Create .venv and install backend dependencies first. See README.md.' }
if (-not (Test-Path -LiteralPath (Join-Path $frontendDirectory 'node_modules'))) { throw 'Run npm ci in frontend first.' }
foreach ($port in @(8000, 5173)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is already in use. Stop the existing service before running this script."
    }
}
$dataDirectory = Join-Path $projectDirectory 'data'
New-Item -ItemType Directory -Path $dataDirectory -Force | Out-Null
$previousMode = $env:EGG_MODE
$env:EGG_MODE = $Mode
$backendProcess = $null
$frontendProcess = $null
try {
    $backendProcess = Start-Process -FilePath $pythonExecutable -ArgumentList '-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8000','--workers','1' -WorkingDirectory $projectDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $dataDirectory 'backend.log') -RedirectStandardError (Join-Path $dataDirectory 'backend-error.log')
    # Launch node directly so Stop-Process targets the server, not an npm shell.
    $nodeExecutable = (Get-Command node.exe).Source
    $viteScript = Join-Path $frontendDirectory 'node_modules\vite\bin\vite.js'
    $quotedViteScript = '"' + $viteScript + '"'
    $frontendProcess = Start-Process -FilePath $nodeExecutable -ArgumentList $quotedViteScript,'--host','127.0.0.1' -WorkingDirectory $frontendDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $dataDirectory 'frontend.log') -RedirectStandardError (Join-Path $dataDirectory 'frontend-error.log')
    Write-Host "OVO $Mode : http://127.0.0.1:5173 · API: http://127.0.0.1:8000/docs"
    Read-Host 'Press Enter to stop the services' | Out-Null
} finally {
    foreach ($serviceProcess in @($backendProcess, $frontendProcess)) {
        if ($null -ne $serviceProcess -and -not $serviceProcess.HasExited) {
            Stop-Process -Id $serviceProcess.Id -ErrorAction SilentlyContinue
        }
    }
    $env:EGG_MODE = $previousMode
}
