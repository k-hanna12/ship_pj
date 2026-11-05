# Use a project virtual environment if present; otherwise use this desktop's Python.
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    $projectPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
}
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Python not found. Create .venv and install requirements-preprocess.txt as described in README.md.'
}
$dependencyDirectory = Join-Path (Split-Path $PSScriptRoot -Parent) '.python-deps'
$previousPythonPath = $env:PYTHONPATH
try {
    if (Test-Path -LiteralPath $dependencyDirectory) {
        $env:PYTHONPATH = "$dependencyDirectory;$previousPythonPath"
    }
    & $projectPython (Join-Path $PSScriptRoot 'preprocess.py') @args
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
