param(
    [string]$ReviewEdits = '',
    [string]$ValidationPython = '',
    [string]$LatexDirectory = 'D:\MiKTeX\miktex\bin\x64'
)

# Apply a reviewed exact-text patch, regenerate tables, compile, and check fidelity.
# Scientific evidence and author declarations are checked against pre_review/.
$ErrorActionPreference = 'Stop'
$paperDirectory = $PSScriptRoot
$projectDirectory = [System.IO.Path]::GetFullPath((Join-Path $paperDirectory '..\..'))
$projectPython = Join-Path $projectDirectory '.venv\Scripts\python.exe'
if (-not $ValidationPython) {
    $ValidationPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
}
$pdflatex = Join-Path $LatexDirectory 'pdflatex.exe'
$bibtex = Join-Path $LatexDirectory 'bibtex.exe'
foreach ($requiredProgram in @($projectPython, $ValidationPython, $pdflatex, $bibtex)) {
    if (-not (Test-Path -LiteralPath $requiredProgram)) {
        throw "Required program not found: $requiredProgram"
    }
}

if (-not $ReviewEdits) {
    $latest = Get-ChildItem -LiteralPath (Join-Path $paperDirectory 'adversarial\revision') -Filter 'round_*_edits.json' -File |
        Sort-Object Name | Select-Object -Last 1
    if ($latest) { $ReviewEdits = $latest.FullName }
}
if ($ReviewEdits) { $ReviewEdits = [System.IO.Path]::GetFullPath($ReviewEdits) }

function Invoke-CheckedProgram {
    param([string]$Program, [string[]]$ProgramArguments, [string]$LogPath)
    & $Program @ProgramArguments 2>&1 | Out-File -LiteralPath $LogPath -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed. See log: $LogPath"
    }
}

$runName = 'build_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
$runDirectory = Join-Path $paperDirectory ('adversarial\builds\' + $runName)
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $paperDirectory 'draft.tex.in') -Destination (Join-Path $runDirectory 'draft_before.tex.in')
Copy-Item -LiteralPath (Join-Path $paperDirectory 'build_evidence.py') -Destination (Join-Path $runDirectory 'build_evidence_before.py')
Push-Location -LiteralPath $paperDirectory
try {
    if ($ReviewEdits) {
        Invoke-CheckedProgram -Program $projectPython -ProgramArguments @(
            (Join-Path $paperDirectory 'adversarial\revision\apply_review_edits.py'),
            '--manifest', $ReviewEdits
        ) -LogPath (Join-Path $runDirectory 'apply_edits.txt')
    }
    Invoke-CheckedProgram -Program $projectPython -ProgramArguments @('build_evidence.py') -LogPath (Join-Path $runDirectory 'build_evidence.txt')
    Invoke-CheckedProgram -Program $ValidationPython -ProgramArguments @('validate_review_cycle.py', '--stage', 'source') -LogPath (Join-Path $runDirectory 'source_validation.txt')
    $latexArguments = @('--disable-installer', '-interaction=nonstopmode', '-halt-on-error', '-no-shell-escape', 'main.tex')
    Invoke-CheckedProgram -Program $pdflatex -ProgramArguments $latexArguments -LogPath (Join-Path $runDirectory 'latex_1.txt')
    Invoke-CheckedProgram -Program $bibtex -ProgramArguments @('main') -LogPath (Join-Path $runDirectory 'bibtex.txt')
    Invoke-CheckedProgram -Program $pdflatex -ProgramArguments $latexArguments -LogPath (Join-Path $runDirectory 'latex_2.txt')
    Invoke-CheckedProgram -Program $pdflatex -ProgramArguments $latexArguments -LogPath (Join-Path $runDirectory 'latex_3.txt')
    Invoke-CheckedProgram -Program $ValidationPython -ProgramArguments @('validate_review_cycle.py', '--stage', 'pdf') -LogPath (Join-Path $runDirectory 'pdf_validation.txt')
    $outputDirectory = Join-Path $projectDirectory 'output\pdf'
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $paperDirectory 'main.pdf') -Destination (Join-Path $outputDirectory 'paa_reviewed.pdf')
    Write-Output 'Manuscript rebuilt; equations, table values, references and evidence checks passed.'
    Write-Output 'Independent style review and rendered-page inspection remain separate from compilation.'
    Write-Output ('Build record: ' + $runDirectory)
}
finally {
    Pop-Location
}
