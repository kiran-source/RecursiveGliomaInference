param(
    [double]$sigma = 0.035,
    [double]$rhoGM = 800,
    [int]$nsamples = 500,
    [int]$max_iter = 200
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check if Docker is actively running
$dockerCheck = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Docker Desktop is NOT currently running!" -ForegroundColor Red
    Write-Host "Please do the following:" -ForegroundColor Yellow
    Write-Host "  1. Open the Windows Start Menu." -ForegroundColor Yellow
    Write-Host "  2. Search for 'Docker Desktop' and launch it." -ForegroundColor Yellow
    Write-Host "  3. Wait ~20 seconds until Docker Desktop says 'Engine running' (green icon in bottom-left)." -ForegroundColor Yellow
    Write-Host "  4. Re-run this script.`n" -ForegroundColor Yellow
    exit 1
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Recursive Glioma Inference (FEniCS + hIPPYlib)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Parameters:" -ForegroundColor Yellow
Write-Host "  sigma    : $sigma"
Write-Host "  rhoGM    : $rhoGM"
Write-Host "  nsamples : $nsamples"
Write-Host "  max_iter : $max_iter"
Write-Host "==========================================================" -ForegroundColor Cyan

docker run --rm -v "${scriptDir}:/workspace" -w /workspace/BayesianInference hippylib/fenics "python3 Rat-3_hyper.py --sigma $sigma --rhoGM $rhoGM --nsamples $nsamples --max_iter $max_iter"
