param(
    [string]$patient = "Patient_001",
    [double]$sigma = 0.035,
    [double]$rhoGM = 800,
    [int]$nsamples = 100,
    [int]$max_iter = 20,
    [switch]$all = $false
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

if ($all) {
    $patientList = @("Patient_001", "Patient_002", "Patient_003", "Patient_004", "Patient_005", 
                     "Patient_006", "Patient_007", "Patient_008", "Patient_009", "Patient_010")
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "Running Recursive Glioma Inference for 10-Patient Cohort" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    # Normalize patient name if single digit or 2 digit
    $pNorm = $patient
    if ($patient -match "^Patient_(\d+)$") {
        $num = [int]$matches[1]
        $pNorm = "Patient_{0:D3}" -f $num
    }
    $patientList = @($pNorm)
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "Running Recursive Glioma Inference for: $pNorm" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan
}

foreach ($p in $patientList) {
    Write-Host "`n>>> Processing Patient: $p (max_iter=$max_iter, nsamples=$nsamples) <<<" -ForegroundColor Yellow
    docker run --rm -v "${scriptDir}:/workspace" -w /workspace/BayesianInference hippylib/fenics `
      "python3 -u patient_hyper.py --patient $p --sigma $sigma --rhoGM $rhoGM --nsamples $nsamples --max_iter $max_iter"
}

Write-Host "`nAll requested runs complete! Check results/ for outputs." -ForegroundColor Green
