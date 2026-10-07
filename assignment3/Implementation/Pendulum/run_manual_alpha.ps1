param(
    [Parameter(Mandatory=$true)]
    [double]$Angle,
    [Parameter(Mandatory=$true)]
    [string]$AlphaValues,
    [Parameter(Mandatory=$true)]
    [string]$Seeds,
    [string]$Config = "config.json"
)
$alphas = $AlphaValues -split ','
$seedList = $Seeds -split ','
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "MANUAL ALPHA SAC TRAINING - PARALLEL EXECUTION" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Target Angle: $Angle°"
Write-Host "Alpha Values: $AlphaValues"
Write-Host "Seeds: $Seeds"
Write-Host "Config: $Config"
Write-Host "Total Experiments: $($alphas.Count * $seedList.Count)"
Write-Host "================================================================================`n" -ForegroundColor Cyan
$jobs = @()
foreach ($alpha in $alphas) {
    foreach ($seed in $seedList) {
        $jobName = "Alpha_${alpha}_Seed_${seed}"
        Write-Host "Starting: Alpha=$alpha, Seed=$seed" -ForegroundColor Green
        $scriptBlock = {
            param($angle, $alpha, $seed, $config)
            python train.py --config $config --target_angle $angle --alpha $alpha --seed $seed
        }
        $job = Start-Job -Name $jobName -ScriptBlock $scriptBlock -ArgumentList $Angle, $alpha, $seed, $Config
        $jobs += $job
    }
}
Write-Host "`nAll jobs started. Waiting for completion..." -ForegroundColor Yellow
Write-Host "You can monitor progress with: Get-Job`n"
$jobs | Wait-Job | Out-Null
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "ALL EXPERIMENTS COMPLETED" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
foreach ($job in $jobs) {
    $status = if ($job.State -eq "Completed") { "SUCCESS" } else { "FAILED" }
    $color = if ($job.State -eq "Completed") { "Green" } else { "Red" }
    Write-Host "$($job.Name): $status" -ForegroundColor $color
}
$jobs | Remove-Job
Write-Host "`nResults saved in: outputs/`n" -ForegroundColor Cyan
