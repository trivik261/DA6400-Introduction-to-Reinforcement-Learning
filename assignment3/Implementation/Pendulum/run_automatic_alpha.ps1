param(
    [Parameter(Mandatory=$true)]
    [string]$Angles,
    [Parameter(Mandatory=$true)]
    [string]$Seeds,
    [string]$Config = "config.json"
)
$angleList = $Angles -split ','
$seedList = $Seeds -split ','
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "AUTOMATIC ALPHA SAC TRAINING - PARALLEL EXECUTION" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Target Angles: $Angles°"
Write-Host "Seeds: $Seeds"
Write-Host "Config: $Config"
Write-Host "Total Experiments: $($angleList.Count * $seedList.Count)"
Write-Host "================================================================================`n" -ForegroundColor Cyan
$jobs = @()
foreach ($angle in $angleList) {
    foreach ($seed in $seedList) {
        $jobName = "Angle_${angle}_Seed_${seed}"
        Write-Host "Starting: Angle=$angle°, Seed=$seed (Automatic Alpha)" -ForegroundColor Green
        $scriptBlock = {
            param($angle, $seed, $config)
            python train.py --config $config --target_angle $angle --seed $seed
        }
        $job = Start-Job -Name $jobName -ScriptBlock $scriptBlock -ArgumentList $angle, $seed, $Config
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
