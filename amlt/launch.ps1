<#
  One-shot AMLT launcher: submit jobs, then keep pushing throttled ones through
  until none are left failed.

  Background:
    `amlt run` succeeds at submit time, but Singularity/AML then FAILS jobs
    asynchronously with TooManyRequests ("Queued resource count >= 15000" —
    a shared subscription+region throttle). So this script submits once, then
    loops `amlt rerun` (which reruns only FAILED jobs, replacing them in place)
    with backoff until everything is past the throttle.

  Usage (inside the `amlt` conda env):
    # all jobs in the config
    .\amlt\launch.ps1 amlt\mujoco_walker_ant.yaml baseline_exp

    # only specific jobs (comma-separated, names as in the yaml; ':' optional)
    .\amlt\launch.ps1 amlt\mujoco_walker_ant.yaml baseline_exp -Jobs walker_param_gpt,ant_dir_lstm

    .\amlt\launch.ps1 amlt\mujoco_walker_ant.yaml baseline_exp -Settle 180 -Description "MuJoCo baselines"
#>
param(
  [Parameter(Mandatory=$true)][string]$Config,
  [Parameter(Mandatory=$true)][string]$Exp,
  [string[]]$Jobs = @(),    # specific job names to handle; empty = all jobs in the config
  [string]$Description = "auto-submitted via launch.ps1",
  [int]$Settle     = 300,   # seconds to wait for throttle failures to surface
  [int]$MaxBackoff = 600,   # cap each wait at 10 min
  [int]$MaxRounds  = 500    # safety cap so a genuinely-broken job can't loop forever
)
$ErrorActionPreference = 'Continue'

# Turn -Jobs into amlt ":name" selectors. Empty => no selectors => all jobs.
$Selectors = @($Jobs | ForEach-Object { ":" + ($_ -replace '^:', '') })
$scope = if ($Selectors.Count -gt 0) { $Selectors -join ' ' } else { '(all jobs)' }

function Get-FailedJobs {
  param([string]$exp, [string[]]$selectors)
  $out = (& amlt status $exp @selectors -s failed --no-calculate-result-size --hide-urls 2>&1 | Out-String)
  $m = [regex]::Matches($out, '(?m)^\s*(:\S+)\s+failed\b')
  return @($m | ForEach-Object { $_.Groups[1].Value })
}

# ---- 1) Submit (-y -d avoids interactive prompts) ----
Write-Host "==> Submitting $scope from '$Config' to experiment '$Exp'..." -ForegroundColor Cyan
& amlt run $Config @Selectors $Exp -y -d $Description

Write-Host "==> Waiting ${Settle}s for the first throttle wave to surface..." -ForegroundColor Cyan
Start-Sleep -Seconds $Settle

# ---- 2) Rerun failed jobs until none remain failed ----
$backoff   = $Settle
$prevCount = [int]::MaxValue
for ($round = 1; $round -le $MaxRounds; $round++) {
  $failed = Get-FailedJobs -exp $Exp -selectors $Selectors
  $n = $failed.Count
  Write-Host ("[round {0}] {1} failed job(s)" -f $round, $n)
  if ($n -gt 0) { Write-Host ("           {0}" -f ($failed -join ', ')) }

  if ($n -eq 0) {
    Write-Host "==> All targeted jobs past the throttle (0 failed). Done." -ForegroundColor Green
    break
  }

  if ($n -lt $prevCount) { $backoff = $Settle }   # progress -> retry sooner
  $prevCount = $n

  # Rerun exactly the jobs we found failed (auto-scopes to -Jobs when given).
  Write-Host "[round $round] amlt rerun $Exp $($failed -join ' ') -y ..."
  & amlt rerun $Exp @failed -y          # replace failed jobs in place

  Write-Host "[round $round] waiting ${backoff}s for jobs to settle / re-fail..."
  Start-Sleep -Seconds $backoff
  $backoff = [Math]::Min([int]($backoff * 1.5), $MaxBackoff)
}

Write-Host ""
Write-Host "==> Final status:" -ForegroundColor Cyan
& amlt status $Exp @Selectors --no-calculate-result-size --hide-urls
