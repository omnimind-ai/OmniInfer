param(
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [ValidateRange(0, 86400)][int]$MaxSamples = 0
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Cooked utilization covers an interval. The first start is deliberately unknown;
# a consumer must not infer it by subtracting the requested sampling period.
$previousTimestamp = $null
$counterArgs = @{ Counter = '\GPU Engine(*)\Utilization Percentage'; SampleInterval = 1 }
if ($MaxSamples -eq 0) { $counterArgs.Continuous = $true }
else { $counterArgs.MaxSamples = $MaxSamples }
Get-Counter @counterArgs |
    ForEach-Object {
        $currentTimestamp = $_.Timestamp.ToUniversalTime().ToString('o')
        $samples = @($_.CounterSamples | ForEach-Object {
            $counterPid = $null
            if ($_.Path -match 'pid_(\d+)_') { $counterPid = [long]$Matches[1] }
            @{
                pid = $counterPid
                path = $_.Path
                utilization = $_.CookedValue
                status = $_.Status
            }
        })
        $row = @{
            interval_start = $previousTimestamp
            interval_end = $currentTimestamp
            samples = $samples
            processes = @(Get-Process | Select-Object Id, ProcessName, CPU, WorkingSet64)
        }
        [IO.File]::AppendAllText(
            $OutputPath, ($row | ConvertTo-Json -Depth 5 -Compress) + [Environment]::NewLine,
            [Text.UTF8Encoding]::new($false))
        $previousTimestamp = $currentTimestamp
    }
