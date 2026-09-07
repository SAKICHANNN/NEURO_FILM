# User-authorized exact scratch cleanup; never scan-select data for deletion.
[CmdletBinding()]
param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$root = 'P:\neuro_film_storage\tmp'
$names = @(
 'cleanup_pending_current_product_smoke_4ca3c039b80b4ce2a34d17ba6c930a57',
 'cleanup_pending_private_product_runtime_05659408',
 'cleanup_pending_private_product_runtime_5853b983',
 'cleanup_pending_private_product_runtime_7a239b8f5',
 'cleanup_pending_private_product_runtime_89358324',
 'cleanup_pending_private_product_runtime_9bb7453af',
 'cleanup_pending_private_product_runtime_cd0281d87',
 'cleanup_pending_private_product_runtime_db5a0dacf',
 'cleanup_pending_private_product_runtime_staged_9bb7453af',
 'p282_source_lock_v1_4_2'
)
$rows = @()
$processes = @(Get-CimInstance Win32_Process)
foreach ($name in $names) {
 $path = [IO.Path]::GetFullPath((Join-Path $root $name))
 if (-not $path.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Path escape' }
 if (-not (Test-Path -LiteralPath $path)) { continue }
 $entry = Get-Item -LiteralPath $path -Force
 if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse root: $path" }
 $items = @(Get-ChildItem -LiteralPath $path -Recurse -Force)
 if (@($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw "Nested reparse: $path" }
 if (@($processes | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -and $_.CommandLine.Contains($name) }).Count) { throw "Live process reference: $name" }
 $files = @($items | Where-Object { -not $_.PSIsContainer })
 $rows += [pscustomobject]@{Path=$path; Files=$files.Count; Bytes=[long](($files | Measure-Object Length -Sum).Sum); Removed=$false}
}
$rows | Format-Table -AutoSize
if (-not $Apply) { return }
$repo = Split-Path $PSScriptRoot -Parent
$out = Join-Path $repo 'outputs/storage_cleanup_20260907_filewise'
if (Test-Path -LiteralPath $out) { throw 'Report directory already exists' }
New-Item -ItemType Directory -Path $out | Out-Null
$before = (Get-PSDrive P).Free
$rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $out 'before.json') -Encoding utf8
foreach ($row in $rows) {
 # Preserve filesystem-error directories rather than retrying or repairing disk.
 $errorsSeen = @()
 $items = @(Get-ChildItem -LiteralPath $row.Path -Recurse -Force)
 foreach ($item in @($items | Where-Object { -not $_.PSIsContainer })) {
  try { Remove-Item -LiteralPath $item.FullName -Force } catch { $errorsSeen += $_.Exception.Message }
 }
 foreach ($dir in @($items | Where-Object { $_.PSIsContainer } | Sort-Object { $_.FullName.Length } -Descending)) {
  if (@(Get-ChildItem -LiteralPath $dir.FullName -Force).Count -eq 0) {
   try { Remove-Item -LiteralPath $dir.FullName -Force } catch { $errorsSeen += $_.Exception.Message }
  }
 }
 if (@(Get-ChildItem -LiteralPath $row.Path -Force).Count -eq 0) {
  try { Remove-Item -LiteralPath $row.Path -Force } catch { $errorsSeen += $_.Exception.Message }
 }
 $row.Removed = -not (Test-Path -LiteralPath $row.Path)
 $remaining = @(Get-ChildItem -LiteralPath $row.Path -Recurse -File -Force -ErrorAction SilentlyContinue)
 $row | Add-Member -NotePropertyName RemainingFiles -NotePropertyValue $remaining.Count
 $row | Add-Member -NotePropertyName RemainingBytes -NotePropertyValue ([long](($remaining | Measure-Object Length -Sum).Sum))
 $row | Add-Member -NotePropertyName Errors -NotePropertyValue $errorsSeen
 $rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $out 'progress.json') -Encoding utf8
 Write-Output "Processed $($row.Path); remaining files=$($row.RemainingFiles), errors=$($errorsSeen.Count)"
}
$report = [ordered]@{BeforeFreeBytes=$before; AfterFreeBytes=(Get-PSDrive P).Free; LogicalBytes=[long](($rows | Measure-Object Bytes -Sum).Sum); Rows=$rows; Scope='Exact stale runtime/quarantined acquisition scratch only; data/models/evidence retained'}
$report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $out 'result.json') -Encoding utf8
$report | ConvertTo-Json -Depth 5
