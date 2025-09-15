# Recover HF or LFS cached models locally (no internet)
$dst = Join-Path $PWD "models"
New-Item -ItemType Directory -Force -Path $dst | Out-Null

$hfRoots = @("$env:USERPROFILE\.cache\huggingface\hub", "$env:LOCALAPPDATA\huggingface\hub")
$want = @{
  "distilbert-base-uncased" = @("models--distilbert-base-uncased")
  "all-MiniLM-L6-v2"        = @("models--sentence-transformers--all-MiniLM-L6-v2")
  "layoutlmv3-base"         = @("models--microsoft--layoutlmv3-base")
}

$found = $false
foreach ($root in $hfRoots) {
  if (-not (Test-Path $root)) { continue }
  Get-ChildItem -Path $root -Recurse -Directory -ErrorAction SilentlyContinue |
    ForEach-Object {
      foreach ($kv in $want.GetEnumerator()) {
        if ($_.Name -eq $kv.Value[0]) {
          $out = Join-Path $dst $kv.Key
          New-Item -ItemType Directory -Force -Path $out | Out-Null
          robocopy $_.FullName $out /E | Out-Null
          Write-Host "Recovered $($kv.Key) -> $out"
          $found = $true
        }
      }
    }
}

# Try Git LFS local store -> rehydrate working tree if objects exist
if (Test-Path ".git\lfs\objects") {
  git lfs checkout | Out-Null
}

if ($found) { Write-Host "Local model recovery complete." } else { Write-Host "No local caches found. Proceeding without BERT." }