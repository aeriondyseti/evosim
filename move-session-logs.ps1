#requires -Version 5
<#
.SYNOPSIS
  Relocate this project's Claude Code session transcript(s) from the old repo
  location to the new one, so you can `claude --resume` from inside the moved repo.

.DESCRIPTION
  Claude Code stores per-project history under
      %USERPROFILE%\.claude\projects\<encoded-cwd>\<session-id>.jsonl
  where <encoded-cwd> is the working-directory path with ':' '\' '/' replaced by '-'.

  This repo moved from
      D:\Development\rpstack\evosim-framework   ->   D:\Development\evosim-framework
  so its transcripts must move from the old project key to the new one.

  IMPORTANT: Run this AFTER you have exited the Claude Code session whose logs you
  are moving. The active transcript file is locked while the session is open; the
  script detects the lock and aborts cleanly if so.

.PARAMETER OldRepo
  Old repository path (default: D:\Development\rpstack\evosim-framework)

.PARAMETER NewRepo
  New repository path (default: D:\Development\evosim-framework)

.PARAMETER NoRewritePaths
  Skip rewriting old path references inside the transcript; just move the files.

.EXAMPLE
  pwsh -File .\move-session-logs.ps1
#>
param(
    [string]$OldRepo = 'D:\Development\rpstack\evosim-framework',
    [string]$NewRepo = 'D:\Development\evosim-framework',
    [switch]$NoRewritePaths
)

$ErrorActionPreference = 'Stop'

function Get-ProjectKey([string]$p) {
    # Claude Code encodes the cwd by replacing ':' '\' and '/' with '-'
    return ($p -replace '[:\\/]', '-')
}

# Build git-bash style path: D:\Development\foo -> /d/Development/foo
function Get-BashPath([string]$p) {
    $drive = $p.Substring(0, 1).ToLower()
    $rest  = $p.Substring(2).Replace('\', '/')
    return "/$drive$rest"
}

$projectsRoot = Join-Path $env:USERPROFILE '.claude\projects'
$oldDir = Join-Path $projectsRoot (Get-ProjectKey $OldRepo)
$newDir = Join-Path $projectsRoot (Get-ProjectKey $NewRepo)

Write-Host "Source project dir: $oldDir"
Write-Host "Target project dir: $newDir"
Write-Host ""

if (-not (Test-Path -LiteralPath $oldDir)) {
    Write-Host "Nothing to do: source project dir does not exist."
    return
}

$files = @(Get-ChildItem -LiteralPath $oldDir -Filter *.jsonl -File)
if ($files.Count -eq 0) {
    Write-Host "No .jsonl transcripts found in source dir."
    return
}

# Verify none of the transcripts are locked (i.e. session still open)
foreach ($f in $files) {
    try {
        $fs = [System.IO.File]::Open($f.FullName, 'Open', 'ReadWrite', 'None')
        $fs.Close()
        $fs.Dispose()
    } catch {
        Write-Error "Transcript is locked (is Claude Code still running?): $($f.FullName)`nExit the session, then re-run this script."
        return
    }
}

New-Item -ItemType Directory -Force -Path $newDir | Out-Null

# Precompute the path forms that may appear inside the transcript JSON
$oldJson = $OldRepo.Replace('\', '\\')   # JSON-escaped Windows path
$newJson = $NewRepo.Replace('\', '\\')
$oldFwd  = $OldRepo.Replace('\', '/')    # forward-slash form
$newFwd  = $NewRepo.Replace('\', '/')
$oldBash = Get-BashPath $OldRepo         # git-bash form: /d/Development/...
$newBash = Get-BashPath $NewRepo
$oldKey  = Get-ProjectKey $OldRepo       # encoded project key, if referenced
$newKey  = Get-ProjectKey $NewRepo

foreach ($f in $files) {
    $dest = Join-Path $newDir $f.Name

    if ($NoRewritePaths) {
        Move-Item -LiteralPath $f.FullName -Destination $dest -Force
        Write-Host "Moved   $($f.Name)"
        continue
    }

    $content = Get-Content -LiteralPath $f.FullName -Raw
    $content = $content.Replace($oldJson, $newJson)
    $content = $content.Replace($oldFwd,  $newFwd)
    $content = $content.Replace($oldBash, $newBash)
    $content = $content.Replace($oldKey,  $newKey)

    # Write transcript to the new location as UTF-8 (no BOM), then remove the original
    [System.IO.File]::WriteAllText($dest, $content, (New-Object System.Text.UTF8Encoding($false)))
    Remove-Item -LiteralPath $f.FullName -Force
    Write-Host "Moved + rewrote paths:  $($f.Name)"
}

# Clean up the old project dir if it is now empty
if (-not (Get-ChildItem -LiteralPath $oldDir -Force)) {
    Remove-Item -LiteralPath $oldDir -Force
    Write-Host "Removed empty source project dir."
}

Write-Host ""
Write-Host "Done. Resume from the new repo with:"
Write-Host "    cd '$NewRepo'"
Write-Host "    claude --resume        # then pick this session"
Write-Host "  (or  claude --continue   to reopen the most recent session)"
