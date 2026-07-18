param(
    [string]$Remote = "origin",
    [string]$Branch = "main",
    [string]$CommitMessage = "",
    [int]$WarnMB = 50,
    [int]$HardLimitMB = 100
)

$ErrorActionPreference = "Stop"

function Format-Size {
    param([double]$Bytes)
    if ($Bytes -ge 1GB) { return "{0:N2} GB" -f ($Bytes / 1GB) }
    if ($Bytes -ge 1MB) { return "{0:N2} MB" -f ($Bytes / 1MB) }
    if ($Bytes -ge 1KB) { return "{0:N2} KB" -f ($Bytes / 1KB) }
    return "{0:N0} B" -f $Bytes
}

function Get-GitCommand {
    $defaultGit = "C:\Program Files\Git\cmd\git.exe"
    if (Test-Path -LiteralPath $defaultGit) { return $defaultGit }
    return "git"
}

function Get-RelativePath {
    param([string]$Root, [string]$FullName)
    $rootWithSlash = $Root.TrimEnd("\") + "\"
    return $FullName.Replace($rootWithSlash, "").Replace("\", "/")
}

function Test-GitIgnored {
    param([string]$Git, [string]$Path)
    & $Git check-ignore -q -- $Path 2>$null
    return $LASTEXITCODE -eq 0
}

function Test-GitLfsTracked {
    param([string]$Git, [string]$Path)
    $attr = & $Git check-attr filter -- $Path 2>$null
    return (($attr -join "`n") -match "filter:\s*lfs")
}

$git = Get-GitCommand

$gitRoot = (& $git rev-parse --show-toplevel 2>$null)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($gitRoot)) {
    Write-Host "Kein Git-Repository gefunden. Bitte Script im Projektordner starten." -ForegroundColor Red
    exit 1
}

$gitRoot = $gitRoot.Trim()
Set-Location $gitRoot

Write-Host ""
Write-Host "Git-Projekt: $gitRoot" -ForegroundColor Cyan
Write-Host "Remote/Branch: $Remote/$Branch" -ForegroundColor Cyan
Write-Host ""

$excludedDirs = @(".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".mpl-cache")
$allFiles = Get-ChildItem -LiteralPath $gitRoot -Recurse -Force -File -ErrorAction SilentlyContinue |
    Where-Object {
        $parts = $_.FullName.Replace($gitRoot, "").Split("\", [System.StringSplitOptions]::RemoveEmptyEntries)
        -not ($parts | Where-Object { $excludedDirs -contains $_ })
    }

$totalBytes = ($allFiles | Measure-Object -Property Length -Sum).Sum
if ($null -eq $totalBytes) { $totalBytes = 0 }

Write-Host "Projektgroesse ohne .git/cache: $(Format-Size $totalBytes)" -ForegroundColor Green
Write-Host "Dateien im Arbeitsbaum: $($allFiles.Count)" -ForegroundColor Green
Write-Host ""

$warnBytes = $WarnMB * 1MB
$hardBytes = $HardLimitMB * 1MB
$largeFiles = $allFiles | Where-Object { $_.Length -ge $warnBytes } | Sort-Object Length -Descending

if ($largeFiles) {
    Write-Host "Grosse Dateien ab $WarnMB MB:" -ForegroundColor Yellow
    foreach ($file in $largeFiles | Select-Object -First 30) {
        $rel = Get-RelativePath $gitRoot $file.FullName
        $ignored = Test-GitIgnored $git $rel
        $lfs = Test-GitLfsTracked $git $rel
        $tag = if ($ignored) { "IGNORIERT" } elseif ($lfs) { "LFS" } elseif ($file.Length -ge $hardBytes) { "BLOCKER" } else { "OK" }
        $color = if ($tag -eq "BLOCKER") { "Red" } elseif ($tag -eq "IGNORIERT" -or $tag -eq "LFS") { "DarkGray" } else { "Yellow" }
        Write-Host ("  {0,10} | {1,-8} | {2}" -f (Format-Size $file.Length), $tag, $rel) -ForegroundColor $color
    }
    if ($largeFiles.Count -gt 30) {
        Write-Host "  ... weitere $($largeFiles.Count - 30) grosse Dateien ausgeblendet" -ForegroundColor DarkGray
    }
    Write-Host ""
}

$untracked = & $git ls-files --others --exclude-standard
$ignored = & $git ls-files --others --ignored --exclude-standard

if ($untracked) {
    Write-Host "Nicht getrackte Dateien, die beim 'git add .' aufgenommen werden koennen:" -ForegroundColor Yellow
    $untracked | Select-Object -First 40 | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    if ($untracked.Count -gt 40) { Write-Host "  ... weitere $($untracked.Count - 40) Dateien" -ForegroundColor DarkGray }
    Write-Host ""
}

if ($ignored) {
    Write-Host "Ignorierte Dateien, die NICHT gepusht werden:" -ForegroundColor DarkGray
    $ignored | Select-Object -First 40 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
    if ($ignored.Count -gt 40) { Write-Host "  ... weitere $($ignored.Count - 40) ignorierte Dateien" -ForegroundColor DarkGray }
    Write-Host ""
}

$blockingFiles = @()
foreach ($file in ($allFiles | Where-Object { $_.Length -ge $hardBytes })) {
    $rel = Get-RelativePath $gitRoot $file.FullName
    $ignored = Test-GitIgnored $git $rel
    $lfs = Test-GitLfsTracked $git $rel
    if (-not $ignored -and -not $lfs) {
        $blockingFiles += [pscustomobject]@{ Path = $rel; Size = $file.Length }
    }
}

if ($blockingFiles.Count -gt 0) {
    Write-Host "Achtung: Diese Dateien sind groesser als $HardLimitMB MB und weder ignoriert noch Git-LFS-getrackt." -ForegroundColor Red
    Write-Host "Ein normaler GitHub-Push wird damit sehr wahrscheinlich abgelehnt:" -ForegroundColor Red
    foreach ($item in $blockingFiles | Select-Object -First 30) {
        Write-Host ("  {0,10} | {1}" -f (Format-Size $item.Size), $item.Path) -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Loesung: Datei ignorieren, aus Git entfernen oder mit Git LFS tracken." -ForegroundColor Yellow
    $continueWithBlockers = Read-Host "Trotzdem fortfahren? (j/N)"
    if ($continueWithBlockers.ToLower() -ne "j") {
        Write-Host "Abgebrochen, damit GitHub den Push nicht wieder ablehnt." -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "Aktueller Git-Status:" -ForegroundColor Cyan
& $git status --short
Write-Host ""

$answer = Read-Host "Jetzt 'git add .', Commit, Pull-Rebase und Push ausfuehren? (j/N)"
if ($answer.ToLower() -ne "j") {
    Write-Host "Abgebrochen. Es wurde nichts gepusht." -ForegroundColor Yellow
    exit 0
}

& $git add .

$staged = & $git diff --cached --name-only
if (-not $staged) {
    Write-Host "Keine neuen Aenderungen zum Committen. Fuehre nur Pull/Push aus." -ForegroundColor Yellow
} else {
    $stagedBlockers = @()
    foreach ($path in $staged) {
        if (Test-Path -LiteralPath $path) {
            $file = Get-Item -LiteralPath $path
            if ($file.Length -ge $hardBytes -and -not (Test-GitLfsTracked $git $path)) {
                $stagedBlockers += [pscustomobject]@{ Path = $path; Size = $file.Length }
            }
        }
    }
    if ($stagedBlockers.Count -gt 0) {
        Write-Host "Nach 'git add .' sind noch Dateien > $HardLimitMB MB ohne LFS gestaged:" -ForegroundColor Red
        foreach ($item in $stagedBlockers) {
            Write-Host ("  {0,10} | {1}" -f (Format-Size $item.Size), $item.Path) -ForegroundColor Red
        }
        Write-Host "Entferne sie z.B. mit: git restore --staged <datei>" -ForegroundColor Yellow
        exit 1
    }

    if ([string]::IsNullOrWhiteSpace($CommitMessage)) {
        $CommitMessage = Read-Host "Commit-Message"
        if ([string]::IsNullOrWhiteSpace($CommitMessage)) {
            $CommitMessage = "Update project files"
        }
    }
    & $git commit -m $CommitMessage
}

& $git pull --rebase $Remote $Branch
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pull/Rebase fehlgeschlagen. Bitte Konflikte loesen, dann erneut pushen." -ForegroundColor Red
    exit $LASTEXITCODE
}

& $git push $Remote $Branch
if ($LASTEXITCODE -eq 0) {
    Write-Host "Push erfolgreich." -ForegroundColor Green
} else {
    Write-Host "Push fehlgeschlagen. Oben steht der genaue GitHub-Fehler." -ForegroundColor Red
    exit $LASTEXITCODE
}
