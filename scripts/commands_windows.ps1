param(
    [ValidateSet('t1n_t1c', 't2w_t2f')][string]$Task = 't1n_t1c',
    [ValidateSet(2, 4, 8)][int]$Patch = 4,
    [switch]$MainStructure
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    python scripts/check_dataset.py --task $Task --scan
    if ($LASTEXITCODE -ne 0) { throw 'Dataset validation failed' }
    python scripts/download_ae.py
    if ($LASTEXITCODE -ne 0) { throw 'MAISI verification failed' }
    python -u precompute_latents.py --task $Task --device cuda --resume
    if ($LASTEXITCODE -ne 0) { throw 'Latent precomputation failed' }
    $structureArgs = @()
    if ($MainStructure) { $structureArgs += '--main-structure' }
    python -u train.py --config "configs/${Task}_p${Patch}.yaml" --device cuda --amp-dtype auto @structureArgs
    if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
} finally {
    Pop-Location
}
