$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
python -m resume_tailor @args
