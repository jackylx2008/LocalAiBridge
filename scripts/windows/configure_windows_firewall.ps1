#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"
$ruleName = "LocalAiBridge llama.cpp 8080"

$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($null -ne $existing) {
    Write-Host "防火墙规则已存在：$ruleName"
    exit 0
}

New-NetFirewallRule `
    -DisplayName $ruleName `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort 8080 `
    -Profile Private `
    -RemoteAddress LocalSubnet | Out-Null

Write-Host "已添加防火墙规则：仅允许专用网络本地子网访问 TCP 8080"
