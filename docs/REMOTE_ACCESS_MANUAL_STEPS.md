# Remote Access Manual Steps

> Last updated: 2026-05-26. These are the remaining account/GUI steps that cannot be completed from this non-interactive Windows shell.

## Goal

Use Tailscale only as the private network between Mac and Windows. Keep your separate external-network VPN/proxy for country-restricted internet access.

Reasoning: Tailscale normally acts as an overlay network and does not touch public internet traffic. Tailscale exit nodes route public internet traffic through another device, similar to a traditional VPN, so do not enable an exit node for this Mac/Windows development link unless you intentionally want Tailscale to replace the other VPN.

## Windows

1. Install Tailscale from <https://tailscale.com/download/windows>.
2. Sign in to the same Tailscale account/tailnet you will use on Mac.
3. Do not enable "Use exit node" on Windows for this workflow.
4. Keep the existing Windows OpenSSH server enabled; this machine already has `sshd` running and `C:\Users\hhvrf\.ssh\authorized_keys` contains the Mac public key.
5. Note the Windows Tailscale IP, usually in the `100.x.y.z` range.

## Mac

1. Install/sign in to Tailscale.
2. Keep your other external-network VPN/proxy as-is.
3. Test plain SSH over the Tailscale IP:

```bash
ssh hhvrf@<windows_tailscale_ip>
```

4. If that works, use the same IP for artifact transfer:

```bash
scp -r hhvrf@<windows_tailscale_ip>:C:/Users/hhvrf/Documents/neuro_film/loras .
```

## Notes

- Windows `winget` and MSI install attempts were made from the shell, but neither completed without interactive elevation/login.
- Tailscale SSH is not required here. Plain OpenSSH over the Tailscale private IP keeps the existing SSH key setup and avoids replacing Windows port 22 behavior.
