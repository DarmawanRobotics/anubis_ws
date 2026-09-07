#!/bin/bash
set -e

USERNAME="${USERNAME:-devuser}"
LINUX_PASSWORD="${LINUX_PASSWORD:-changeme}"
VNC_RESOLUTION="${VNC_RESOLUTION:-1920x1080}"
VNC_DEPTH="${VNC_DEPTH:-24}"
HOME_DIR="/home/$USERNAME"

log() { echo "[entrypoint] $*"; }

# Check for VNC server binaries (vncserver or tigervncserver) and store the path in VNCSERVER_BIN
VNCSERVER_BIN="$(command -v vncserver 2>/dev/null || command -v tigervncserver 2>/dev/null || true)"

# Set the Linux password for the specified user
echo "${USERNAME}:${LINUX_PASSWORD}" | chpasswd
log "linux password set for ${USERNAME}"

# Set up USB device access for the plugdev group if /dev/bus/usb exists
if [ -d /dev/bus/usb ]; then
    chgrp -R plugdev /dev/bus/usb 2>/dev/null || true
    chmod -R g+rw /dev/bus/usb 2>/dev/null || true
    log "usb device group access applied"
fi

# Set up SSH server
ssh-keygen -A

SSH_PASSWORD_AUTH="${SSH_PASSWORD_AUTH:-yes}"
if [ "$SSH_PASSWORD_AUTH" = "no" ]; then
    if [ ! -s "$HOME_DIR/.ssh/authorized_keys" ]; then
        log "WARNING: SSH_PASSWORD_AUTH=no but no authorized_keys found for ${USERNAME} — you will be locked out. Falling back to password auth."
        SSH_PASSWORD_AUTH="yes"
    else
        sed -i 's/^PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
        log "password auth disabled — pubkey-only login"
    fi
fi

# Default to password auth if no authorized_keys are present
/usr/sbin/sshd -D &
log "sshd started on port 2222 (password auth: ${SSH_PASSWORD_AUTH})"

# Set up VNC server if the necessary binaries are available
if [ -z "$VNCSERVER_BIN" ] || ! command -v x11vnc > /dev/null 2>&1; then
    log "WARNING: vnc tooling missing (vncserver/tigervncserver or x11vnc) — ssh still available for debugging"
else
    mkdir -p "$HOME_DIR/.vnc"
    # x11vnc -storepasswd writes the same RFB-standard encrypted passwd format
    # tigervnc's own vncpasswd would — used as a workaround since this base
    # image's tigervnc-common build doesn't ship a passwd tool at all
    x11vnc -storepasswd "$LINUX_PASSWORD" "$HOME_DIR/.vnc/passwd" > /dev/null 2>&1
    chmod 600 "$HOME_DIR/.vnc/passwd"
    cp "$HOME_DIR/.vnc/xstartup.template" "$HOME_DIR/.vnc/xstartup"
    chmod +x "$HOME_DIR/.vnc/xstartup"
    chown -R "$USERNAME:$(id -g "$USERNAME")" "$HOME_DIR/.vnc"
    rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 2>/dev/null || true

    if su - "$USERNAME" -c "$VNCSERVER_BIN :1 -geometry ${VNC_RESOLUTION} -depth ${VNC_DEPTH} -localhost no"; then
        log "vnc ready on :1 (port 5901)"
    else
        log "WARNING: vnc failed to start — ssh still available for debugging"
    fi
fi

# If command-line arguments are provided, execute them as the specified user; otherwise, tail the VNC log files or sleep indefinitely
if [ "$#" -gt 0 ]; then
    exec su - "$USERNAME" -c "$*"
else
    exec tail -F "$HOME_DIR/.vnc"/*.log 2>/dev/null || exec sleep infinity
fi