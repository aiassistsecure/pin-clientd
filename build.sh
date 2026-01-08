#!/bin/bash

set -e

echo ""
echo "     █████╗ ██╗ █████╗ ███████╗    ██████╗ ██╗███╗   ██╗"
echo "    ██╔══██╗██║██╔══██╗██╔════╝    ██╔══██╗██║████╗  ██║"
echo "    ███████║██║███████║███████╗    ██████╔╝██║██╔██╗ ██║"
echo "    ██╔══██║██║██╔══██║╚════██║    ██╔═══╝ ██║██║╚██╗██║"
echo "    ██║  ██║██║██║  ██║███████║    ██║     ██║██║ ╚████║"
echo "    ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝╚═╝  ╚═══╝"
echo ""
echo "    PIN Client Daemon - Headless Build"
echo "    https://AiAssist.net"
echo ""

cd "$(dirname "$0")"

if ! command -v cargo &> /dev/null; then
    echo "    [ERROR] Rust is not installed."
    echo "    Install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "    [BUILD] Compiling release binary..."
echo ""

cargo build --release

echo ""
echo "    ╔════════════════════════════════════════════════════════╗"
echo "    ║                   BUILD SUCCESSFUL                     ║"
echo "    ╚════════════════════════════════════════════════════════╝"
echo ""
echo "    Binary: target/release/pin-clientd"
echo ""
echo "    Installation:"
echo ""
echo "      sudo mkdir -p /opt/pin-clientd"
echo "      sudo cp target/release/pin-clientd /opt/pin-clientd/"
echo "      sudo cp config.example.json /opt/pin-clientd/config.json"
echo "      sudo nano /opt/pin-clientd/config.json  # Edit your credentials"
echo "      sudo cp pin-clientd.service /etc/systemd/system/"
echo "      sudo systemctl daemon-reload"
echo "      sudo systemctl enable pin-clientd"
echo "      sudo systemctl start pin-clientd"
echo ""
echo "    View logs:"
echo "      journalctl -u pin-clientd -f"
echo ""
