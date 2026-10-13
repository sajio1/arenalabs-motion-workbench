#!/usr/bin/env bash
# Explicit deployment only. Usage: sudo bash tools/bootstrap_cloud_demo.sh PUBLIC_IP
set -euo pipefail
PUBLIC_HOST=${1:?Supply the instance public IP or hostname}
[[ "$PUBLIC_HOST" =~ ^[a-zA-Z0-9.-]+$ ]] || { echo 'Invalid public host'; exit 1; }
ROOT=/opt/arenalab
apt-get update -qq
apt-get install -y nginx python3-venv ffmpeg libegl1 libgl1 libglib2.0-0
chown -R ubuntu:ubuntu "$ROOT"
sudo -u ubuntu python3 -m venv --system-site-packages "$ROOT/.venv"
sudo -u ubuntu "$ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
sudo -u ubuntu "$ROOT/.venv/bin/python" -m pip install --no-build-isolation -e "$ROOT[runtime,viewer]"
sudo -u ubuntu "$ROOT/.venv/bin/python" -m pip install numpy==1.26.4 opencv-python==4.11.0.86 mujoco==3.3.7 trimesh==4.12.2 transformers==5.17.0
cat >/etc/systemd/system/arenalab.service <<'SERVICE'
[Unit]
Description=ArenaLabs interactive motion demo
After=network-online.target
[Service]
User=ubuntu
WorkingDirectory=/opt/arenalab
Environment=MUJOCO_GL=egl
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/arenalab/.venv/bin/python -m greenwich_pipelines.viewer.server --config /opt/arenalab/configs/viewer.cloud.json
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
SERVICE
cat >/etc/nginx/sites-available/arenalab <<'NGINX'
server {
    listen 80 default_server;
    server_name PUBLIC_HOST;
    gzip on;
    gzip_types application/json application/javascript text/css image/svg+xml;
    gzip_min_length 1024;
    gzip_comp_level 4;
    client_max_body_size 128m;
    # Same-origin public demo; backend remains bound to loopback.
    set $allowed_origin "";
    if ($http_origin = "") { set $allowed_origin 1; }
    if ($http_origin = "http://PUBLIC_HOST") { set $allowed_origin 1; }
    if ($allowed_origin = "") { return 403; }
    # Private model/input/registry files are never web downloads.
    location ~ ^/assets/(inputs/|source-recordings/|uploaded/registry\.json|viewer-conversions\.json) { return 404; }
    location ~ ^/assets/.*\.(npz|npy|safetensors|pt|py|xml|urdf|log|zip|tar)$ { return 404; }
    location ~ ^/assets/jobs/[^/]+/(?!playback\.json$) { return 404; }
    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host localhost;
        proxy_set_header Origin http://localhost:8765;
        proxy_read_timeout 600s;
        proxy_request_buffering off;
    }
}
NGINX
sed -i "s/PUBLIC_HOST/$PUBLIC_HOST/g" /etc/nginx/sites-available/arenalab
ln -sfn /etc/nginx/sites-available/arenalab /etc/nginx/sites-enabled/default
nginx -t
systemctl daemon-reload
systemctl enable --now arenalab
systemctl restart nginx
curl --fail --retry 10 --retry-delay 2 "http://127.0.0.1/api/catalog/locomotion" -o /tmp/arenalab-catalog.json
echo "Demo: http://$PUBLIC_HOST/locomotion"
