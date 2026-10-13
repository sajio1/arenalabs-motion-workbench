#!/usr/bin/env bash
# Explicit hosting setup, never part of a motion conversion.
set -euo pipefail
DOMAIN=${1:?Supply the demo DNS name}
[[ "$DOMAIN" =~ ^[a-zA-Z0-9.-]+$ ]] || exit 1
export DOMAIN
mkdir -p /var/www/arenalab-acme
cp /etc/nginx/sites-available/arenalab /etc/nginx/sites-available/arenalab.before-domain
python3 - <<'PY'
from pathlib import Path
p=Path('/etc/nginx/sites-available/arenalab')
t=p.read_text()
if 'location ^~ /.well-known/acme-challenge/' not in t:
    t=t.replace('    location / {','    location ^~ /.well-known/acme-challenge/ { root /var/www/arenalab-acme; }\n    location / {')
p.write_text(t)
PY
nginx -t
systemctl reload nginx
certbot certonly --webroot -w /var/www/arenalab-acme -d "$DOMAIN" --non-interactive --agree-tos --register-unsafely-without-email
python3 - <<'PY'
import os,re
from pathlib import Path
d=os.environ['DOMAIN'];p=Path('/etc/nginx/sites-available/arenalab')
t=p.read_text().replace('listen 80 default_server;','listen 443 ssl;')
t=re.sub(r'server_name [^;]+;',f'server_name {d};',t)
t=t.replace('    gzip on;',f'    ssl_certificate /etc/letsencrypt/live/{d}/fullchain.pem;\n    ssl_certificate_key /etc/letsencrypt/live/{d}/privkey.pem;\n    ssl_protocols TLSv1.2 TLSv1.3;\n    gzip on;')
t=t.replace('    if ($allowed_origin = "")',f'    if ($http_origin = "https://{d}") {{ set $allowed_origin 1; }}\n    if ($allowed_origin = "")')
t=t.replace('    location / {','    location = / { return 302 /locomotion; }\n    location / {')
http=f'''server {{
    listen 80 default_server;
    server_name {d};
    location ^~ /.well-known/acme-challenge/ {{ root /var/www/arenalab-acme; }}
    location / {{ return 308 https://{d}$request_uri; }}
}}
'''
p.write_text(http+t)
PY
mkdir -p /etc/letsencrypt/renewal-hooks/deploy
printf '#!/bin/sh\nnginx -t && systemctl reload nginx\n' >/etc/letsencrypt/renewal-hooks/deploy/arenalab-nginx.sh
chmod +x /etc/letsencrypt/renewal-hooks/deploy/arenalab-nginx.sh
nginx -t
systemctl reload nginx
echo "Demo: https://$DOMAIN/locomotion"
