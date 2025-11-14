sudo nano /etc/systemd/system/todesk.service

[Unit]
Description=ToDesk remote control service
After=network.target

[Service]
Type=simple
ExecStart=/opt/todesk/bin/ToDesk
Restart=always
RestartSec=5s
User=forcifer-123
\# 您可以根据需要修改WorkingDirectory，如果ToDesk依赖特定的工作目录
\# WorkingDirectory=/home/forcifer-123

[Install]
WantedBy=multi-user.target

