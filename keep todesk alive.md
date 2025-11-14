sudo systemctl edit todeskd.service

\### Anything between here and the comment below will become the contents of the drop-in file

[Service]
Restart=always
RestartSec=5s

\### Edits below this comment will be discarded

保存-离开

sudo systemctl show -p Restart -p RestartUSec todeskd.service
