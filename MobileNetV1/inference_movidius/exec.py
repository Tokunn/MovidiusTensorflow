#!/usr/bin/env sh

scp mlt_realtime_movidius_v1.py categories.txt rpi:~/
ssh rpi -X << EOF
hostname
DISPLAY=:0
sudo modprobe bcm2835-v4l2
chmod 755 mlt_realtime_movidius_v1.py
./mlt_realtime_movidius_v1.py
EOF
