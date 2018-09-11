#!/bin/bash
killall nodejs
nodejs web/server.js &
cd cnn
./recognize_weburl.py "http://mo-ka.co/pi.php" false

