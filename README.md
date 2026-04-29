# WiFiUAV_StreamVODS
Reverse-engineered Drone's app Based on WiFiUAV 

# HardWare
[S25 Pro](https://alibaba.com/product-detail/S25-PRO-optical-flow-dual-cameras-1601160813563.html)

# Requirement
- Python 3.10(+)
- opencv-python numpy

# How To Use
  Windows 

  netsh advfirewall firewall add rule name="S25 drone" dir=in protocol=UDP remoteip=192.168.169.1 action=allow
  and then run stream_drone.py
  Remove later:
  netsh advfirewall firewall delete rule name="S25 drone"
