import socket

if "ordnance" in socket.gethostname().lower():
    ROOT_DIR = '/home/sohrab/hdd/data/chest'
elif 'heg-rl001' in socket.gethostname().lower():
    ROOT_DIR = '/data/dt_group/images/chest'
