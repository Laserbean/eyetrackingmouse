from ahk import AHK
ahk = AHK()


import ctypes
user32 = ctypes.windll.user32
screensize = ( user32.GetSystemMetrics(0)    ,    user32.GetSystemMetrics(1) )

mout = (34, 56)

mspeed = 0
ahk.mouse_move(x=mout[0]*screensize[0]/100, y =mout[1]*screensize[1]/100, speed=mspeed) 
print(ahk.mouse_position)