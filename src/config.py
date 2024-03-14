import taichi as ti 

# Resolution
RES = WIDTH, HEIGHT = 1600,900 # Change this to 1920, 1080 if you have a good CPU
MAX_FPS = 60
platform = ti.cpu # Change the backend to be used: ti.cpu or ti.gpu

# Speed settings
zoomIn = 0.985 
zoomOut = 1/zoomIn
vel = 0.04

# Render settings
REDRAW_PERCENT = 0.75  # Change to 0.5 for better rendering speed
SIM_RANGE = 100
MAX_ITER = 300   
GRADIENT_LENGTH=MAX_ITER//10

