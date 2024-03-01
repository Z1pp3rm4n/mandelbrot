import taichi as ti
import taichi.math as tm

@ti.func 
def cmul(ax:ti.float64,ay:ti.float64,bx:ti.float64,by:ti.float64):
    return ax*bx - ay*by, ax*by + ay*bx

@ti.func
def cpow(zx:ti.float64,zy:ti.float64, n:int):
    ix, iy = ti.float64(1.0), ti.float64(0.0)
    while n > 1:
        if n % 2 == 1:
            ix, iy = cmul(ix,iy, zx,zy)
            n -= 1
        zx,zy = cmul(zx,zy,zx,zy)
        n //= 2
    return cmul(ix,iy, zx,zy)

@ti.func 
def mandelbrot3(zx:ti.float64,zy:ti.float64, cx :ti.float64,cy :ti.float64):
    zx, zy = cpow(zx,zy, 6)
    zx += cx
    zy += cy
    return zx,zy

@ti.func
def mandelbrot_func(zx:ti.float64,zy:ti.float64, cx :ti.float64,cy :ti.float64):
    
    zx , zy = cmul(zx,zy,zx,zy)
    zx += cx 
    zy += cy
    return zx, zy

@ti.func
def burning_ship(zx:ti.float64,zy:ti.float64, cx:ti.float64,cy:ti.float64):
    zx = abs(zx)
    zy = abs(zy)
    zx_old = zx 
    zx = zx**2 - zy**2 + cx
    zy = 2*zx_old*zy + cy
    return zx, zy

