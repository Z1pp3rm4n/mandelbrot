import taichi as ti
import taichi.math as tm

# Module for different mandelbrot functions. 
# All are of the form f(z,c) -> z

@ti.func 
def cmul(ax:ti.float64,ay:ti.float64,bx:ti.float64,by:ti.float64):
    """
    Complex multiplication
    Args:
        a = (ax,ay)
        b = (bx,by)
    Returns:
        c = a*b
    """
    return ax*bx - ay*by, ax*by + ay*bx

@ti.func
def cpow(zx:ti.float64,zy:ti.float64, n:int):
    """
    Complex power with positive integer exponent
    Args:
        z = (zx,zy)
        n: Exponent, must be a positive integer
    Returns:
        z**n
    """
    ix, iy = ti.float64(1.0), ti.float64(0.0)
    while n > 1:
        if n % 2 == 1:
            ix, iy = cmul(ix,iy, zx,zy)
            n -= 1
        zx,zy = cmul(zx,zy,zx,zy)
        n //= 2
    return cmul(ix,iy, zx,zy)

@ti.func
def burning_ship(zx:ti.float64,zy:ti.float64, cx:ti.float64,cy:ti.float64):
    zx , zy = cpow(abs(zx), abs(zy), 2)
    return zx + cx, zy + cy


@ti.func
def mandelbrot2(zx:ti.float64,zy:ti.float64, cx :ti.float64,cy :ti.float64):
    zx , zy = cmul(zx,zy,zx,zy)
    zx += cx 
    zy += cy
    return zx, zy

@ti.func 
def mandelbrot3(zx:ti.float64,zy:ti.float64, cx :ti.float64,cy :ti.float64):
    zx, zy = cpow(zx,zy,3)
    return zx + cx, zy + cy

@ti.func
def mandelbrot4(zx:ti.float64,zy:ti.float64, cx :ti.float64,cy :ti.float64):
    zx, zy = cpow(zx,zy,4)
    return zx + cx, zy + cy

@ti.func
def mandelbrot5(zx:ti.float64,zy:ti.float64, cx :ti.float64,cy :ti.float64):
    zx, zy = cpow(zx,zy,5)
    return zx + cx, zy + cy

@ti.func
def mandelbrot6(zx:ti.float64,zy:ti.float64, cx :ti.float64,cy :ti.float64):
    zx, zy = cpow(zx,zy,6)
    return zx + cx, zy + cy

@ti.func
def mandelbrot7(zx:ti.float64,zy:ti.float64, cx :ti.float64,cy :ti.float64):
    zx, zy = cpow(zx,zy,7)
    return zx + cx, zy + cy
