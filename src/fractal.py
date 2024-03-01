import numpy as np
import pygame as pg
import taichi as ti
import taichi.math as tm


REDRAW_PERCENT = 0.75
SIM_RANGE = 100
MAX_ITER = 1000

@ti.func
def find_iter(x: ti.float64, y:ti.float64, x0:ti.float64, y0:ti.float64 , max_iter):
    x2 = x**2
    y2 = y**2 
    iter = 0 
    while (x2 + y2 <= 4 and iter < max_iter):
        x,y  = x2 - y2 + x0, 2*x*y + y0
        x2 = x**2
        y2 = y**2 
        iter += 1
    return iter


@ti.func
def mandelbrot_func(zx:ti.float64, zy:ti.float64, cx:ti.float64, cy:ti.float64):
    zx_old = zx
    zx =  zx**2 - zy**2 + cx
    zy = 2*zx*zy + cy
    return zx,zy

@ti.func
def get_color(iter: ti.int32, max_iter: ti.int32):
    # col = ti.cast((iter/max_iter * 255), ti.int32)
    return 0 if iter == max_iter else ti.cast((iter/max_iter * 255), ti.int32)


@ti.data_oriented
class Fractal:
    def __init__(self, width, height, func):
        self.bailout = 2.0
        self.max_iter = MAX_ITER
        self.func = func

        # Determines the coordinates of the fractal
        self.x_center = 0.0
        self.y_center = 0.0
        self.width = width
        self.height = height
        self.scale = 1.0

        # Memory to save the fractal calculations to / xaos algorithm
        self.buffer_id = 0
        self.screen_array = np.full((2,self.width, self.height, 3), [0, 0, 0], dtype=np.uint32)
        self.xcoords = ti.field(ti.float64, (2,self.width,))
        self.ycoords = ti.field(ti.float64, (2,self.height,))
        self.xlookup = ti.field(ti.int32, (self.width))
        self.ylookup = ti.field(ti.int32, (self.height,))

        # 
        self.first_render()

    @ti.func
    def find_iter(self, x: ti.float64, y:ti.float64, bailout, max_iter):
        iter = 0
        zx, zy = ti.float64(0.0), ti.float64(0.0)
        ptot = 8
        ckx, cky = zx, zy 
        bail = False
        while (not bail and ptot != max_iter):
            ckx,cky = zx,zy
            ptot += ptot
            if (ptot > max_iter): ptot = max_iter
            while(not bail and iter < ptot):
                zx, zy = self.func(zx,zy, x, y)
                iter += 1
                if (zx**2 + zy**2 > bailout ** 2): 
                    bail = True
                if (zx == ckx and zy == cky):
                    bail = True 
                    iter = max_iter
                
        return iter

        # iter = 0
        # zx, zy = ti.float64(0.0), ti.float64(0.0)
        # while(zx**2 + zy**2 <= bailout**2 and iter < max_iter):
        #     zx, zy = self.func(zx,zy,x,y)
        #     iter +=1
        # 
        # return iter

            
    def fast_render(self):
        self.buffer_id ^= 1
        self.fill_coords(self.buffer_id, self.x_center, self.y_center, self.scale)
        self.fill_xlookup()
        self.fill_ylookup()
        self.approximate_pixels(self.bailout, self.max_iter, self.buffer_id, self.screen_array)
        


    def fill_xlookup(self):
        best = np.zeros(shape=(self.width,), dtype=np.int32)
        dist = np.zeros(shape=self.width, dtype=np.float64)
        self.compare_x_coords(self.buffer_id, best, dist)
        prio = np.argsort(dist)
        self.fill_x_table(self.buffer_id, best, prio)
        

    @ti.kernel
    def compare_x_coords(self, bid: int, best:ti.types.ndarray(), dist:ti.types.ndarray()): # type: ignore
        for i in range(self.width):
            diff_best = ti.float64(1e10)
            id_best = -1

            for j in range(i - SIM_RANGE, i + SIM_RANGE):
                if j < 0 or j >= self.width:
                    continue

                diff = abs(self.xcoords[bid, i] - self.xcoords[bid^1, j])
                if diff < diff_best:
                    diff_best = diff
                    id_best = j

            best[i] = id_best
            dist[i] = diff_best
        
    @ti.kernel
    def fill_x_table(self, bid: int, best:ti.types.ndarray(), prio:ti.types.ndarray()): # type: ignore
        for i in range(self.width):
            id_origin = prio[i]
            id_best = best[id_origin]

            if i > self.width - self.width*REDRAW_PERCENT/100:
                self.xlookup[id_origin] = -1
            else:
                self.xlookup[id_origin] = id_best
                self.xcoords[bid, id_origin] = self.xcoords[bid^1, id_best]    

    def fill_ylookup(self):
        best = np.zeros(shape=(self.height,), dtype=np.int32)
        dist = np.zeros(shape=(self.height,), dtype=np.float64)
        self.compare_y_coords(self.buffer_id, best, dist)
        prio = np.argsort(dist)
        self.fill_y_table(self.buffer_id, best, prio)
        

    @ti.kernel
    def compare_y_coords(self, bid: int, best:ti.types.ndarray(), dist:ti.types.ndarray()):# type: ignore
        for i in range(self.height):
            diff_best = 1e10
            id_best = -1

            for j in range(i - SIM_RANGE, i + SIM_RANGE):
                if j < 0 or j >= self.height:
                    continue

                diff = abs(self.ycoords[bid, i] - self.ycoords[bid^1, j])
                if diff < diff_best:
                    diff_best = diff
                    id_best = j

            best[i] = id_best
            dist[i] = diff_best
        
    @ti.kernel
    def fill_y_table(self, bid: int, best:ti.types.ndarray(), prio:ti.types.ndarray()): # type: ignore
        for i in range(self.height):
            id_origin = prio[i]
            id_best = best[id_origin]

            if i > self.height - self.height*REDRAW_PERCENT/100:
                self.ylookup[id_origin] = -1
            else:
                self.ylookup[id_origin] = id_best
                self.ycoords[bid, id_origin] = self.ycoords[bid^1, id_best]    
    
    @ti.kernel
    def approximate_pixels(self, bailout:float, max_iter:int, bid:int, screen_array: ti.types.ndarray()): # type: ignore
        for xid, yid in ti.ndrange(self.width, self.height):
            xid_best = self.xlookup[xid]
            yid_best = self.ylookup[yid]

            if (xid_best != -1 and yid_best != -1):
                screen_array[bid, xid,yid, 0] = screen_array[bid^1, xid_best, yid_best, 0]
                screen_array[bid, xid,yid, 1] = screen_array[bid^1, xid_best, yid_best, 1]
                screen_array[bid, xid,yid, 2] = screen_array[bid^1, xid_best, yid_best, 2]
            else:
                real = self.xcoords[bid, xid]
                imag = self.ycoords[bid, yid]
                iter = self.find_iter(real, imag, bailout, max_iter)
                # iter = find_iter(0.0,0.0,real,imag, self.max_iter)
                col = get_color(iter, max_iter)
                screen_array[bid, xid,yid, 0] = col
                screen_array[bid, xid,yid, 1] = col
                screen_array[bid, xid,yid, 2] = col


    @ti.kernel
    def fill_coords(self, bid: int, x_center: ti.float64, y_center:ti.float64, scale:ti.float64):
        half_width = self.width //2
        half_height = self.height //2
        step = 4.0 / self.height * scale
        for i in range(self.width):
            self.xcoords[bid, i] = x_center + (i - half_width)*step
        for i in range(self.height):
            self.ycoords[bid, i] = y_center + (i - half_height)*step

    def first_render(self):
        self.fill_coords(self.buffer_id, self.x_center, self.y_center, self.scale)
        self.calculate_all(self.bailout, self.max_iter, self.buffer_id, self.screen_array)
    
    @ti.kernel
    def calculate_all(self, bailout:float, max_iter:int,  bid: int, screen_array: ti.types.ndarray()): # type: ignore
        for i,j in ti.ndrange(self.width, self.height):
            x = self.xcoords[bid, i]
            y = self.ycoords[bid, j]
            iter = self.find_iter(x,y, bailout, max_iter)
            # iter = find_iter(0.0,0.0,x,y, self.max_iter)
            col = get_color(iter, max_iter)

            screen_array[bid, i, j, 0] = col
            screen_array[bid, i, j, 1] = col
            screen_array[bid, i, j, 2] = col

    def update(self):
        self.fast_render()
    
    def get_results(self):
        return self.screen_array[self.buffer_id]
