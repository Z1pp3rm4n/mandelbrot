import numpy as np
import pygame as pg
import taichi as ti
import taichi.math as tm


REDRAW_PERCENT = 0.75
SIM_RANGE = 100
MAX_ITER = 1000
GRADIENT_LENGTH=260

vec3 = ti.types.vector(3,float)
a,b,c,d = vec3(0.8,0.5,0.4),vec3(0.2,0.4,0.2),vec3(2.0,1.0,1.0),vec3(0.0,0.25,0.25) # pinkish scheme
# a,b,c,d = vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.3,0.20,0.20) # blueish scheme




@ti.func
def palette(t):
    # col = ti.cast((iter/max_iter * 255), ti.int32)
    # t = iter / max_iter
    # return 0 if iter == max_iter else ti.cast((iter/max_iter * 255), ti.int32)
    return a + b*tm.cos(6.28318*(c*t + d))



@ti.data_oriented
class Fractal:
    def __init__(self, width, height, func, bailout=2.0, max_iter=MAX_ITER, exponent=2):
        self.bailout = bailout
        self.max_iter = max_iter
        self.func = func
        self.exponent = exponent
        # Determines the coordinates of the fractal
        self.x_center = 0.0
        self.y_center = 0.0
        self.width = width
        self.height = height
        self.scale = 1.0

        # Memory to save the fractal calculations to / xaos algorithm
        self.buffer_id = 0

        self.screen_array = np.full((2, self.width, self.height, 3), [0, 0, 0], dtype=np.uint32)

        self.xcoords = ti.field(ti.float64, (2,self.width,))
        self.ycoords = ti.field(ti.float64, (2,self.height,))
        self.xlookup = ti.field(ti.int32, (self.width))
        self.ylookup = ti.field(ti.int32, (self.height,))

    
    def create_julia(self, cx, cy):
        size = self.height // 2
        return Julia(size,size,self.func, cx, cy, self.bailout, self.max_iter, self.exponent)

    @ti.func
    def get_color(self, x,y):
        return self.get_color_4(0.0, 0.0, x,y )

    @ti.func
    def get_color_4(self, zx:ti.float64, zy:ti.float64, cx: ti.float64, cy:ti.float64):
        iter = 0
        period = 8
        ckx, cky = zx, zy 
        bail = False
        while (not bail and period != self.max_iter):
            ckx,cky = zx,zy
            period += period
            if (period > self.max_iter): period = self.max_iter
            while(not bail and iter < period):
                zx, zy = self.func(zx,zy, cx, cy)
                iter += 1
                if (zx**2 + zy**2 > self.bailout ** 2): 
                    bail = True
                if (zx == ckx and zy == cky):
                    bail = True 
                    iter = self.max_iter
                
        for _ in range(2):
            zx, zy = self.func(zx,zy,cx,cy)
        logz_n = tm.log(zx**2 + zy**2)
        nu = tm.log(logz_n / ti.log(self.exponent))/tm.log(self.bailout)
        mu = iter + 3 - nu
        return vec3(0.0) if iter == self.max_iter else palette((mu % GRADIENT_LENGTH) / GRADIENT_LENGTH) * 255

        # iter = 0
        # zx, zy = ti.float64(0.0), ti.float64(0.0)
        # while(zx**2 + zy**2 <= bailout**2 and iter < max_iter):
        #     zx, zy = self.func(zx,zy,x,y)
        #     iter +=1
        # 
        # return iter


  
    def update(self):
        self.render_xaos()
    
    def get_results(self):
        return self.screen_array[self.buffer_id]
    

    def render_naive(self):
        self.fill_coords(self.buffer_id, self.x_center, self.y_center, self.scale)
        self.fill_pixels_naive(self.buffer_id, self.screen_array)

    @ti.kernel
    def fill_coords(self, bid: int, x_center: ti.float64, y_center:ti.float64, scale:ti.float64):
        half_width = self.width //2
        half_height = self.height //2
        step = 4.0 / self.height * scale
        for i in range(self.width):
            self.xcoords[bid, i] = x_center + (i - half_width)*step
        for i in range(self.height):
            self.ycoords[bid, i] = y_center + (i - half_height)*step

    @ti.kernel
    def fill_pixels_naive(self, bid: int, screen_array: ti.types.ndarray()): # type: ignore
        for xid, yid in ti.ndrange(self.width, self.height):
            x = self.xcoords[bid, xid]
            y = self.ycoords[bid, yid]
            col = self.get_color(x,y)
            screen_array[bid,xid,yid,0] = int(col.x)
            screen_array[bid,xid,yid,1] = int(col.y)
            screen_array[bid,xid,yid,2] = int(col.z)

       
    def render_xaos(self):
        self.buffer_id ^= 1
        self.fill_coords(self.buffer_id, self.x_center, self.y_center, self.scale)
        self.fill_xlookup()
        self.fill_ylookup()
        self.fill_iters_xaos(self.buffer_id, self.screen_array)

    @ti.kernel
    def fill_iters_xaos(self, bid:int, screen_array:ti.types.ndarray()): # type: ignore
        for xid, yid in ti.ndrange(self.width, self.height):
            xid_best = self.xlookup[xid]
            yid_best = self.ylookup[yid]

            if (xid_best != -1 and yid_best != -1):
                screen_array[bid, xid,yid, 0] = screen_array[bid^1, xid_best, yid_best, 0]
                screen_array[bid, xid,yid, 1] = screen_array[bid^1, xid_best, yid_best, 1]
                screen_array[bid, xid,yid, 2] = screen_array[bid^1, xid_best, yid_best, 2]
            else:
                x = self.xcoords[bid, xid]
                y = self.ycoords[bid, yid]
                col = self.get_color(x,y)
                screen_array[bid,xid,yid,0] = int(col.x)
                screen_array[bid,xid,yid,1] = int(col.y)
                screen_array[bid,xid,yid,2] = int(col.z)                

               



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
    





    
class Julia(Fractal):
    def __init__(self, width, height, func, cx, cy, bailout=2.0, max_iter=MAX_ITER, exponent=2):
        super().__init__(width, height, func, bailout, max_iter, exponent)
        self.cx = cx
        self.cy = cy

    @ti.func
    def get_color(self, x: ti.float64, y:ti.float64):
        return self.get_color_4(x, y, self.cx, self.cy)

