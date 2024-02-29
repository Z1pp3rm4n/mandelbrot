import numpy as np
import pygame as pg
import taichi as ti


RES = WIDTH, HEIGHT = 1920, 1080
REDRAW_PERCENT = 0.75
SIM_RANGE = 60

platform = ti.cpu
ti.init(platform)


max_iter = 1000
fps = 60

zoomIn = 0.990
zoomOut = 1.009
vel = 0.04

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
def get_color(iter: ti.int32, max_iter: ti.int32):
    col = ti.cast((iter/max_iter * 255), ti.int32)
    return col
    

@ti.data_oriented
class Fractal:
    def __init__(self, screen):
        self.screen = screen 

        self.x_center = 0.0
        self.y_center = 0.0
        self.scale = 1.0

        self.buffer_id = 0
        self.screen_field = ti.Vector.field(3, dtype=ti.int32, shape=(2,WIDTH,HEIGHT))
        self.screen_array = np.full((2,WIDTH, HEIGHT, 3), [0, 0, 0], dtype=np.uint32)
        self.xcoords = ti.field(ti.float64, (2,WIDTH,))
        self.ycoords = ti.field(ti.float64, (2,HEIGHT,))
        self.xlookup = ti.field(ti.int32, (WIDTH))
        self.ylookup = ti.field(ti.int32, (HEIGHT,))


    def fast_render(self):
        self.fill_coords(self.buffer_id, self.x_center, self.y_center, self.scale)
        self.fill_xlookup()
        self.fill_ylookup()
        self.approximate_pixels(self.buffer_id, self.screen_array)

    def fill_xlookup(self):
        best = np.zeros(shape=(WIDTH,), dtype=np.int32)
        dist = np.zeros(shape=WIDTH, dtype=np.float64)
        self.compare_x_coords(self.buffer_id, best, dist)
        prio = np.argsort(dist)
        self.fill_x_table(self.buffer_id, best, prio)
        

    @ti.kernel
    def compare_x_coords(self, bid: int, best:ti.types.ndarray(), dist:ti.types.ndarray()):
        for i in range(WIDTH):
            diff_best = 1e10
            id_best = -1

            for j in range(i - SIM_RANGE, i + SIM_RANGE):
                if j < 0 or j >= WIDTH:
                    continue

                diff = abs(self.xcoords[bid, i] - self.xcoords[bid^1, j])
                if diff < diff_best:
                    diff_best = diff
                    id_best = j

            best[i] = id_best
            dist[i] = diff_best
        
    @ti.kernel
    def fill_x_table(self, bid: int, best:ti.types.ndarray(), prio:ti.types.ndarray()):
        for i in range(WIDTH):
            id_origin = prio[i]
            id_best = best[id_origin]

            if i > WIDTH - WIDTH*REDRAW_PERCENT/100:
                self.xlookup[id_origin] = -1
            else:
                self.xlookup[id_origin] = id_best
                self.xcoords[bid, id_origin] = self.xcoords[bid^1, id_best]    

    def fill_ylookup(self):
        best = np.zeros(shape=(HEIGHT,), dtype=np.int32)
        dist = np.zeros(shape=(HEIGHT,), dtype=np.float64)
        self.compare_y_coords(self.buffer_id, best, dist)
        prio = np.argsort(dist)
        self.fill_y_table(self.buffer_id, best, prio)
        

    @ti.kernel
    def compare_y_coords(self, bid: int, best:ti.types.ndarray(), dist:ti.types.ndarray()):
        for i in range(HEIGHT):
            diff_best = 1e10
            id_best = -1

            for j in range(i - SIM_RANGE, i + SIM_RANGE):
                if j < 0 or j >= HEIGHT:
                    continue

                diff = abs(self.ycoords[bid, i] - self.ycoords[bid^1, j])
                if diff < diff_best:
                    diff_best = diff
                    id_best = j

            best[i] = id_best
            dist[i] = diff_best
        
    @ti.kernel
    def fill_y_table(self, bid: int, best:ti.types.ndarray(), prio:ti.types.ndarray()):
        for i in range(HEIGHT):
            id_origin = prio[i]
            id_best = best[id_origin]

            if i > HEIGHT - HEIGHT*REDRAW_PERCENT/100:
                self.ylookup[id_origin] = -1
            else:
                self.ylookup[id_origin] = id_best
                self.ycoords[bid, id_origin] = self.ycoords[bid^1, id_best]    
    
    @ti.kernel
    def approximate_pixels(self, bid:int, screen_array: ti.types.ndarray()):
        for xid, yid in ti.ndrange(WIDTH, HEIGHT):
            xid_best = self.xlookup[xid]
            yid_best = self.ylookup[yid]

            if (xid_best != -1 and yid_best != -1):
                screen_array[bid, xid,yid, 0] = screen_array[bid^1, xid_best, yid_best, 0]
                screen_array[bid, xid,yid, 1] = screen_array[bid^1, xid_best, yid_best, 1]
                screen_array[bid, xid,yid, 2] = screen_array[bid^1, xid_best, yid_best, 2]
            else:
                real = self.xcoords[bid, xid]
                imag = self.ycoords[bid, yid]
                iter = find_iter(0.0,0.0, real, imag, max_iter)
                col = get_color(iter, max_iter)
                screen_array[bid, xid,yid, 0] = col
                screen_array[bid, xid,yid, 1] = col
                screen_array[bid, xid,yid, 2] = col


    @ti.kernel
    def fill_coords(self, bid: int, x_center: ti.float64, y_center:ti.float64, scale:ti.float64):
        HALF_WIDTH = WIDTH //2
        HALF_HEIGHT = HEIGHT //2
        step = 4.0 / HEIGHT * scale
        for i in range(WIDTH):
            self.xcoords[bid, i] = x_center + (i - HALF_WIDTH)*step
        for i in range(HEIGHT):
            self.ycoords[bid, i] = y_center + (i - HALF_HEIGHT)*step

    def first_render(self):
        self.fill_coords(self.buffer_id, self.x_center, self.y_center, self.scale)
        self.calculate_all(self.buffer_id, self.screen_array)
    
    @ti.kernel
    def calculate_all(self, bid: int, screen_array: ti.types.ndarray()):
        for i,j in ti.ndrange(WIDTH, HEIGHT):
            x = self.xcoords[bid, i]
            y = self.ycoords[bid, j]
            iter = find_iter(0.0, 0.0, x, y, max_iter)
            col = get_color(iter, max_iter)

            screen_array[bid, i, j, 0] = col
            screen_array[bid, i, j, 1] = col
            screen_array[bid, i, j, 2] = col
            

    def draw(self):
        pg.surfarray.blit_array(self.screen, self.screen_array[self.buffer_id])
    
    def update(self):
        if platform == ti.gpu:
            self.first_render()
        else:
            self.fast_render()
            self.buffer_id ^=1
        self.draw()
        
               
    def test_init(self):
        self.fast_render()
        self.buffer_id = 1
        for i in range(WIDTH):
            self.xlookup[i] = i
        for j in range(HEIGHT):
            self.xlookup[j] = j
  

class App: 
    def __init__(self):
        self.clock = pg.time.Clock()

        self.main_screen = pg.display.set_mode((WIDTH,HEIGHT),pg.SCALED)
        self.mandelbrot = Fractal(self.main_screen)
        self.mandelbrot.test_init()
    
    def control(self):
        pressed_key = pg.key.get_pressed()

        if pressed_key[pg.K_a]:
            self.mandelbrot.x_center -= vel * self.mandelbrot.scale
        if pressed_key[pg.K_d]:
            self.mandelbrot.x_center += vel * self.mandelbrot.scale
        if pressed_key[pg.K_w]:
            self.mandelbrot.y_center -= vel * self.mandelbrot.scale
        if pressed_key[pg.K_s]:
            self.mandelbrot.y_center += vel * self.mandelbrot.scale

        # stable zoom and movement
        if pressed_key[pg.K_UP]:
            self.mandelbrot.scale *= zoomIn
        if pressed_key[pg.K_DOWN]:
            self.mandelbrot.scale *= zoomOut
            

    def run(self):
        while True:
            self.main_screen.fill('black')
            self.control()
            self.mandelbrot.update()
            pg.display.flip()

            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')



if __name__ == '__main__':
    app = App()
    app.run()
