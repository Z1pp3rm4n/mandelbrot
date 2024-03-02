import numpy as np
import pygame as pg
import taichi as ti
from fractal import Fractal
from functions import mandelbrot_func, burning_ship, mandelbrot3

RES = WIDTH, HEIGHT = 1920, 1080


platform = ti.cpu
ti.init(platform)

zoomIn = 0.990
zoomOut = 1.009
vel = 0.04


  
class App: 
    def __init__(self):
        self.clock = pg.time.Clock()

        self.main_screen = pg.display.set_mode((WIDTH,HEIGHT),pg.SCALED)
        self.sub_screen = pg.Surface((HEIGHT//2, HEIGHT//2))
        self.mandelbrot = Fractal(WIDTH, HEIGHT, mandelbrot_func)
        self.mandelbrot.render_naive()
        self.draw_main()

        self.JULIA_MODE = False
    
    def handle_events(self):
        events = pg.event.get()
        for e in events:
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_j: 
                    self.JULIA_MODE = not self.JULIA_MODE
            if e.type == pg.QUIT:
                exit()

    
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
    
    def zoom_loop(self):
        self.control()
        self.mandelbrot.update()
        self.draw_main()


    def julia_loop(self):
        if pg.mouse.get_pressed()[0]:
            xid,yid = pg.mouse.get_pos()
            bid = self.mandelbrot.buffer_id
            cx = self.mandelbrot.xcoords[bid, xid]
            cy = self.mandelbrot.ycoords[bid, yid]
            julia = self.mandelbrot.create_julia(cx,cy)
            julia.render_naive()
            julia_results = julia.get_results()
            self.draw_julia(julia_results)
            

    def draw_julia(self, julia_results):
        pg.surfarray.blit_array(self.sub_screen, julia_results)
        self.main_screen.blit(self.sub_screen, dest=(0,0))
        

    def draw_main(self):
        screen_array = self.mandelbrot.get_results()
        pg.surfarray.blit_array(self.main_screen, screen_array)

    def run(self):
        while True:
            self.handle_events()
            if self.JULIA_MODE:
                self.julia_loop()
            else:
                self.zoom_loop()
            pg.display.flip()

            
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')



if __name__ == '__main__':
    app = App()
    app.run()