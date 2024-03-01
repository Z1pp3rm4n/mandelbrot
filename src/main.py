import numpy as np
import pygame as pg
import taichi as ti
from fractal import Fractal
from functions import mandelbrot_func, burning_ship, mandelbrot3

RES = WIDTH, HEIGHT = 1600, 800


platform = ti.cpu
ti.init(platform)

zoomIn = 0.990
zoomOut = 1.009
vel = 0.04


  
class App: 
    def __init__(self):
        self.clock = pg.time.Clock()

        self.main_screen = pg.display.set_mode((WIDTH,HEIGHT),pg.SCALED)
        self.mandelbrot = Fractal(WIDTH, HEIGHT, mandelbrot_func)
        self.draw()
    
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
    
    def draw(self):
        screen_array = self.mandelbrot.get_results()
        pg.surfarray.blit_array(self.main_screen, screen_array)

    def run(self):
        while True:
            self.control()
            self.mandelbrot.update()
            self.draw()
            pg.display.flip()

            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')



if __name__ == '__main__':
    app = App()
    app.run()