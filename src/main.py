import numpy as np
import pygame as pg
import taichi as ti
from sys import exit
from config import platform, RES, WIDTH, HEIGHT, MAX_FPS, zoomIn, zoomOut, vel
from fractal import Mandelbrot
from enum import Enum
import functions as func



ti.init(platform)
pg.init()
font = pg.font.Font(pg.font.get_default_font(), 20)

class Mode(Enum):
    ZOOM = 0
    FAST_JULIA = 1
    ORBIT = 2
  
class App: 
    def __init__(self):
        self.clock = pg.time.Clock()

        self.main_screen = pg.display.set_mode((WIDTH,HEIGHT),pg.SCALED)
        self.sub_screen = pg.Surface((HEIGHT//2, HEIGHT//2))
        self.mandelbrot = Mandelbrot(WIDTH, HEIGHT, func.mandelbrot2, exponent=2, bailout=2)
        self.mandelbrot.render_all()
        self.draw_mandelbrot()

        self.show_status = False
        self.mode = Mode.ZOOM
    
    def handle_events(self):
        """
        Handles one-time keypress events such as: Mode switching, Toggle Information, Quitting
        """
        events = pg.event.get()
        for e in events:
            if e.type == pg.QUIT:
                    pg.quit()
                    exit()   
            if e.type == pg.KEYDOWN:                 

                if e.key == pg.K_i:
                    self.show_status = not self.show_status
                if e.key == pg.K_r and pg.key.get_mods() & pg.KMOD_CTRL:
                    self.mandelbrot.reset_scale()
                    self.draw_mandelbrot()
                self.control_mode(e)
                self.control_mandelbrot_type(e)

    def control_mode(self, e):
        """
        Switch between run modes: 
        J -> FAST_JULIA
        O -> ORBIT
        """
        if e.key == pg.K_j: 
            if self.mode != Mode.FAST_JULIA:
                self.mode = Mode.FAST_JULIA
                self.draw_mandelbrot()
            else:
                self.mode = Mode.ZOOM
        if e.key == pg.K_o:
            if self.mode != Mode.ORBIT:
                self.mode = Mode.ORBIT
                self.draw_mandelbrot()
            else:
                self.mode = Mode.ZOOM        

    def control_mandelbrot_type(self, e):
        """
        Switch to other mandelbrot fractals if the pressed key is a number n from 1-7
        n = 1: burning ship 
        n = 2-7: Mandelbrot power^n
        Args:
            e: Key Down Event
        """
        f = None
        exponent = 0
        match e.key:
            case pg.K_1: 
                f = func.burning_ship
                exponent = 2
            case pg.K_2:
                f = func.mandelbrot2
                exponent = 2
            case pg.K_3: 
                f = func.mandelbrot3
                exponent = 3
            case pg.K_4:
                f= func.mandelbrot4
                exponent = 4
            case pg.K_5:
                f = func.mandelbrot5
                exponent = 5
            case pg.K_6:
                f = func.mandelbrot6
                exponent = 6
            case pg.K_7:
                f = func.mandelbrot7
                exponent = 7
        if f != None:
            self.mandelbrot = Mandelbrot(WIDTH, HEIGHT, f, exponent=exponent)
            self.mandelbrot.render_all()
            self.draw_mandelbrot()


    def draw_status(self):
        """
        Shows current status:
        """
        xid, yid = pg.mouse.get_pos()
        x, y = self.mandelbrot.get_coords(xid,yid)
        zoom_level = 1/self.mandelbrot.scale
        exponent = self.mandelbrot.exponent
        bailout = self.mandelbrot.bailout
        max_iter = self.mandelbrot.max_iter

        texts = [f"x = {x}",
            f"y = {y}",
            f"Zoom level = {zoom_level:.2E}",
            f"Exponent = {exponent}",
            f"Bailout = {bailout}",
            f"Max Iter = {max_iter}"]
        
        y_offset = HEIGHT * 2 // 3
        # Draws each line on separate rows
        for line in texts:
            line_surface = font.render(line, True, pg.Color("white"), pg.Color("dodgerblue"))
            self.main_screen.blit(line_surface, (0, y_offset))
            line_height = line_surface.get_height()
            y_offset += line_height


    def orbit_loop(self):
        """
        On mouse right click: show the orbit of the number at that coordinate
        """
        if pg.mouse.get_pressed()[0]:
            xid,yid = pg.mouse.get_pos()
            orbit = self.mandelbrot.get_orbit(xid, yid)
            self.draw_mandelbrot()
            for p in orbit:
                x = p[0]
                y = p[1]
                if 0 <= x and x <= WIDTH and 0 <= y and y <= HEIGHT:
                    pg.draw.circle(self.main_screen, pg.Color('darkorange3'), (x,y), radius=5)
                                  
    
    def control_movement(self):
        """
        Control the movement in ZOOM mode:
        WASD: move 
        Up/Down Arrow: Zoom In / Out
        """
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
        """
        """
        self.control_movement()
        if platform == ti.cpu:
            self.mandelbrot.render_xaos()
        if platform == ti.gpu:
            self.mandelbrot.render_all()
        self.draw_mandelbrot()


    def fast_julia_loop(self):
        """
        On mouse right click: Creates a mini Julia Fractal at top left of screen
        """
        if pg.mouse.get_pressed()[0]:
            xid,yid = pg.mouse.get_pos()
            cx, cy = self.mandelbrot.get_coords(xid,yid)
            julia = self.mandelbrot.create_julia(cx,cy)
            julia.render_all()
            julia_results = julia.get_results()
            self.draw_julia(julia_results)
            

    def draw_julia(self, julia_results):
        pg.surfarray.blit_array(self.sub_screen, julia_results)
        self.main_screen.blit(self.sub_screen, dest=(0,0))
        

    def draw_mandelbrot(self):
        screen_array = self.mandelbrot.get_results()
        pg.surfarray.blit_array(self.main_screen, screen_array)

    def run(self):
        while True:
            self.handle_events()
            if self.mode == Mode.FAST_JULIA:
                self.fast_julia_loop()
            elif self.mode == Mode.ORBIT:
                self.orbit_loop()
            else:
                self.zoom_loop()
            
            if self.show_status:
                self.draw_status()

            pg.display.flip() # update screen
            self.clock.tick(MAX_FPS) # limit FPS to MAX_FPS
            pg.display.set_caption(f'Mode: {self.mode.name} FPS: {self.clock.get_fps() :.2f}')



if __name__ == '__main__':
    app = App()
    app.run()