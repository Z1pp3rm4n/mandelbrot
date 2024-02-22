import numpy as np
import pygame as pg
import numba 

res = width, height = 800,450
offset = np.array([1.3* width, height]) // 2
max_iter = 30
zoom = 2.2 / height

texture = pg.image.load('texture.jpg')
texture_size = min(texture.get_size()) - 1
texture_array = pg.surfarray.array3d(texture)

class Fractal:
    def __init__(self, func):
        pixel_array = np.full((width,height,3), [0,0,0], dtype=np.uint32)
        pass

    def update(self):
        pass


class FractalScreen:
    def __init__(self, fractal, screen):
        self.screen = screen
        self.fractal = fractal

    def render(self):
        pass


class CopyFractal: 
    def __init__(self, app):
        self.app = app
        self.screen_array = np.full((width,height,3), [0,0,0], dtype=np.uint8)
        self.x = np.linspace(0,width,width, dtype=np.float32)
        self.y = np.linspace(0,height,height, dtype=np.float32)

        # self.col = np.empty((width,height))
    
    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def render(screen_array):
        for x in numba.prange(width):
            for y in numba.prange(height):
                c = (x - offset[0]) * zoom + 1j * (y-offset[1])*zoom
                z = 0
                num_iter = 0
                for i in range(max_iter):
                    z = z ** 2 + c
                    if z.real ** 2 + z.imag ** 2 > 4:
                        break
                    num_iter += 1
                col = int(texture_size * num_iter /max_iter)
                screen_array[x,y] = texture_array[col,col]
        return screen_array
        

    def update(self):
        self.screen_array = self.render(self.screen_array)

    def draw(self):
        pg.surfarray.blit_array(self.app.main_screen, self.screen_array)

    def run(self):
        self.update()
        self.draw()



class App: 
    def __init__(self):
        self.clock = pg.time.Clock()

        self.main_screen = pg.display.set_mode((width,height),pg.SCALED)
        self.fractal = CopyFractal(self)

        

        pass

    def run(self):
        while True:
            self.main_screen.fill('black')
            
            # Test
            self.fractal.run()
            pg.display.flip()



            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')



if __name__ == '__main__':
    app = App()
    app.run()




    








