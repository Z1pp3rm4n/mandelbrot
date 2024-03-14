## Mandelbrot Zoomer in Python with PyGame and Taichi
For details of the implementation and mathematical background, see projekt.pdf

### Controls:
- Zoom Mode:
	- `WASD` : move
	- `Arrow Up/Down`: zoom in/out
	- `Ctrl+R`: reset view
	- `I`: show debug information
-  Julia / Orbit Mode: right click to show Julia Set / Orbit Graph
- Toggle Modes:
	- `J`: toggle Julia Mode
	- `O`: toggle Orbit Mode
- Switch Mandelbrots:
	- `1`: Burning Ship Fractal
	- `2-7`: Mandelbrot Powers z = z^n + c

### Dependencies:
- Pygame
- Numpy
- Taichi

### Credits:
- Xaos-Algorithm Implementation: https://github.com/ttsiodras/MandelbrotSSE
- Palettes: https://iquilezles.org/articles/palettes/