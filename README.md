## Mandelbrot Zoomer in Python with PyGame and Taichi
Runs with `python src/main.py`

Change run parameters in `config.py`
### Dependencies:
- Pygame
- Numpy
- Taichi
For details of the implementation and mathematical background, see projekt.pdf and projekt.ipynb

### Controls:
- Zoom Mode:
	- `WASD` : move
	- `Arrow Up/Down`: zoom in/out
	- `Ctrl+R`: reset view
	- `I`: show debug information
-  Julia / Orbit Mode: left click to show Julia Set / Orbit Graph
- Toggle Modes:
	- `J`: toggle Julia Mode
	- `O`: toggle Orbit Mode
- Switch Mandelbrots:
	- `1`: Burning Ship Fractal
	- `2-7`: Mandelbrot Powers z = z^n + c


### Credits:
- Xaos-Algorithm Implementation: https://github.com/ttsiodras/MandelbrotSSE
- Palettes: https://iquilezles.org/articles/palettes/