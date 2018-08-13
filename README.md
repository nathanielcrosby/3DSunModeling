# 3DSunModeling
This turns 2D images of the Sun into 3D printable models in .stl file form.
I created this while interning at the Harvard-Smithsonian Center for Astrophysics.
It is intended to create hemispherical 3D prints of the whole sun and flat 3D prints of specific features of the sun (ex. solar flares).
These are to be used for educational purposes, specifically for the visually impaired.

## Getting Started
 ### Prerequisites
  - os
  - subprocess
  - numpy
  - matplotlib
  - scipy
  - skimage
  - stl_tools
  - numpy-stl
  - imageio
  
### Installing
  Still working on making into an actual library...
  
### Examples
All Commands Start With:
```
  date = '2014/05/16'
  r = 460. #px

  image, header, x, y, z = 3DSunModeling.3DPlot.image_to_xyz_mesh(date, r=r, base_len=228.6, offset_x=0., offset_y=0., 
  scale_factor_percent=0.25, minimum_intensity_threshold=0.45, buffer_zone=0., buffer_val=1.1,
  buffer=True, exp=2.0, scale_bool=True, earth=True, interval=2., local=True, index=2, flat_base=False)
```
To Do any of the Following just add these lines:

3D Full Sun .stl File
  ```
  3DSunModeling.3DPlot.stl_mesh_maker(x, y, z, interval=1, fname='test1.stl')
  ```
  
  3D Full Sun Movie
  ```
  3DSunModeling.3DPlot.make_movie(x, y, z, image, file='movie.gif', fps=30, st_ang=0, en_ang=360, 
  st_elev_ang=90, en_elev_ang=0, time=10)
  ```
  
  3D Full Sun 3D Plot
  ```
  3DSunModeling.3DPlot.ThreeDPlot(x, y, z, image, stride=10, figx=10., figy=10., save=False, file='3d.png')
  ```
  
  2D Full Sun .stl File
  ```
  3DSunModeling.3DPlot.image_to_flat_stl(date, scale=100., width=100., depth=100., height=20.)
  ```
  
### Authors
  Nathaniel Crosby
  

