from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
import numpy as np, Image
import sys, time, os
import pycuda.driver as cuda
import pycuda.gl as cuda_gl
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
#import pycuda.curandom as curandom
import pyglew as glew

##Add Modules from other directories
##sys.path.append( "/home/bruno/Desktop/Dropbox/Developer/pyCUDA/myLibraries" )
##from myTools import *


#dev = setDevice()
#cuda_gl.make_context(dev)

nWidth = 512*2*2
nHeight = 512*2
nData = nWidth*nHeight



gl_Tex = None
gl_PBO = None
cuda_POB = None
cuda_POB_ptr = None
colorMap_rgba_d = None
plot_rgba_d = None
plotData_d = None

get_rgbaKernel = None
copyKernel = None
block2D = (16,16, 1)
grid2D = (nWidth/block2D[0], nHeight/block2D[1] ) 



frames = 0
plot_rgba = np.zeros(nData)
solid = np.ones(nData)
nCol = 236
maxVar = 1.
minVar = 0.
nCol = np.int32( nCol )
minVar = np.float32( minVar )
maxVar = np.float32( maxVar )

frameCount = 0
fpsCount = 0
fpsLimit = 8
timer = 0.0


def initCUDA():
  global get_rgbaKernel, copyKernel
  cudaAnimCode = SourceModule('''
  #include <stdint.h>
  #include <cuda.h>

  __global__ void get_rgba_kernel (int ncol, float minvar, float maxvar, float *plot_data, unsigned int *plot_rgba_data,
				  unsigned int *cmap_rgba_data){
  // CUDA kernel to fill plot_rgba_data array for plotting    
    int t_i = blockIdx.x*blockDim.x + threadIdx.x;
    int t_j = blockIdx.y*blockDim.y + threadIdx.y;
    int tid = t_i + t_j*blockDim.x*gridDim.x;

    float frac = (plot_data[tid]-minvar)/(maxvar-minvar);
    int icol = (int)(frac * ncol);
    plot_rgba_data[tid] = cmap_rgba_data[icol];
  }
  
  ''')
  get_rgbaKernel = cudaAnimCode.get_function('get_rgba_kernel')
  print "CUDA 2D animation initialized"

  
def initData():  
  global plot_rgba_d, colorMap_rgba_d, plotData_d
  #print "Loading Color Map"
  colorMap = np.loadtxt("cmap.dat")
  colorMap_rgba = []
  for i in range(colorMap.shape[0]):
    r, g, b = colorMap[i]
    colorMap_rgba.append( int(255)<<24 | int(b*255)<<16 | int(g*255)<<8 | int(r*255)<<0 )
  colorMap_rgba = np.array(colorMap_rgba)
  colorMap_rgba_h = np.array(colorMap_rgba).astype(np.uint32)
  colorMap_rgba_d = gpuarray.to_gpu( colorMap_rgba_h )
  plot_rgba_h = np.zeros(nData).astype(np.uint32)
  plot_rgba_d = gpuarray.to_gpu( plot_rgba_h )
  plotData_h = np.random.rand(nData).astype(np.float32)
  plotData_d = gpuarray.to_gpu(plotData_h)

def get_rgba( ptr ):
  #global plotData_d, plot_rgba_d, colorMap_rgba_d
  get_rgbaKernel(nCol, minVar, maxVar, plotData_d, np.intp(ptr), colorMap_rgba_d, grid=grid2D, block=block2D )



def keyPressed(*args):
  ESCAPE = '\033'
  # If escape is pressed, kill everything.
  if args[0] == ESCAPE:
    print "Ending Simulation"
    #cuda.Context.pop()
    sys.exit()

def stepFunc():
  #print "Default step function"
  return 0


def displayFunc():
  global plot_rgba
  global frames, timer
  global cuda_POB
 
  timer = time.time()
  frames += 1
  
  stepFunc()
  
  
  cuda_POB_map = cuda_POB.map()
  cuda_POB_ptr, cuda_POB_size = cuda_POB_map.device_ptr_and_size()

  get_rgba( cuda_POB_ptr ) 
  cuda_POB_map.unmap()
  
  glClear(GL_COLOR_BUFFER_BIT) # Clear
  glBindTexture(GL_TEXTURE_2D, gl_Tex)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO)
  
  #NON-GL_INTEROPITAL
  #plot_rgba = plot_rgba_d.get()
  ##Fill the pixel buffer with the plot_rgba array
  #glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, plot_rgba.nbytes, plot_rgba, GL_STREAM_COPY)
  
  # Copy the pixel buffer to the texture, ready to display
  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,nWidth,nHeight,GL_RGBA,GL_UNSIGNED_BYTE,None)  
  
  #Render one quad to the screen and colour it using our texture
  #i.e. plot our plotvar data to the screen
  glClear(GL_COLOR_BUFFER_BIT)
  glBegin(GL_QUADS)
  glTexCoord2f (0.0, 0.0)
  glVertex3f (0.0, 0.0, 0.0)
  glTexCoord2f (1.0, 0.0)
  glVertex3f (nWidth, 0.0, 0.0)
  glTexCoord2f (1.0, 1.0)
  glVertex3f (nWidth, nHeight, 0.0)
  glTexCoord2f (0.0, 1.0)
  glVertex3f (0.0, nHeight, 0.0)
  glEnd()
  timer = time.time()-timer
  computeFPS()
  glutSwapBuffers()

GL_initialized = False  
def initGL():
  global GL_initialized
  if GL_initialized: return
  glutInit()
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
  glutInitWindowSize(nWidth, nHeight)
  glutInitWindowPosition(50, 50)
  glutCreateWindow("Window")
  glew.glewInit()
  glClearColor(0.0, 0.0, 0.0, 0.0)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glOrtho(0,nWidth,0.,nHeight, -200.0, 200.0)
  GL_initialized = True
  print "OpenGL initialized"
  

  


def createPBO():
  global gl_Tex
  #Create texture which we use to display the result and bind to gl_Tex
  glEnable(GL_TEXTURE_2D)
  gl_Tex = glGenTextures(1)
  glBindTexture(GL_TEXTURE_2D, gl_Tex)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, nWidth, nHeight, 0, 
		GL_RGBA, GL_UNSIGNED_BYTE, None);
  #print "Texture Created"
# Create pixel buffer object and bind to gl_PBO. We store the data we want to
# plot in memory on the graphics card - in a "pixel buffer". We can then 
# copy this to the texture defined above and send it to the screen
  global gl_PBO, cuda_POB
  
  gl_PBO = glGenBuffers(1)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO)
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, nWidth*4*nHeight, None, GL_STREAM_COPY)
  cuda_POB = cuda_gl.RegisteredBuffer(long(gl_PBO))
  #cuda_POB_map = cuda_POB.map()
  #cuda_POB_ptr, cuda_POB_size = cuda_POB_map.device_ptr_and_size()
  #print "Buffer Created"

def computeFPS():
    global frameCount, fpsCount, fpsLimit, timer
    frameCount += 1
    fpsCount += 1
    if fpsCount == fpsLimit:
        ifps = 1.0 /timer
        glutSetWindowTitle("CUDA 2D animation: %f fps" % ifps)
        fpsCount = 0

def startGL():
  glutDisplayFunc(displayFunc)
  #glutReshapeFunc(resize)
  glutIdleFunc(displayFunc)
  glutKeyboardFunc( keyPressed )
  #import pycuda.autoinit
  print "Starting GLUT main loop..."
  glutMainLoop()


def animate():
  if not GL_initialized: initGL()
  import pycuda.gl.autoinit
  initCUDA()
  createPBO()
  initData()
  startGL()

  
