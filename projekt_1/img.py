from PIL import Image 
import numpy as np
 
im2 = Image.open('./projekt_1/b.png').convert('L')                  
im2 = np.array(im2) 