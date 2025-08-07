from PIL import Image
  
# Import an image from directory:
input_image = Image.open("/media/gabsp00/4202077D02077567/Backup_ssd/Gabslau/TCC/ambvirlx/Teste_Keras_MDTS/instance-level_human_parsing/instance-level_human_parsing/Validation/Categories/000000213.png")
input_image =  input_image.convert("RGB")
  
# Extracting pixel map:
pixel_map = input_image.load()
  
# Extracting the width and height 
# of the image:
width, height = input_image.size
  
# trying to change colors
for i in range(width):
    for j in range(height):
        if (pixel_map[i, j] == (0,128,0)):
        	pixel_map[i, j] = (2,2,2)
        elif (pixel_map[i, j] ==(128,0,0)):
        	pixel_map[i, j] = (1,1,1)
        elif (pixel_map[i, j] == (128,128,0)):
        	pixel_map[i, j] = (3,3,3)
  
# Saving the final output
input_image.save("000000213.png", format="png")
  
# use input_image.show() to see the image on the
# output screen.
#input_image.show()
