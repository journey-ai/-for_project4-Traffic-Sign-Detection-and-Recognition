# Importing the required libraries
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

# Loading desired images
img = load_img('images/t1.jpg')

# For processing, we are converting the image(s) to an array
data = img_to_array(img)

# Expanding dimension to one sample
samples = expand_dims(data, 0)

# Calling ImageDataGenerator for creating data augmentation generator.
datagen = ImageDataGenerator(rotation_range=90)
it = datagen.flow(samples, batch_size=1)
# Preparing the Samples and Plot for displaying output
for i in range(9):
	# preparing the subplot
	pyplot.subplot(330 + 1 + i)
	# generating images in batches
	batch = it.next()
	# Remember to convert these images to unsigned integers for viewing 
	image = batch[0].astype('uint8')
	# Plotting the data
	pyplot.imshow(image)
# Displaying the figure
pyplot.show()


datagen = ImageDataGenerator(width_shift_range=[-20,20])
it = datagen.flow(samples, batch_size=1)
# Preparing the Samples and Plot for displaying output
for i in range(9):
	# preparing the subplot
	pyplot.subplot(330 + 1 + i)
	# generating images in batches
	batch = it.next()
	# Remember to convert these images to unsigned integers for viewing 
	image = batch[0].astype('uint8')
	# Plotting the data
	pyplot.imshow(image)
# Displaying the figure
pyplot.show()



datagen = ImageDataGenerator(height_shift_range=0.4)
it = datagen.flow(samples, batch_size=1)
# Preparing the Samples and Plot for displaying output
for i in range(9):
	# preparing the subplot
	pyplot.subplot(330 + 1 + i)
	# generating images in batches
	batch = it.next()
	# Remember to convert these images to unsigned integers for viewing 
	image = batch[0].astype('uint8')
	# Plotting the data
	pyplot.imshow(image)
# Displaying the figure
pyplot.show()


datagen = ImageDataGenerator(horizontal_flip=True)
it = datagen.flow(samples, batch_size=1)
# Preparing the Samples and Plot for displaying output
for i in range(9):
	# preparing the subplot
	pyplot.subplot(330 + 1 + i)
	# generating images in batches
	batch = it.next()
	# Remember to convert these images to unsigned integers for viewing 
	image = batch[0].astype('uint8')
	# Plotting the data
	pyplot.imshow(image)
# Displaying the figure
pyplot.show()


datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
it = datagen.flow(samples, batch_size=1)
# Preparing the Samples and Plot for displaying output
for i in range(9):
	# preparing the subplot
	pyplot.subplot(330 + 1 + i)
	# generating images in batches
	batch = it.next()
	# Remember to convert these images to unsigned integers for viewing 
	image = batch[0].astype('uint8')
	# Plotting the data
	pyplot.imshow(image)
# Displaying the figure
pyplot.show()


datagen = ImageDataGenerator(zoom_range=[0.7,1.3])
it = datagen.flow(samples, batch_size=1)
# Preparing the Samples and Plot for displaying output
for i in range(9):
	# preparing the subplot
	pyplot.subplot(330 + 1 + i)
	# generating images in batches
	batch = it.next()
	# Remember to convert these images to unsigned integers for viewing 
	image = batch[0].astype('uint8')
	# Plotting the data
	pyplot.imshow(image)
# Displaying the figure
pyplot.show()



# Creating an iterator for data augmentation

datagen = ImageDataGenerator(shear_range=30)
it = datagen.flow(samples, batch_size=1)
# Preparing the Samples and Plot for displaying output
for i in range(9):
	# preparing the subplot
	pyplot.subplot(330 + 1 + i)
	# generating images in batches
	batch = it.next()
	# Remember to convert these images to unsigned integers for viewing 
	image = batch[0].astype('uint8')
	# Plotting the data
	pyplot.imshow(image)
# Displaying the figure
pyplot.show()

