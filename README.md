# Colorization-and-Enhancement-of-Old-Images-
Colorization and Enhancement of Old Images Using Generative Adversarial Networks(GAN) with Pix2pix algorithm.
![Pix2pix_Architecture](https://user-images.githubusercontent.com/44208327/172026753-410766a6-9e1a-420a-8dbc-8ed4c6145161.png)

That is pix2pix architecture.


This project is for colorization,repair,enhancement of images. Also we classify images for indoor , outdoor and face.
We did classification because we do not have images enough that's why we classify the image then added model weights to make better colorization.
Besides we did face detection thus we can make firm rate of face in the image. If the rate under of 60% in images the model won't do face colorization. Also if the conditions are not met, the user will be notified.( like 'That is not suitable for face colorization.') Because our purpose is colorization just portrait.

Finally we made image enhancement. It is for converting low resolution image to high resolution image.

As a Result:

![Colorization py](https://user-images.githubusercontent.com/44208327/172026615-868d7499-beca-4651-981d-4a543342190c.png)

(a) is original image , (b) is colorized image. As you can see the result is so close the original image.


![sekil7](https://user-images.githubusercontent.com/44208327/172026629-ee543820-2f31-4534-a7f8-37dcd2bb7187.png)

(a) is gray image (b) is colorized image , (c) is enhancement image.


![face_colorization](https://user-images.githubusercontent.com/44208327/172026644-bcd93e26-03c5-4718-95f9-5369842b93a1.png)

(a) is a original image , (b) is a colorized image. You have to attention just the face. The model can not colorize in the back well.

![image_repair](https://user-images.githubusercontent.com/44208327/172026685-8c2d9662-cfa7-4fa8-8ee0-9e6e2ca1b78a.png)

(a) is a original image , (b) is a repaired image , (c) is a corrupted image. The result is so similar to the original.

![face_detection](https://user-images.githubusercontent.com/44208327/172026766-4eeef3f8-7386-4a4e-b197-1880e7b48a0e.png)

Face detection.
