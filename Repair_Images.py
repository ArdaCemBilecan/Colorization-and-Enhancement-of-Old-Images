import tensorflow as tf
from tensorflow.keras.models import *
import time
from matplotlib import pyplot as plt
from glob import glob
import numpy as np
import cv2


# This train is for LR to HR converting with pix2pix
# LR and HR images should be colorful images. Class will convert to HR images for Pix2pix.
class Pix2Pix:

    def __init__(self,input_img_path,real_img_path,test_img_path, output_img_path, model_save_path,loss_object,epoch):
        self.input_img_path = glob(input_img_path)
        self.real_img_path = glob(real_img_path)
        self.test_img_path = test_img_path
        self.output_img_path = output_img_path
        self.model_save_path = model_save_path
        self.epoch = epoch
        self.loss_object = loss_object
    
    
    def gray_imread(self,path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        return img
    

   
    def reshape(self,img):
      img = np.asarray(img)
      img = img.reshape(256,256,1)
      return img
    
    
    def load_images(self):
        input_images = []
        real_images = []
        for path in self.input_img_path:
            image = self.gray_imread(path)
            input_images.append(image)
        for path in self.real_img_path:
            image = self.gray_imread(path)
            real_images.append(image)
        
        test_img = self.gray_imread(self.test_img_path)

        for i in range(len(input_images)):
          input_images[i] = self.reshape(input_images[i])
          real_images[i] = self.reshape(real_images[i])
        
        test_img = self.reshape(test_img)
        
        return input_images , real_images , test_img
    
    def downsample(self,filters, size, apply_batchnorm=True):
      initializer = tf.random_normal_initializer(0., 0.02)

      result = tf.keras.Sequential()
      result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',kernel_initializer=initializer, use_bias=False))
      # Burada 2'ye bölüyoruz 256 --> 128
      if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

      result.add(tf.keras.layers.LeakyReLU())

      return result
  
    
    def upsample(self,filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
    
        result = tf.keras.Sequential()
        result.add(
          tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          use_bias=False))
      # burda da 2 kat arttırıyoruz
        result.add(tf.keras.layers.BatchNormalization())
    
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
    
        result.add(tf.keras.layers.ReLU())
    
        return result
    
    
    def Generator(self):
      inputs = tf.keras.layers.Input(shape=[256, 256, 1])

      down_stack = [
        self.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        self.downsample(128, 4),  # (batch_size, 64, 64, 128)
        self.downsample(256, 4),  # (batch_size, 32, 32, 256)
        self.downsample(512, 4),  # (batch_size, 16, 16, 512)
        self.downsample(512, 4),  # (batch_size, 8, 8, 512)
        self.downsample(512, 4),  # (batch_size, 4, 4, 512)
        self.downsample(512, 4),  # (batch_size, 2, 2, 512)
        self.downsample(512, 4)  # (batch_size, 1, 1, 512)
      ]

      up_stack = [
        self.upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        self.upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        self.upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        self.upsample(256, 4),  # (batch_size, 16, 16, 512)
        self.upsample(128, 4),  # (batch_size, 32, 32, 256)
        self.upsample(64, 4),  # (batch_size, 64, 64, 128)
        self.upsample(32, 4) # (batch_size, 128, 128 64)
      ]
      initializer = tf.random_normal_initializer(0., 0.02)
      last = tf.keras.layers.Conv2DTranspose(3, 4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=initializer,
                                             activation='tanh')  # (batch_size, 256, 256, 3)
    # Build U-NET
      x = inputs

      # Downsampling through the model
      skips = []
      for down in down_stack:
        x = down(x)
        skips.append(x)

      skips = reversed(skips[:-1]) # son elemani almadan terste yazdirir

      # Upsampling and establishing the skip connections
      for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

      x = last(x)

      return tf.keras.Model(inputs=inputs, outputs=x)
  
    
  
    def generator_loss(self,disc_generated_output, gen_output, target):
      LAMBDA = 100
      gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
      # Mean absolute error
      l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
      total_gen_loss = gan_loss + (LAMBDA * l1_loss)
      return total_gen_loss, gan_loss, l1_loss
  
    
  
    
    def Discriminator(self):
      initializer = tf.random_normal_initializer(0., 0.02)
      inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image') 
      tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image') 
    
      x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 512, 512, channels*2)
    
      down1 = self.downsample(64, 4,False)(x) # (batch_size, 128, 128, 64)
      down2 = self.downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
      down3 = self.downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)
    
      zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
      conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)
    
      batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    
      leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    
      zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
    
      last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)
    
      return tf.keras.Model(inputs=[inp, tar], outputs=last)
  
    
  

    def discriminator_loss(self,disc_real_output, disc_generated_output):
      real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
    
      generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    
      total_disc_loss = real_loss + generated_loss 
    
      return total_disc_loss
        


        
    
    def train_step(self,input_image, target,generator_optimizer,discriminator_optimizer,generator,discriminator):
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        print(input_image.shape)
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

      generator_gradients = gen_tape.gradient(gen_total_loss,
                                              generator.trainable_variables)
      discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   discriminator.trainable_variables)

      generator_optimizer.apply_gradients(zip(generator_gradients,
                                              generator.trainable_variables))
      discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  discriminator.trainable_variables))
      
        
    
    def generate_optimizers(self):
        generator_optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam") #SGD kullanabiliriz
        discriminator_optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
        return generator_optimizer , discriminator_optimizer
    
    
    
    def generate_images(self,model, test_input,step):
        prediction = model(test_input, training=True)
        pre = prediction[0]
        pre = (pre+1)*127.5
        pre = np.uint8(pre)
        name = self.output_img_path+'Repaired_{step}.png'.format(step=step)
        plt.imsave(name,pre,cmap='gray')
        


    def saveGenerator(self,generator,total_loop):
        name = self.model_save_path+'{total_loop}.h5'.format(total_loop = total_loop)
        generator.save(name)
        
    
     
    def fit(self):
      total_loop = 0
      input_images , real_images , test_img = self.load_images()
      generator = self.Generator()
      discriminator = self.Discriminator()
      generator_optimizer , discriminator_optimizer = self.generate_optimizers()
      start = time.time()
      step = 0
      i = 0
      test_img = np.array(test_img).reshape(1,256,256,1)
      input_images = np.array(input_images).reshape(-1,1,256,256,1)
      real_images = np.array(real_images).reshape(-1,1,256,256,1)

      while step < self.epoch:
        print("Step = ",step)
        while i < len(input_images):
              self.train_step(input_images[i], real_images[i],generator_optimizer,discriminator_optimizer,generator,discriminator)
              if (total_loop%2 == 0):
                  print('i= ',i)
                  print(f'Time taken for 2000 steps: {time.time()-start:.2f} sec\n')
                  self.generate_images(generator,test_img,total_loop)
                  start = time.time()
              i +=1
              total_loop = total_loop+1
        if step % 10 == 0 and step != 0:
            self.saveGenerator(total_loop)
        step+=1
        i = 0
      self.generate_images(generator,test_img,step)


    
    
    
lr_img_path = 'your_path'
hr_img_path = 'your_path'
test_img_path = 'your_path'
output = 'your_path'
model_save = 'your_path'
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
pix2pix = Pix2Pix(lr_img_path,hr_img_path,test_img_path,output,model_save,loss_object,5)
pix2pix.fit()
        
#input_images , real_images , test_img =  pix2pix.load_images()
    
