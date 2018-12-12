import tensorflow as tf
tf.enable_eager_execution()

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import PIL

BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 100
OUTPUT_CHANNELS = 3
checkpoint_dir = './training_checkpoints'
output_dir = './generated_images'


class Downsample(tf.keras.Model):
    
    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()
  
    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Upsample(tf.keras.Model):
    
    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                       (size, size),
                                                       strides=2,
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x1, x2, training):
        x = self.up_conv(x1)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)
        if x2 is not None:
            x = tf.concat([x, x2], axis=-1)
        return x


class ResidualBlock(tf.keras.Model):
    
    def __init__(self, filters, size):
        super(ResidualBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            use_bias=False)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
  
    def call(self, x, training):
        y = self.conv1(x)
        y = self.batchnorm1(y, training=training)
        y = tf.nn.relu(y)
        y = self.conv2(y)
        y = self.batchnorm2(y, training=training)
        y = tf.math.add(x, y)
        return y


class UNetGenerator(tf.keras.Model):
    
    def __init__(self, noise=False):
        super(UNetGenerator, self).__init__()
        self.noise_inputs = noise
        initializer = tf.random_normal_initializer(0., 0.02)
    
        self.down1 = Downsample(64, 4, apply_batchnorm=False)
        self.down2 = Downsample(128, 4)
        self.down3 = Downsample(256, 4)
        self.down4 = Downsample(512, 4)
        self.down5 = Downsample(512, 4)
        self.down6 = Downsample(512, 4)
        self.down7 = Downsample(512, 4)
        self.down8 = Downsample(512, 4)

        self.up1 = Upsample(512, 4, apply_dropout=True)
        self.up2 = Upsample(512, 4, apply_dropout=True)
        self.up3 = Upsample(512, 4, apply_dropout=True)
        self.up4 = Upsample(512, 4)
        self.up5 = Upsample(256, 4)
        self.up6 = Upsample(128, 4)
        self.up7 = Upsample(64, 4)

        self.last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS,
                                                    (4, 4),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer)
  
    @tf.contrib.eager.defun
    def call(self, x, training):
        # x shape == (bs, 256, 256, 3)
        if self.noise_inputs:
            z = tf.random_normal(shape=[BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1])
            x = tf.concat([x, z], axis=-1) # (bs, 256, 256, 4)
        
        x1 = self.down1(x, training=training) # (bs, 128, 128, 64)
        x2 = self.down2(x1, training=training) # (bs, 64, 64, 128)
        x3 = self.down3(x2, training=training) # (bs, 32, 32, 256)
        x4 = self.down4(x3, training=training) # (bs, 16, 16, 512)
        x5 = self.down5(x4, training=training) # (bs, 8, 8, 512)
        x6 = self.down6(x5, training=training) # (bs, 4, 4, 512)
        x7 = self.down7(x6, training=training) # (bs, 2, 2, 512)
        x8 = self.down8(x7, training=training) # (bs, 1, 1, 512)

        x9 = self.up1(x8, x7, training=training) # (bs, 2, 2, 1024)
        x10 = self.up2(x9, x6, training=training) # (bs, 4, 4, 1024)
        x11 = self.up3(x10, x5, training=training) # (bs, 8, 8, 1024)
        x12 = self.up4(x11, x4, training=training) # (bs, 16, 16, 1024)
        x13 = self.up5(x12, x3, training=training) # (bs, 32, 32, 512)
        x14 = self.up6(x13, x2, training=training) # (bs, 64, 64, 256)
        x15 = self.up7(x14, x1, training=training) # (bs, 128, 128, 128)

        x16 = self.last(x15) # (bs, 256, 256, 3)
        x16 = tf.nn.tanh(x16)

        return x16


class ResNet9Generator(tf.keras.Model):
    
    def __init__(self, noise=False):
        super(ResNet9Generator, self).__init__()
        self.noise_inputs = noise
        initializer = tf.random_normal_initializer(0., 0.02)
        
        self.init = tf.keras.layers.Conv2D(64,
                                           (7, 7),
                                           padding='same',
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        
        self.down1 = Downsample(128, 4)
        self.down2 = Downsample(256, 4)
        
        self.res1 = ResidualBlock(256, 3)
        self.res2 = ResidualBlock(256, 3)
        self.res3 = ResidualBlock(256, 3)
        self.res4 = ResidualBlock(256, 3)
        self.res5 = ResidualBlock(256, 3)
        self.res6 = ResidualBlock(256, 3)
        self.res7 = ResidualBlock(256, 3)
        self.res8 = ResidualBlock(256, 3)
        self.res9 = ResidualBlock(256, 3)

        self.up1 = Upsample(128, 4)
        self.up2 = Upsample(64, 4)

        self.last = tf.keras.layers.Conv2D(OUTPUT_CHANNELS,
                                           (7, 7),
                                           padding='same',
                                           kernel_initializer=initializer)
  
    @tf.contrib.eager.defun
    def call(self, x, training):
        # x shape == (bs, 256, 256, 3)
        if self.noise_inputs:
            z = tf.random_normal(shape=[BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1])
            x = tf.concat([x, z], axis=-1) # (bs, 256, 256, 4)
        
        x1 = self.init(x) # (bs, 256, 256, 64)
        x1 = self.batchnorm(x1, training=training)
        x1 = tf.nn.relu(x1)
        
        x2 = self.down1(x1, training=training) # (bs, 128, 128, 128)
        x3 = self.down2(x2, training=training) # (bs, 64, 64, 256)
        
        x4 = self.res1(x3, training=training)
        x4 = self.res2(x4, training=training)
        x4 = self.res3(x4, training=training)
        x4 = self.res4(x4, training=training)
        x4 = self.res5(x4, training=training)
        x4 = self.res6(x4, training=training)
        x4 = self.res7(x4, training=training)
        x4 = self.res8(x4, training=training)
        x4 = self.res9(x4, training=training)
        
        x5 = self.up1(x4, None, training=training) # (bs, 128, 128, 128)
        x6 = self.up2(x5, None, training=training) # (bs, 256, 256, 64)

        x7 = self.last(x6) # (bs, 256, 256, 3)
        x7 = tf.nn.tanh(x7)

        return x7


class PatchDiscriminator(tf.keras.Model):
    
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
    
        self.down1 = Downsample(64, 4, False)
        self.down2 = Downsample(128, 4)
        self.down3 = Downsample(256, 4)
    
        # we are zero padding here with 1 because we need our shape to 
        # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512, 
                                           (4, 4), 
                                           strides=1, 
                                           kernel_initializer=initializer, 
                                           use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
    
        # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.last = tf.keras.layers.Conv2D(1, 
                                           (4, 4), 
                                           strides=1,
                                           kernel_initializer=initializer)
  
    @tf.contrib.eager.defun
    def call(self, inp, tar, training):
        # concatenating the input and the target
        x = tf.concat([inp, tar], axis=-1) # (bs, 256, 256, channels*2)
        x = self.down1(x, training=training) # (bs, 128, 128, 64)
        x = self.down2(x, training=training) # (bs, 64, 64, 128)
        x = self.down3(x, training=training) # (bs, 32, 32, 256)

        x = self.zero_pad1(x) # (bs, 34, 34, 256)
        x = self.conv(x)      # (bs, 31, 31, 512)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad2(x) # (bs, 33, 33, 512)
        # don't add a sigmoid activation here since
        # the loss function expects raw logits.
        x = self.last(x)      # (bs, 30, 30, 1)

        return x


class C_GAN():
    
    def __init__(self, generator, discriminator, generator_learning_rate=2e-4, discriminator_learning_rate=2e-4, ckpt_freq=20, name=None):
        self.name = ('' if name is None else name + '_') + generator.__class__.__name__ + '_' + discriminator.__class__.__name__
        self.generator = generator
        self.discriminator = discriminator
        self.ckpt_freq = ckpt_freq
        
        self.generator_optimizer = tf.train.AdamOptimizer(generator_learning_rate, beta1=0.5)
        self.discriminator_optimizer = tf.train.AdamOptimizer(discriminator_learning_rate, beta1=0.5)
        
        self.checkpoint_prefix = os.path.join(checkpoint_dir, self.name + '_ckpt')
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        
    
    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_generated_output),
                                                   logits = disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (LAMBDA * l1_loss)

        return total_gen_loss
    

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_real_output), 
                                                    logits = disc_real_output)
        generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.zeros_like(disc_generated_output), 
                                                         logits = disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
    
    
    def generate_image(self, input_image, target, name=None):
        gen_output = self.generator(input_image, training=True)
        plt.figure(figsize=(15,15))

        display_list = [input_image[0], target[0], gen_output[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')

        plt.show()
        if name is not None:
            plt.savefig(os.path.join(output_dir, self.name + name + '.png'))
        
    
    def restore_from_checkpoint(self):
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        self.checkpoint.restore(checkpoint_file)
    
    
    def train(self, dataset, epochs):
        training_stats = []
        
        for epoch in range(epochs):
            avg_gen_loss = 0
            avg_disc_loss = 0
            it = 0
            start = time.time()
            
            for target, input_image in dataset:
                it += 1
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    gen_output = self.generator(input_image, training=True)
                    
                    disc_real_output = self.discriminator(input_image, target, training=True)
                    disc_generated_output = self.discriminator(input_image, gen_output, training=True)
                    
                    gen_loss = self.generator_loss(disc_generated_output, gen_output, target)
                    disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
                    avg_gen_loss += gen_loss.numpy()
                    avg_disc_loss += disc_loss.numpy()
                
                generator_gradients = gen_tape.gradient(gen_loss, 
                                                        self.generator.variables)
                discriminator_gradients = disc_tape.gradient(disc_loss, 
                                                             self.discriminator.variables)
                
                self.generator_optimizer.apply_gradients(zip(generator_gradients, 
                                                             self.generator.variables))
                self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                                 self.discriminator.variables))
            
            time_taken = time.time() - start
            avg_gen_loss /= it
            avg_disc_loss /= it
            training_stats.append((epoch + 1, time_taken, avg_gen_loss, avg_disc_loss))
            
            if (epoch + 1) % self.ckpt_freq == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                with open(self.name + 'training_stats.pickle', 'wb') as f:
                    pickle.dump(training_stats, f)
                
            
            print('Time taken for epoch {} is {} sec. Gen_Loss: {} Disc_Loss: {} '.format(epoch + 1, time_taken, avg_gen_loss, avg_disc_loss))
    
    
    def evaluate(self, dataset):
        avg_mse = 0
        it = 0
        
        for target, input_image in dataset:
            it += 1
            gen_output = self.generator(input_image, training=True)
            avg_mse += tf.reduce_mean(tf.squared_difference(target, gen_output)).numpy()
        
        avg_mse /= it
        return avg_mse
    

    def summary(self):
        try:
            print('GAN:', self.name)
            print()
            print('Generator Summary')
            print('-----------------')
            self.generator.summary()
            print()
            print('Discriminator Summary')
            print('---------------------')
            self.discriminator.summary()
        except:
            print('Error : Model can not be summarized. Please ensure that model is built.')
    

    def detailed_losses(self, dataset):
        loss = []

        for target, input_image in dataset:
            gen_output = self.generator(input_image, training=True)        
            disc_real_output = self.discriminator(input_image, target, training=True)
            disc_generated_output = self.discriminator(input_image, gen_output, training=True)

            gen_gan_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_generated_output),
                                                           logits = disc_generated_output)
            gen_l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
            total_gen_loss = gen_gan_loss + LAMBDA * gen_l1_loss

            disc_real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_real_output), 
                                                             logits = disc_real_output)
            disc_generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.zeros_like(disc_generated_output), 
                                                                  logits = disc_generated_output)
            total_disc_loss = disc_real_loss + disc_generated_loss

            loss.append((gen_gan_loss, gen_l1_loss, total_gen_loss, disc_real_loss, disc_generated_loss, total_disc_loss))
        
        loss = np.array(loss)
        avg_loss = np.mean(loss, axis=0)
        return loss, avg_loss

    
    def predict(self, input_image):
        # Input 3 x H x W (values b/w 0 and 255)
        # Rescale and normalize input
        input_image = tf.cast(input_image, tf.float32)
        input_image = tf.image.resize_images(input_image, size=[IMG_HEIGHT, IMG_WIDTH],
                                             align_corners=True,
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        input_image = (input_image / 127.5) - 1
        input_image = tf.expand_dims(input_image, 0)
        
        # Generate output image (values b/w -1 and 1)
        gen_output = self.generator(input_image, training=True)
        return gen_output[0]