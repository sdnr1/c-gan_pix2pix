import tensorflow as tf
import numpy as np
import pickle
from initializeDatabase import load_database
from c_gan import C_GAN

EPOCHS = 200

# LOAD DATASET
train_dataset, test_dataset = load_database()


# U-NET GENERATOR W/O NOISE
unet_gan = C_GAN(generator='unet')
unet_gan.train(train_dataset, EPOCHS)

idx = 0
for tar, img in test_dataset.take(10):
    idx += 1
    unet_gan.generate_image(img, tar, name='unet_img' + str(idx))

mse = unet_gan.evaluate(test_dataset)
with open('mse_unet.pickle', 'wb') as f:
    pickle.dump(mse, f)

print("Mean Squared Error:", mse)


# RESNET-9 GENERATOR W/O NOISE
resnet_gan = C_GAN(generator='unet')
resnet_gan.train(train_dataset, EPOCHS)

idx = 0
for tar, img in test_dataset.take(10):
    idx += 1
    resnet_gan.generate_image(img, tar, name='resnet_img' + str(idx))

mse = resnet_gan.evaluate(test_dataset)
with open('mse_resnet.pickle', 'wb') as f:
    pickle.dump(mse, f)

print("Mean Squared Error:", mse)