# Image Translation using Conditional Adversarial Network

Conditional GAN for [pix2pix Aerial-to-Map images dataset](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/).

This is an implementation of Condition GAN based on a paper by Isola et al. Link : [https://arxiv.org/abs/1611.07004](https://arxiv.org/abs/1611.07004)

Implemantated using Tensorflow framework for python. This implementation has a U-Net generator as proposed by Isola et al. and also a ResNet-9 generator, of which, any one can be chosen. Optionally, noise can also added to input of generator.

### Results

The performance of each model was evaluated by compluting Mean Squared Error (MSE) between the predicted and expected output on test data.

| Model               | MSE     |
|---------------------|---------|
| U-Net               | 0.01834 |
| U-Net with noise    | 0.01852 |
| ResNet-9            | 0.01443 |
| ResNet-9 with noise | 0.01476 |

The study showed that ResNet-9 generator performed about 21.3% better as compared to U-Net generator. Although, addition of noise to generator input did not improve results.

See [Project Report](https://github.com/sdnr1/c-gan_pix2pix/blob/master/project_report.pdf) for details on results, model architecture, choice of hyperparameters, data preprocessing, etc.

# Usage

### Steps to use code

1. Import required functions.

```
from initializeDataset import load_dataset
from c_gan import C_GAN, UNetGenerator, ResNet9Generator, PatchDiscriminator
```

2. Create object of `C_GAN` class.

```
# for U-Net generator
gan = C_GAN(generator=UNetGenerator(), discriminator=PatchDiscriminator())

# for ResNet-9 generator
gan = C_GAN(generator=ResNet9Generator(), discriminator=PatchDiscriminator())

# for U-Net generator with noise added to inputs
gan = C_GAN(generator=UNetGenerator(noise=True), discriminator=PatchDiscriminator())

# for ResNet-9 generator with noise added to inputs
gan = C_GAN(generator=UNetGenerator(noise=True), discriminator=PatchDiscriminator())

# Default training rate is 2e-4 for both generator and discriminator.
# Use 'generator_learning_rate' and 'discriminator_learning_rate' parameters to specify different learning rates.
# See code for more details.
```

3. Train model.

```
EPOCHS = 200
gan.train(train_dataset, EPOCHS)
```

4. Evaluate model (Calculate Mean Sqaured Error).

```
mse = gan.evaluate(test_dataset)
```

5. Restoring model from checkpoint

```
gan.restore_from_checkpoint()
```

6. Generate samples outputs

```
NUM_SAMPLES = 10
for tar, img in test_dataset.take(NUM_SAMPLES):
    gan.generate_image(img, tar)
```

7. Predict map image from a given aerial image

```
# Input image can be of any size (square images are prefered)
# Outputs is a 3 x 256 x 256 RGB image of prediction (range of values [-1, 1])
gan.predict(input_image) 
```
