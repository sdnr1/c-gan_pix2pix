import tensorflow as tf
import os
import numpy as np

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load_image(image_file, is_train):
    image = tf.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    
    w = tf.shape(image)[1]
    
    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    if is_train:
        # random jittering
    
        # resizing to 286 x 286 x 3
        input_image = tf.image.resize_images(input_image, [286, 286], 
                                             align_corners=True,
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize_images(real_image, [286, 286],
                                            align_corners=True,
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
        # randomly cropping to 256 x 256 x 3
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
        input_image, real_image = cropped_image[0], cropped_image[1]

        if np.random.random() > 0.5:
            # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)
    else:
        input_image = tf.image.resize_images(input_image, size=[IMG_HEIGHT, IMG_WIDTH],
                                             align_corners=True,
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize_images(real_image, size=[IMG_HEIGHT, IMG_WIDTH],
                                            align_corners=True,
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
    # normalizing the images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def load_dataset():
    tf.set_random_seed(1234)
    np.random.seed(4321)

    path_to_zip = tf.keras.utils.get_file('maps.tar.gz',
                                        cache_subdir=os.path.abspath('.'),
                                        origin='https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz', 
                                        extract=True)

    PATH = os.path.join(os.path.dirname(path_to_zip), 'maps/')

    train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.map(lambda x: load_image(x, True))
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(PATH+'val/*.jpg')
    test_dataset = test_dataset.map(lambda x: load_image(x, False))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset