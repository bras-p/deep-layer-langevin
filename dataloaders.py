import tensorflow as tf
import tensorflow_datasets as tfds
from abc import ABC, abstractmethod

class DatasetLoader(ABC):
    @abstractmethod
    def loadData(self):
        pass


class ImageLoader(DatasetLoader):
    def __init__(
        self,
        dataset_name = 'cifar10',
        batch_size = 512,
        rescale = lambda x: x/255.,
        augment = lambda x: x,
    ):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.rescale = rescale
        self.augment = augment

    def loadData(self):
        ds = (
            tfds.load(self.dataset_name, split='train', shuffle_files=True, as_supervised=True)
            .shuffle(1024)
            .map( lambda x, y: (self.augment(self.rescale(x)), y), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
            # .map( lambda x, y: (tf.cast(x,tf.float32)/255., y), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        ds_test = (
            tfds.load(self.dataset_name, split='test', shuffle_files=True, as_supervised=True)
            .shuffle(1024)
            .map( lambda x, y: (self.rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
            # .map( lambda x, y: (tf.cast(x,tf.float32)/255., y), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        return ds, ds_test
