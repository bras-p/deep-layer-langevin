import numpy as np
import tensorflow as tf

from models.dense import DenseModel
from models.conv_dense import ConvDense
from models.highway import HighwayModel
from models.resnet import ResNet
from models.densenet import DenseNet

from optimizers.ladam import LAdam, LayerLAdam
from optimizers.lrmsprop import LRMSprop, LayerLRMSprop
from optimizers.ladadelta import LAdadelta, LayerLAdadelta
from experiment import Experiment, plot_data
from dataloaders import ImageLoader
from schedules import LogSchedule


dataset_name = 'cifar10'
classes = 10
dataset_len = 50000

models_dict = {
    'dense': DenseModel(nb_units=10*[64], classes=classes),
    'conv_dense': ConvDense(filters= 32, kernel_size=4, nb_conv=2, nb_units=2*[64], classes=classes),
    'highway': HighwayModel(nb_units=10*[64], classes=classes),
    'resnet': ResNet(input_shape=(32, 32, 3), filters=16, block_layers=[5,5], hidden_units=512, classes= classes, zero_padding=(0, 0), mode='resnet'),
    'vggnet': ResNet(input_shape=(32, 32, 3), filters=16, block_layers=[5,5,5], hidden_units=512, classes= classes, zero_padding=(0, 0), mode='vgg'),
    'densenet': DenseNet(input_shape=(32,32,3), classes=classes)
}

model_name = 'resnet'
model_builder = models_dict[model_name]

batch_size = 512
def rescale(x):
    x = tf.cast(x, tf.float32)
    return x/255.

def augment(x):
    x = tf.image.resize_with_crop_or_pad(x, 32 + 4, 32 + 4)
    x = tf.image.random_crop(x,[32,32,3])
    x = tf.image.random_flip_left_right(x)
    return x

dataloader = ImageLoader(
    dataset_name = dataset_name,
    batch_size = batch_size,
    rescale = rescale,
    augment = augment)

EPOCHS = 50
lr_0 = 2e-1
epoch_change = 40
steps_per_epoch = int(np.ceil(dataset_len/batch_size))
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[epoch_change*steps_per_epoch], values=[lr_0, lr_0/10])
sigma_0 = 5e-3
sigma_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[epoch_change*steps_per_epoch], values=[sigma_0, 0.])
# sigma_schedule = LogSchedule(sigma_0, (epoch_change*steps_per_epoch)**(-1))

model = model_builder.getModel()
optimizers = [
    LayerLAdadelta(learning_rate=lr_schedule, sigma=0.),
    # LayerLAdam(learning_rate=lr_schedule, sigma=sigma_schedule),
    # LayerLAdam(learning_rate=lr_schedule, sigma=sigma_schedule, langevin_layers=range(int(0.15*len(model.layers)))),
    LayerLAdadelta(learning_rate=lr_schedule, sigma=sigma_schedule, langevin_layers=range(int(0.3*len(model.layers)))),
    # LayerLAdam(learning_rate=lr_schedule, sigma=sigma_schedule, langevin_layers=range(int(0.9*len(model.layers)))),
    # LayerLAdam(learning_rate=lr_schedule, sigma=sigma_schedule, langevin_layers=range(int(1.*len(model.layers)))),
]

base = 'C:/Users/pierr/Documents/These/These/deep_NN_langevin/new_version/data/'


experiment = Experiment(
    model_builder = model_builder,
    dataloader = dataloader,
    EPOCHS = EPOCHS,
    optimizers = optimizers,
    base = base,
)


experiment.load_data()
experiment.run_experiment()
experiment.plot()

# experiment.save_data(model_name + '/ladadelta_cifar10')



# # plot
# dir_name = 'resnet/ladam_2x5_cifar10'
# plot_data(base, 'resnet/lrmsprop_2x5_cifar100')

# input_shape = ds.element_spec[0].shape[1:]




