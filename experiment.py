import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import mkdir, listdir

from optimizers.base import set_langevin



class Experiment():

    def __init__(
        self,
        model_builder,
        dataloader,
        EPOCHS = 10,
        optimizers = ['adam'],
        base = '',
    ):
        self.model_builder = model_builder
        self.dataloader = dataloader

        self.EPOCHS = EPOCHS
        self.optimizers = optimizers
        self.base = base
        self.models = {}


    def load_data(self):
        self.ds, self.ds_test = self.dataloader.loadData()
    
    
    def getModel(self, optimizer):
        model = self.model_builder.getModel()
        model.compile(optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        return model
    

    def run_experiment(self):
        model = self.getModel('adam')
        self.eval_train_init = model.evaluate(self.ds)
        self.eval_test_init = model.evaluate(self.ds_test)
        model.save_weights(self.base + '/checkpoints/initial_checkpoint')

        self.models = {}
        for optimizer in self.optimizers:
            self.models[optimizer] = self.getModel(optimizer)
            self.models[optimizer].evaluate(self.ds_test)
            self.models[optimizer].load_weights(self.base + '/checkpoints/initial_checkpoint')
            set_langevin(self.models[optimizer])
            self.models[optimizer].fit(self.ds, epochs=self.EPOCHS, validation_data=self.ds_test)


    def plot(self):
        fig, (ax1, ax2) = plt.subplots(2)
        for optimizer in self.optimizers:
            ax1.plot([self.eval_train_init[0]] + self.models[optimizer].history.history['loss'], label='sigma='+print_schedule(optimizer.sigma))
            ax2.plot([self.eval_test_init[1]] + self.models[optimizer].history.history['val_accuracy'], label='sigma='+print_schedule(optimizer.sigma))
        plt.legend()
        plt.show()


    def save_data(self, dir_name='experiment'):
        mkdir(self.base + dir_name)
        for k in range(len(self.optimizers)):
            df = pd.DataFrame({'time':np.arange(self.EPOCHS+1), 'f':[self.eval_test_init[1]]+self.models[self.optimizers[k]].history.history['val_accuracy']})
            df.to_csv(self.base + dir_name + '/' + str(k) + '_v_acc' + '.csv')
        for k in range(len(self.optimizers)):
            df = pd.DataFrame({'time':np.arange(self.EPOCHS+1), 'f':[self.eval_train_init[0]]+self.models[self.optimizers[k]].history.history['loss']})
            df.to_csv(self.base + dir_name + '/' + str(k) + '_loss' + '.csv')

        f = open(self.base + dir_name + '/' + "experiment_settings.txt", "w+")
        f.write('Dataset: ' + self.dataloader.dataset_name +'\n')
        f.write('batch size: ' + str(self.dataloader.batch_size)+'\n')
        f.write('Model: ' + self.model_builder.__class__.__name__ +'\n')
        f.write('Model options: ' + str(self.model_builder.__dict__)+'\n')

        f.write('Learning rate: ' + print_schedule(self.optimizers[0].learning_rate)+'\n'+'\n')

        for k in range(len(self.optimizers)):
            f.write('Optimizer ' + str(k) + ' : ' + self.optimizers[k]._name + '  ')
            if hasattr(self.optimizers[k], 'sigma'):
                f.write('Sigma schedule: ' + print_schedule(self.optimizers[k].sigma) + '\n')
        f.close()


def print_schedule(lr_schedule):
    if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
        return str(lr_schedule.__dict__)
    else:
        return str(lr_schedule)


def plot_data(base, dir_name):
    fs = listdir(base+dir_name)
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set(ylabel='Train loss')
    ax2.set(ylabel='Test accuracy')
    for f in fs:
        if f[-8:-4] == 'loss':
            df = pd.read_csv(base+dir_name +'/'+ f)
            ax1.plot(df['f'])
        if f[-9:-4] =='v_acc':
            df = pd.read_csv(base+dir_name +'/'+ f)
            ax2.plot(df['f'])
    plt.title(dir_name)
    plt.legend()
    plt.show()




