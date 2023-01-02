import tensorflow as tf

class LogSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate=1e-4, decay_rate=1e4):
    self.initial_learning_rate = initial_learning_rate
    self.decay_rate = decay_rate

  def __call__(self, step):
     return self.initial_learning_rate / tf.sqrt( tf.math.log( self.decay_rate*step +tf.exp(1.) ) )


