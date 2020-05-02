# Import useful libraries
#from model_layer import Model
from model_layer_1_x import Model_1_x
from tensorflow.python.client import device_lib

# Global Variables:
log_period_samples = 20000
batch_size = 100

# Use a tuple (num_epochs, learning_rate) as keys, and a tuple (training_accuracy, testing_accuracy)
setting = [(5, 0.0001), (5, 0.005), (5, 0.1)]

print('Training Model 1')
#Create an instance of 2.0 Model
'''
obj = Model(setting)
obj.ModelLayer_1()
'''

#Create an instance of 1_x Model
obj_1_x = Model_1_x(setting, log_period_samples, batch_size)
print('Training Model 1')
obj_1_x.ModelLayer_0()
print('Training Model 2')
obj_1_x.ModelLayer_1()
print('Training Model 3')
obj_1_x.ModelLayer_3()
print('Training Model 4')
obj_1_x.convModel()
print('Training Model 4.2')
obj_1_x.convModelAdv()
print('plot_learning_curves')
obj_1_x.evaluation_metrics()

#device_lib.list_local_devices()