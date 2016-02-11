import neural_network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()



#net = neural_network.Network([784, 10])

#assert isinstance(test_data, object)
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

print training_data[0]



