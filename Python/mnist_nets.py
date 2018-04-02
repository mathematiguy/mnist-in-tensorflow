def mnist_convnet(train_images, train_labels, learning_rate=0.001, drop_prob=0.8):
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.normalization import local_response_normalization
    from tflearn.layers.estimator import regression

    # Building convolutional network
    network = input_data(shape = [None, 28, 28, 1], name = 'input')
    network = conv_2d(network, 32, 3, activation = 'relu', regularizer = 'L2', name = 'conv2D')
    network = max_pool_2d(network, 2, name = 'max_pool_2d')
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation = 'relu', regularizer = 'L2', name = 'conv2D')
    network = max_pool_2d(network, 2, name = 'max_pool_2d')
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation = 'tanh', name = 'tabh')
    network = dropout(network, drop_prob, name = 'dropout')
    network = fully_connected(network, 256, activation = 'tanh', name = 'tabh')
    network = dropout(network, drop_prob, name = 'dropout')
    network = fully_connected(network, 10, activation = 'softmax', name = 'softmax')
    network = regression(network, optimizer = 'adam', learning_rate = learning_rate,
                  loss = 'categorical_crossentropy', name = 'target')
    return network