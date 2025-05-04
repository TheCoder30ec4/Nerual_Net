from layer import Layer
import numpy as np 


class Activation(Layer):
    
    def __init__(self,activation,prime_activation):
        self.activation = activation
        self.prime_activation  = prime_activation
        
    
    def forward(self, input):
        self.input = input  
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient,self.prime_activation(self.input))
        