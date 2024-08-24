# The idea in this very simple code is to understand teh concept of Hooks since it is being used in this directory
# I start to work really hard to understand how it work and how make it even better.
import torch

a = torch.tensor(2.0,requires_grad=True) # you have acculamtegrade nodes
# for each tensor that accumlategradient as the flwo backword to leaf graphs
# a and b are leaf nodes

b = torch.tensor(3.0,requires_grad=True)

# c greating by operation it is not leaf node
# intermediate node
# it has graident function properties
c = a+b
# when oyu call c.backward to start backpropgagting the gradient
# at backward you have gradient function point to a ndoe in the backward graph
# you pass the gradient to the node specificied in gradient function .
# you start with default gradient value of 1 if you donot specificy in backgrwod
c.backward()