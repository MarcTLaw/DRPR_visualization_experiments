#!/usr/bin/python

import os
import torch

import torchfile
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import sklearn.manifold
import numpy

dataset = "stl10"
#dataset = "mnist"
#dataset = "cifar100"

temperature = 5
if dataset == "cifar100":
    temperature = 4


save_to_file = True

numpyy = numpy.load('%s/proba_data_%d.npy' % (dataset, temperature))
yq = torch.from_numpy(numpyy)
yq = yq.float()
y_size = yq.size()

n_class = y_size[0]
n_query = n_class
d = y_size[1]

print("Processing %d samples of dimensionality %d" % (n_class, d))


def softmax_class(x_query, x_proto, weights=None):
    n_example = x_query.size(0)
    n_class = x_proto.size(0)
    d = x_query.size(1)
    assert d == x_proto.size(1)
    # compute query distribution over class
    y = torch.pow(x_proto.unsqueeze(0).expand(n_query, n_class, d) - x_query.unsqueeze(1).expand(n_query, n_class, d), 2).sum(2).squeeze()
    y = torch.exp(-y)
    
    if weights is not None:
        # apply class weights
        y = y * weights.unsqueeze(0).expand_as(y)
    
    y = y / y.sum(1, keepdim=True).expand_as(y)
    
    return y

def kl_div(y_target, y_pred):
    return (-y_target * torch.log(y_pred + 1e-8)).sum()


def frobenius_distance(y_target, y_pred):
    return torch.pow((y_target - y_pred),2).sum()



bool3d = False
output_dim = 2
if bool3d:
    output_dim = 3

class ClusterEmbedding(nn.Module):
    def __init__(self, y_target):
        super(ClusterEmbedding, self).__init__()

        self.n_examples = y_target.size(0)
        self.n_clusters = y_target.size(1)
        
        self.inds = Variable(torch.arange(0, self.n_examples).long())
        self.y_target = Variable(y_target)
        
        self.embedding = nn.Embedding(self.n_examples, output_dim)

    def nb_clusters(self):
        return self.n_clusters


    def forward(self):
        return self.embedding.forward(self.inds)
    
    def mu(self):
        z = self.forward()
        return torch.mm(torch.mm(torch.diag(1.0 / self.y_target.sum(0).squeeze()), self.y_target.t()), z)
    
    def pi(self):
        return self.y_target.sum(0).squeeze() / self.y_target.size(0)
    
    def loss(self):
        z = self.forward()
        y_hat = softmax_class(z, self.mu(), weights=self.pi())
        return kl_div(self.y_target,y_hat)



    def frobenius(self):
        z = self.forward()
        y_hat = softmax_class(z, self.mu(), weights=self.pi())
        return frobenius_distance(self.y_target, y_hat)

    def soft_matrix(self):
        z = self.forward()
        return softmax_class(z, self.mu(), weights=self.pi())
        
        

embedding = ClusterEmbedding(yq)

optimizer = optim.RMSprop(embedding.parameters(), lr=0.001, alpha=0.99, eps=1e-06, weight_decay=0, momentum=0, centered=False)

n_step = 8000
for iteration in range(n_step):
    iteration = iteration + 1
    optimizer.zero_grad()
    loss = embedding.loss()
    frob = embedding.frobenius().data[0]
    print("iter {:d}/{:d}, loss = {:0.6f}, frob = {:0.6f}".format(iteration, n_step, loss.data[0], frob))
    loss.backward()
    optimizer.step()

save_directory = "%s_learned_representations" % (dataset)
save_file = True
file_name = "%s/data_%d_%d_d.txt"% (save_directory,temperature,output_dim)
if save_file:
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    u = embedding.forward()
    y = embedding.soft_matrix()
    fdata = open(file_name,"w")
    fclustering = open("%s/softmatrix_%d_d.txt"% (save_directory,output_dim),"w")
    for i in range(n_class):
        for j in range(output_dim):
            fdata.write("%f " % u[i][j].data.numpy().astype(float)[0])
        for j in range(d):
            fclustering.write("%f " % y[i][j].data.numpy().astype(float)[0])
        fdata.write("\n")
        fclustering.write("\n")

    fdata.close()
    fclustering.close()
    print("%s written" % file_name)

