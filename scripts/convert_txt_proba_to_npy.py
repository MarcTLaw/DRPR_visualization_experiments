import numpy
from tempfile import TemporaryFile

dataset = "cifar100"
nb_examples = 10000
nb_classes = 10
temperatures = [1,2,5,10]

if dataset=="stl10":
    nb_examples = 10000

if dataset=="cifar100":
    temperatures = [1,2,4,5,10]
    nb_classes = 100


for temperature in temperatures:
    # converting .txt matrix into numpy matrix
    a = numpy.fromfile('proba_data_%d.txt' % temperature,sep=" ")
    # setting the correct dimensions
    b = numpy.reshape(a,[nb_examples,nb_classes])

    # saving .npy file
    outfile = open('proba_data_%d.npy' % temperature,'wb')
    numpy.save(outfile, b)
    outfile.close()
