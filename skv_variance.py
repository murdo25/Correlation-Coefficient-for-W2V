import math
import numpy as np
import sys
from penseur import penseur

# This will load word2vec using the full corpus
p = penseur.Penseur()


#example scholar commands
#s.return_words(adv+vec,10)
#s.get_vector('dog_NN')


#example files
#input file name  (DIRECT PATH TO FILE)
#fname = 'ownership.txt'
#fname = 'skv_analogies.txt'
#fname = 'skv2_analogies.txt'
fname = 'skv_test1.txt'


#input file
if(len(sys.argv) > 1):
    fname = sys.argv[1]

#grab examples 
with open(fname) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content] 

#Clean
terms = []
for i in content:
    terms.append(i.split('::'))

vectors = []
for i in terms:
    #vectors.append(s.get_vector(i[1]) - s.get_vector(i[0]))
    vectors.append(p.get_vector(i[1])[0] - p.get_vector(i[0])[0])
    
vectors = np.vstack(vectors)

#compute average vector:
#adv = np.mean(vectors,axis=0)/float(len(content))
adv = np.average(vectors,axis=0)



#compute numpy.corrcoeef 
#https://www.youtube.com/watch?v=uzW9WKHzSYM
def variance(x):
    #sum(x^2)/n  - mean(x)^2
    sumz = np.sum([i**2 for i in x])
    mean = np.mean(x)**2
    return sumz/float(len(x)) - mean
#or
#def variance(x):
#    return np.var(x,axis=0)


#covariance
def coVar(x,y):
    sumz = np.sum([x[i]*y[i] for i in range(len(x))])
    mean = np.mean(x)*np.mean(y)
    return sumz/float(len(x)) - mean


#Correlation Coefficient
def corCof(m,x):
    numerator = coVar(m,x)
    denom = math.sqrt(variance(x)) * math.sqrt(variance(m))
    return numerator/denom


#list all the correlation coefficient of each vector with respect to the mean.
for i in range(len(vectors)):
    print(i+1,corCof(adv,vectors[i]),terms[i][0],terms[i][1])



#experiment 
#ice = s.get_vector("ice_NN")-s.get_vector("water_NN")


