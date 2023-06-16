import math
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.metrics import fowlkes_mallows_score
    #Cosine Similarity
def cos_cdist(x1 : torch.Tensor, x2 : torch.Tensor, eps : float = 1e-8):
    eps = torch.tensor(eps, device=x1.device)
    norms1 = x1.norm(dim=1).unsqueeze_(1).max(eps)
    norms2 = x2.norm(dim=1).unsqueeze_(0).max(eps)
    cdist = x1 @ x2.T
    cdist.div_(norms1).div_(norms2)
    return cdist
    #Normal Kernel generation
def gaussian_kernel(mu,sigma, dim,nFeatures):
    return torch.normal(mu,sigma,size=(dim,nFeatures))
    #Stitched kernel generation using multiple models
    #dim=target dim
    #n=number of dimensions taken from each model
    #modeldim=dim of models being trained to get stitched dimensions
def stitched_kernel(algorithm, xtr, label,dim,modeldim,n):
    newbasis=torch.zeros(dim,xtr.size(1))

    i=0
    while i<dim:
        model=HDCModel(torch.unique(label).size(0),xtr.size(1),modeldim)
        train_start(model,algorithm,xtr,label)
        newbasis[i:i+n,:]=model.basis[keep_n_best_var(model.classes,n)]

        i+=n
    return newbasis
    #neuralHD but keep best n instead of drop out worst n
def keep_n_best_var(classes,n):
    var = torch.var(classes, 0) 
    order = torch.argsort(-var)
    return order[:n]


class HDCModel:
    def __init__(self, classes : int, features : int, dim : int = 400,mu=0,sigma=1, kernel=None, classh=None):
        #Configure for hdb, hdc, and hde classes
        self.mu=mu
        self.sigma=sigma
        self.nClasses = classes
        self.nFeatures= features
        #hypervector size
        self.dimensionality=dim
        # self.learningrate=lr
        self.base = torch.empty(self.dimensionality).uniform_(0.0, 2*math.pi)
        # Initialize basis in gaussian distribution
        if kernel is not None:
            self.basis = kernel 
        else: 
            self.basis=torch.normal(self.mu,self.sigma,size=(self.dimensionality,self.nFeatures))
        # Initialize classification hypervectors
        if classh is not None:
            self.classes=torch.normal(0,1,(self.nClasses, self.dimensionality))
        else:
            self.classes = torch.zeros((self.nClasses, self.dimensionality))
    def encode(self,x):
        n = x.size(0)
        bsize = min([x.size(1),1024])
        h = torch.empty(n, self.basis.shape[0], device=x.device, dtype=x.dtype)
        temp = torch.empty(bsize, self.basis.shape[0], device=x.device, dtype=x.dtype)

        # we need batches to remove memory usage
        for i in range(0, n, bsize):
            torch.matmul(x[i:i+bsize], self.basis.T, out=temp)

            # self.noise ... I haven't seen any indication that it works better 
            # if self.noise:
            torch.add(temp, self.base, out=h[i:i+bsize])#h[i:i+bsize]=temp# torch.add(temp, self.base, out=h[i:i+bsize])
            # else:
            # h[i:i+bsize]=temp
            h[i:i+bsize].cos_().mul_(temp.sin_())
        return h
    def __call__(self, x : torch.Tensor):
        #return predicted values
        return self.predict(x)
    def predict(self,x):
        #return predictions based on similarity of encoded inputs to classification hypervectors
        return  cos_cdist(self.encode(x), self.classes).argmax(1)
    def test(self,x_encoded, y_labels):
        yhat= cos_cdist(x_encoded, self.classes).argmax(1)
        return (yhat==y_labels).float().mean()
        
class TrainProcess(ABC):
    @abstractmethod
    def apply(self, model, train, label,xtr):
        #do something to model.classes
        pass
class FebHDVersion2(TrainProcess):
    def __init__(self,batchsize : int, learningrate):
        self.batch_size=batchsize
        self.learningrate=learningrate
    def apply(self,model,train,label,xtr):
        tempclasses=torch.zeros(model.classes.shape)
        norm=cos_cdist(train,model.classes).mean(0).unsqueeze(0)
        for i in range(0,label.size(0),self.batch_size):
            sample = train[i:i+self.batch_size] 
            vals = cos_cdist(sample, model.classes)#/norm
            guesses = vals.argmax(1)
            for j in range(0,guesses.size(0)):
                # model.classes[guesses[j]]+=self.learningrate*train[i+j]
                tempclasses[guesses[j]]+=self.learningrate*train[i+j]
        model.classes[:]=tempclasses
class FebHDVersion2Inv(TrainProcess):
    def __init__(self,batchsize : int, learningrate):
        self.batch_size=batchsize
        self.learningrate=learningrate
    def apply(self,model,train,label,xtr):
        tempclasses=torch.zeros(model.classes.shape)
        norm=cos_cdist(train,model.classes).mean(0).unsqueeze(0)
        for i in range(0,label.size(0),self.batch_size):
            sample = train[i:i+self.batch_size] 
            vals = cos_cdist(sample, model.classes)#/norm
            guesses = vals.argmin(1)
            for j in range(0,guesses.size(0)):
                # model.classes[guesses[j]]+=self.learningrate*train[i+j]
                tempclasses[guesses[j]]+=self.learningrate*train[i+j]
#                 self.learningrate*train[i+j]
        model.classes[:]=tempclasses
        # model.classes=torch.nn.functional.normalize(model.classes)
class FebHDVersion2Dense(TrainProcess):
    def __init__(self,batchsize : int, learningrate):
        self.batch_size=batchsize
        self.learningrate=learningrate
    def apply(self,model,train,label,xtr):
        tempclasses=torch.zeros(model.classes.shape)
        norm=cos_cdist(train,model.classes).mean(0).unsqueeze(0)
        for i in range(0,label.size(0),self.batch_size):
            sample = train[i:i+self.batch_size] 
            vals = cos_cdist(sample, model.classes)#/norm
            guesses = vals.argmax(1)
            for j in range(0,guesses.size(0)):
                # model.classes[guesses[j]]+=self.learningrate*train[i+j]
                tempclasses[guesses[j]]+=2*self.learningrate*train[i+j]*(1-vals[0,guesses[j]])
                # print((1-vals[0,:]).shape)
                # print(train[i+j].shape)
                tempclasses-=(self.learningrate*train[i+j].unsqueeze(1)@(1-vals[0,:]).unsqueeze(0)).T
#                 model.classes[guesses[j]]-=self.learningrate*train[i+j]*(1-vals[0,guesses[j]])
#                 model.classes[answers[j]]+=self.learningrate*train[i+j]*(1-vals[0,answers[j]])
        model.classes[:]=tempclasses
        # model.classes=torch.nn.functional.normalize(model.classes)
class FebHDRepeater(TrainProcess):
    def __init__(self, component, maxreps):
        self.component=component
        self.maxreps=maxreps
    def apply(self, model, train, label,xtr):
        i=0
        while i<self.maxreps and model.test(train, label)<1:
            # print(model.test(train, label))
            label=cos_cdist(train, model.classes).argmax(1)
            self.component.apply(model,train,label,xtr)
            i+=1

class ClassicVersion0(TrainProcess):
    def __init__(self,batchsize : int, learningrate, a, b):
        self.batch_size=batchsize
        self.learningrate=learningrate
        self.a=a
        self.b=b
    #Old OnlineHD implementation- usually a bit worse due to floating point error
    def apply(self,model, train, label,xtr):
        n = train.size(0)
        batch_size = min([label.size(0), self.batch_size])#64
        for i in range(0, n, batch_size):
            h_ = train[i:i+batch_size]
            y_ = label[i:i+batch_size]
            scores = cos_cdist(h_, model.classes)#cos
            y_pred = scores.argmax(1)
            wrong = y_ != y_pred
            # computes alphas to update model
            # alpha1 = 1 - delta[lbl] -- the true label coefs
            # alpha2 = delta[max] - 1 -- the prediction coefs
            aranged = torch.arange(h_.size(0), device=h_.device)
            alpha1 = (1.0 - scores[aranged,y_]).unsqueeze_(1)
            alpha2 = (self.a*scores[aranged,y_pred] - self.b*1.0).unsqueeze_(1)

            for lbl in y_.unique():
                m1 = wrong & (y_ == lbl) # mask of missed true lbl
                m2 = wrong & (y_pred == lbl) # mask of wrong preds
                model.classes[lbl] += self.learningrate*(alpha1[m1]*h_[m1]).sum(0)
                model.classes[lbl] += self.learningrate*(alpha2[m2]*h_[m2]).sum(0)
        return
    #ClassicVersion1: sum before mult
class ClassicVersion1(TrainProcess):
    def __init__(self,batchsize : int, learningrate):
        self.batch_size=batchsize
        self.learningrate=learningrate
    #Old OnlineHD implementation- usually a bit worse due to floating point error
    def apply(self,model, train, label,xtr):
        n = train.size(0)
        batch_size = min([label.size(0), self.batch_size])#64
        for i in range(0, n, batch_size):
            h_ = train[i:i+batch_size]
            y_ = label[i:i+batch_size]
            scores = cos_cdist(h_, model.classes)#cos
            y_pred = scores.argmax(1)
            wrong = y_ != y_pred
            # computes alphas to update model
            # alpha1 = 1 - delta[lbl] -- the true label coefs
            # alpha2 = delta[max] - 1 -- the prediction coefs
            aranged = torch.arange(h_.size(0), device=h_.device)
            alpha1 = (1.0 - scores[aranged,y_]).unsqueeze_(1)
            alpha2 = (scores[aranged,y_pred] - 1.0).unsqueeze_(1)

            for lbl in y_.unique():
                m1 = wrong & (y_ == lbl) # mask of missed true lbl
                m2 = wrong & (y_pred == lbl) # mask of wrong preds
                model.classes[lbl] += self.learningrate*(alpha1[m1]*h_[m1]).sum(0)
                model.classes[lbl] += self.learningrate*(alpha2[m2]*h_[m2]).sum(0)
        return
#ClassicVersion2: mult before sum
class ClassicVersion2(TrainProcess):
    def __init__(self,batchsize : int, learningrate):
        self.batch_size=batchsize
        self.learningrate=learningrate
    def apply(self,model, train, label,xtr):
        assert model.dimensionality == train.size(1)

        #I'm not certain if randomizing does anything
        r = torch.randperm(label.size(0))
        label=label[r]
        train=train[r,:]
        # correct = 0
        # count = 0
        for i in range(0,label.size(0),self.batch_size):
            sample = train[i:i+self.batch_size] 
            answers = label[i:i+self.batch_size]
            vals = cos_cdist(sample, model.classes)
            guesses = vals.argmax(1)
            for j in range(0,answers.size(0)):
                if guesses[j] != answers[j]:
                    model.classes[guesses[j]]-=self.learningrate*train[i+j]*(1-vals[0,guesses[j]])
                    model.classes[answers[j]]+=self.learningrate*train[i+j]*(1-vals[0,answers[j]])
        return
#Normalize classes
class Normalize(TrainProcess):
    def apply(self, model, train, label,xtr):
        model.classes=torch.nn.functional.normalize(model.classes)
        return
#drop dimensions - NeuralHD style
class ResetLowVarianceDims(TrainProcess):
    def __init__(self, percentDrop):
        self.percentDrop=percentDrop
    def apply(self,model, train, label,xtr):
        normed_classes = torch.nn.functional.normalize(model.classes)
        var = torch.var(normed_classes, 0) 
        # assert len(var) == self.dimensionality
        # rank each entry in variances from smallest to largest
        order = torch.argsort(var)
        #drop amountDrop bases
        toDrop = order[:int(self.percentDrop*model.dimensionality)]
        for i in toDrop:
            #generate a new ith vector in the basis
            model.basis[i] = torch.normal(model.mu,model.sigma, size=(model.nFeatures,)) ##I need to make this happen as a function of the model
        #Update Classes
        #            --------------
        #This code was left out. Maybe useful?
        for i in toDrop:
            model.classes[:,i] = torch.zeros(model.nClasses)
        train[:]=model.encode(xtr)
        return

class ResetLowVarianceDimsForEachClass(TrainProcess):
    def __init__(self, percentDrop):
        self.percentDrop=percentDrop
    def apply(self,model, train, label,xtr):
        normed_classes = torch.nn.functional.normalize(model.classes)
        varbyclass = abs(normed_classes.mean(0)-normed_classes)[:6]
        # assert len(var) == self.dimensionality
        # rank each entry in variances from smallest to largest

        #stopped
        order = torch.argsort(var)
        #drop amountDrop bases
        toDrop = order[:int(self.percentDrop*model.dimensionality)]
        for i in toDrop:
            #generate a new ith vector in the basis
            model.basis[i] = torch.normal(model.mu,model.sigma, size=(model.nFeatures,)) ##I need to make this happen as a function of the model
        #Update Classes
        #            --------------
        #This code was left out. Maybe useful?
        for i in toDrop:
            model.classes[:,i] = torch.zeros(model.nClasses)
        train[:]=model.encode(xtr)
        return
#Change basis slightly
class AddNoise(TrainProcess):
    def __init__(self, noiseratio):
        self.ratio=noiseratio
    def apply(self, model, train,label,xtr):
        model.basis[:]+=self.ratio*torch.normal(0,1,model.basis.shape)
        train[:]=model.encode(xtr)
        # print(model.basis[0][0])
#put sequential training algorithms together in a schedule
class TrainBlock(TrainProcess):
    def __init__(self, components=[]):
        self.components=components
    def append_component(self, component):
        self.components.append(component)
    def apply(self, model, train,label,xtr):
        #do something to model.classes
        for component in self.components:
            component.apply(model,train,label,xtr)
        return
#repeat training algorithms
class TrainRepeater(TrainProcess):
    def __init__(self, component, repetitions):
        self.component=component
        self.repetitions=repetitions
    def apply(self, model, train, label,xtr):
        for _ in range(self.repetitions):
            self.component.apply(model,train,label,xtr)
    def applyBIC(self, model,train,label,xtr):
        maxval=0
        temp=None
        for _ in range(self.repetitions):
            self.component.apply(model,train,label,xtr)
            trainaccuracy=model.test(train,label)
            if trainaccuracy>maxval:
                temp=copy.deepcopy(model.classes)
                maxval=trainaccuracy
        model.classes[:]=temp
#print stuff during training
# class AddVectorFromDissimilar(TrainProcess):
#     def __init__(self,similaritymax):
#         self.goal=similaritymax
#         return
#     def apply(self,model,train,label,xtr):
#         similarities=cos_cdist(train, self.classes)
#         similarities=similarities.max(1).values
#         toadd=train[similarities<goal]
#         if len(toadd)==0:
#             return False
#         model.classes=torch.vstack(model.classes,toadd.sum(0))
#         return True
class DebugPrinter(TrainProcess):
    def __init__(self, whattoprint, function=lambda x:x):
        self.whattoprint=whattoprint
        self.function=function
    def apply(self, model, train, label,xtr):
        print(self.function(locals()[self.whattoprint]))
#apply training with generalized applicability - ytr must be .long
def train_start(model,trainblock,xtr,ytr):
    xencoded=model.encode(xtr)
    trainblock.apply(model,xencoded,ytr,xtr)
#get accuracy
def eval_acc(model,xte,yte):
    yhat=model(xte)
    eval=[yhat[i]==yte[i] for i in range(len(yte))]
    return sum(eval)/len(yte)
#get inferred accuracy for unsupervised learning
def eval_inferred_acc(model,xtrain,ytrain):
    labels=model(xtrain)
    yhat=torch.zeros(ytrain.shape)
    modes=[]
    for i in torch.unique(ytrain):
        n=torch.mode(ytrain[labels==i]).values
        modes.append(n)
    for i in range(ytrain.shape[0]):
        yhat[i]=modes[labels[i]]
    eval=[yhat[i]==ytrain[i] for i in range(len(ytrain))]
    return sum(eval)/len(ytrain)
def torch_train_test_split(x,y,trainratio):
    cutoff=int(y.size(0)*trainratio)
    r=torch.randperm(y.size(0))
    xtrain=x[r][cutoff:]
    xtest=x[r][:cutoff]
    ytrain=y[r][cutoff:]
    ytest=y[r][:cutoff]
    return xtrain,xtest,ytrain,ytest
def eval_fowlkes_mallows(model,X,y):
    labels=model(X)
    return fowlkes_mallows_score(labels,y)
# OnlineHDv2=TrainRepeater(ClassicVersion2(64,.0001),15)

class TargetUpdate(TrainProcess):
    def __init__(self,batchsize : int, learningrate):
        self.batch_size=batchsize
        self.learningrate=learningrate
    def apply(self,model, train, label,xtr):
        assert model.dimensionality == train.size(1)

        #I'm not certain if randomizing does anything
        r = torch.randperm(label.size(0))
        label=label[r]
        train=train[r,:]
        # correct = 0
        # count = 0
        for i in range(0,label.size(0),self.batch_size):
            sample = train[i:i+self.batch_size] 
            answers = label[i:i+self.batch_size]
            vals = cos_cdist(sample, model.classes)
            targets=label
            for j in range(0,answers.size(0)):
                if vals[j,0]/vals[j,1]<label[j]:
                    model.classes[0]-=sample[j]
                    model.classes[1]+=sample[j]
                else:
                    model.classes[0]-=sample[j]
                    model.classes[1]+=sample[j]
        return

class StochasticUpdate1D(TrainProcess):
    def __init__(self,batchsize : int, learningrate):
        self.batch_size=batchsize
        self.learningrate=learningrate
    def apply(self,model, train, label,xtr):
        assert model.dimensionality == train.size(1)

        #I'm not certain if randomizing does anything
        r = torch.randperm(label.size(0))
        label=label[r]
        train=train[r,:]
        # correct = 0
        # count = 0
        
        for i in range(0,label.size(0),self.batch_size):
            
            sample = train[i:i+self.batch_size] 
            answers = label[i:i+self.batch_size]
            # print(model.classes.shape, answers.shape)
            # for j in range(len(model.classes)):
                # print(cos_cdist(sample,model.classes[j:j+1]).shape,answers.shape)
            # print(answers.shape)
            # print(cos_cdist(sample,model.classes[j:j+1]).squeeze().shape)
            deltas=answers-cos_cdist(sample,model.classes).squeeze()
            derivative=-sample
            update=-derivative.T@deltas*self.learningrate
            model.classes+=update
            # print(answers)
            # print(cos_cdist(sample/sample.norm(dim=1),model.classes[j:j+1]))
                #[64,2000]*[2000,1]=[64,1]
            # np.linalg.solve()
        return

class StochasticUpdate2D(TrainProcess):
    def __init__(self,batchsize : int, learningrate, roadtotarget=None):
        self.batch_size=batchsize
        self.learningrate=learningrate
        if roadtotarget:
            self.roadtotarget=roadtotarget#write method next to get targets if they are far off
    def apply(self,model, train, label,xtr):
        assert model.dimensionality == train.size(1)

        #I'm not certain if randomizing does anything
        r = torch.randperm(label.size(0))
        label=label[r]
        train=train[r,:]
        # correct = 0
        # count = 0
        
        for i in range(0,label.size(0),self.batch_size):
            
            sample = train[i:i+self.batch_size] 
            answers = label[i:i+self.batch_size]
            # print(model.classes.shape, answers.shape)
            for j in range(len(model.classes)):
                # print(cos_cdist(sample,model.classes[j:j+1]).shape,answers.shape)
                deltas=answers[:,j]-cos_cdist(sample,model.classes[j:j+1]).squeeze()#
                # print(deltas.shape)
                derivative=-sample
                update=-derivative.T@deltas*self.learningrate
                model.classes[j]+=update

