from Module import *
#OnlineHD implementation
class OnlineHDv1(TrainProcess):
    def __init__(self,epochs,batchsize,learningrate):
        self.algorithm=TrainRepeater(ClassicVersion1(batchsize,learningrate),epochs)
    def apply(self,model,train,label,xtr):
        self.algorithm.apply(model,train,label,xtr)
#Same but with v2 mult b4 sum
class OnlineHDv2(TrainProcess):
    def __init__(self,epochs,batchsize,learningrate):
        self.algorithm=TrainRepeater(ClassicVersion2(batchsize,learningrate),epochs)
    def apply(self,model,train,label,xtr):
        self.algorithm.apply(model,train,label,xtr)
#NeuralHD with v1
class NeuralHDv1(TrainProcess):
    def __init__(self,epochs,regenloops,batchsize,learningrate, dropoutrate):
        if dropoutrate==0:
            trainblock=TrainBlock(
            [Normalize(),OnlineHDv1(epochs,batchsize,learningrate)]
            )
        else:
            trainblock=TrainBlock(
            [ResetLowVarianceDims(.1), Normalize(),OnlineHDv1(epochs,batchsize,learningrate)]
            )
        self.algorithm=TrainRepeater(trainblock,regenloops) # repeat regenloops times
    def apply(self,model,train,label,xtr):
        self.algorithm.apply(model,train,label,xtr)
#NeuralHD with v2
class NeuralHDv2(TrainProcess):
    def __init__(self,epochs,regenloops,batchsize,learningrate, dropoutrate):
        if dropoutrate==0:
            trainblock=TrainBlock(
            [Normalize(),OnlineHDv2(epochs,batchsize,learningrate)]
            )
        else:
            trainblock=TrainBlock(
            [ResetLowVarianceDims(.1), Normalize(),OnlineHDv2(epochs,batchsize,learningrate)]
            )
        self.algorithm=TrainRepeater(trainblock,regenloops) # repeat regenloops times

    def apply(self,model,train,label,xtr):
        self.algorithm.apply(model,train,label,xtr)
#NeuralHD of batch size batchsize followed by a single pass of batchsize 1 with v2
class SingleStopv2(TrainProcess):
    def __init__(self, epochs,epochs2,batchsize,learningrate, learningrate2):
        self.algorithm=TrainBlock(
            [OnlineHDv2(epochs,batchsize,learningrate),Normalize(),OnlineHDv2(epochs2,1,learningrate2)]
            )
    def apply(self,model,train,label,xtr):
        self.algorithm.apply(model,train,label,xtr)
class NeuralHDv2(TrainProcess):
    def __init__(self,epochs,regenloops,batchsize,learningrate, dropoutrate):
        if dropoutrate==0:
            trainblock=TrainBlock(
            [Normalize(),OnlineHDv2(epochs,batchsize,learningrate)]
            )
        else:
            trainblock=TrainBlock(
            [ResetLowVarianceDims(dropoutrate), Normalize(),OnlineHDv2(epochs,batchsize,learningrate)]
            )
        self.algorithm=TrainRepeater(trainblock,regenloops) # repeat regenloops times

    def apply(self,model,train,label,xtr):
        self.algorithm.apply(model,train,label,xtr)