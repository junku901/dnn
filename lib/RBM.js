var math = require('./utils').math;
RBM = module.exports = function (settings) {
    var self = this;

    self.nVisible = settings['n_visible'];
    self.nHidden = settings['n_hidden'];
    self.settings = {
        'log level' : 1 // 0 : nothing, 1 : info, 2: warn
    };

    if(typeof settings['W'] === 'undefined') {
        var a = 1. / self.nVisible;
        settings['W'] = math.randMat(self.nVisible,self.nHidden,-a,a);
    }

    if(typeof settings['hbias'] === 'undefined')
        settings['hbias'] = math.zeroVec(self.nHidden);

    if(typeof settings['vbias'] === 'undefined')
        settings['vbias'] = math.zeroVec(self.nVisible);

    self.input = settings['input'];
    self.W = settings['W'];
    self.hbias = settings['hbias'];
    self.vbias = settings['vbias'];
}

RBM.prototype.train = function(settings) {
    var self = this;
    var lr=0.8, k= 1, epochs = 1500; // default
    if(typeof settings['input'] !== 'undefined')
        self.input = settings['input'];
    if(typeof settings['lr'] !== 'undefined')
        lr = settings['lr'];
    if(typeof settings['k'] !== 'undefined')
        k = settings['k'];
    if(typeof settings['epochs'] !== 'undefined')
        epochs = settings['epochs'];

    var i,j;
    var currentProgress = 1;
    for(i=0;i<epochs;i++) {
        /* CD - k . Contrastive Divergence */
        var ph = self.sampleHgivenV(self.input);
        var phMean = ph[0], phSample = ph[1];
        var chainStart = phSample;
        var nvMeans, nvSamples, nhMeans, nhSamples;

        for(j=0 ; j<k ; j++) {
            if (j==0) {
                var gibbsVH = self.gibbsHVH(chainStart);
                nvMeans = gibbsVH[0], nvSamples = gibbsVH[1], nhMeans = gibbsVH[2], nhSamples = gibbsVH[3];
            } else {
                var gibbsVH = self.gibbsHVH(nhSamples);
                nvMeans = gibbsVH[0], nvSamples = gibbsVH[1], nhMeans = gibbsVH[2], nhSamples = gibbsVH[3];
            }
        }

        var deltaW = math.mulMatScalar(math.minusMat(math.mulMat(math.transpose(self.input),phMean), math.mulMat(math.transpose(nvSamples),nhMeans)),1. / self.input.length);
        var deltaVbias = math.meanMatAxis(math.minusMat(self.input,nvSamples),0);
        var deltaHbias = math.meanMatAxis(math.minusMat(phSample,nhMeans),0);

        self.W = math.addMat(self.W, math.mulMatScalar(deltaW,lr));
        self.vbias = math.addVec(self.vbias, math.mulVecScalar(deltaVbias,lr));
        self.hbias = math.addVec(self.hbias, math.mulVecScalar(deltaHbias,lr));
        if(self.settings['log level'] > 0) {
            var progress = (1.*i/epochs)*100;
            if(progress > currentProgress) {
                console.log("RBM",progress.toFixed(0),"% Completed.");
                currentProgress+=8;
            }
        }
    }
    if(self.settings['log level'] > 0)
        console.log("RBM Final Cross Entropy : ",self.getReconstructionCrossEntropy())
};

RBM.prototype.propup = function(v) {
    var self = this;
    var preSigmoidActivation = math.addMatVec(math.mulMat(v,self.W),self.hbias);
    return math.activateMat(preSigmoidActivation, m.sigmoid);
};

RBM.prototype.propdown = function(h) {
    var self = this;
    var preSigmoidActivation = math.addMatVec(math.mulMat(h,math.transpose(self.W)),self.vbias);
    return math.activateMat(preSigmoidActivation, m.sigmoid);
};

RBM.prototype.sampleHgivenV = function(v0_sample) {
    var self = this;
    var h1_mean = self.propup(v0_sample);
    var h1_sample = math.probToBinaryMat(h1_mean);
    return [h1_mean,h1_sample];
};

RBM.prototype.sampleVgivenH = function(h0_sample) {
    var self = this;
    var v1_mean = self.propdown(h0_sample);
    var v1_sample = math.probToBinaryMat(v1_mean);
    return [v1_mean,v1_sample];
};

RBM.prototype.gibbsHVH = function(h0_sample) {
    var self = this;
    var v1 = self.sampleVgivenH(h0_sample);
    var h1 = self.sampleHgivenV(v1[1]);
    return [v1[0],v1[1],h1[0],h1[1]];
};

RBM.prototype.reconstruct = function(v) {
    var self = this;
    var h = math.activateMat(math.addMatVec(math.mulMat(v,self.W),self.hbias), math.sigmoid);
    var reconstructedV = math.activateMat(math.addMatVec(math.mulMat(h,math.transpose(self.W)),self.vbias), math.sigmoid);
    return reconstructedV;
};

RBM.prototype.getReconstructionCrossEntropy = function() {
    var self = this;
    var reconstructedV = self.reconstruct(self.input);
    var a = math.activateTwoMat(self.input,reconstructedV,function(x,y){
        return x*Math.log(y);
    });

    var b = math.activateTwoMat(self.input,reconstructedV,function(x,y){
        return (1-x)*Math.log(1-y);
    });

    var crossEntropy = -math.meanVec(math.sumMatAxis(math.addMat(a,b),1));
    return crossEntropy

};
RBM.prototype.set = function(property,value) {
    var self = this;
    self.settings[property] = value;
}