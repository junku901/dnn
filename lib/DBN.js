/**
 * Created by joonkukang on 2014. 1. 13..
 */
var math = require('./utils').math;
HiddenLayer = require('./HiddenLayer');
RBM = require('./RBM');
MLP = require('./MLP');

DBN = module.exports = function (settings) {
    var self = this;
    self.x = settings['input'];
    self.y = settings['label'];
    self.sigmoidLayers = [];
    self.rbmLayers = [];
    self.nLayers = settings['hidden_layer_sizes'].length;
    self.hiddenLayerSizes = settings['hidden_layer_sizes'];
    self.nIns = settings['n_ins'];
    self.nOuts = settings['n_outs'];
    self.settings = {
        'log level' : 1 // 0 : nothing, 1 : info, 2: warn
    };

    // Constructing Deep Neural Network
    var i;
    for(i=0 ; i<self.nLayers ; i++) {
        var inputSize, layerInput;
        if(i == 0)
            inputSize = settings['n_ins'];
        else
            inputSize = settings['hidden_layer_sizes'][i-1];

        if(i == 0)
            layerInput = self.x;
        else
            layerInput = self.sigmoidLayers[self.sigmoidLayers.length-1].sampleHgivenV();

        var sigmoidLayer = new HiddenLayer({
            'input' : layerInput,
            'n_in' : inputSize,
            'n_out' : settings['hidden_layer_sizes'][i],
            'activation' : math.sigmoid
        });
        self.sigmoidLayers.push(sigmoidLayer);

        var rbmLayer = new RBM({
            'input' : layerInput,
            'n_visible' : inputSize,
            'n_hidden' : settings['hidden_layer_sizes'][i]
        });
        self.rbmLayers.push(rbmLayer);
    }
    self.outputLayer = new HiddenLayer({
        'input' : self.sigmoidLayers[self.sigmoidLayers.length-1].sampleHgivenV(),
        'n_in' : settings['hidden_layer_sizes'][settings['hidden_layer_sizes'].length - 1],
        'n_out' : settings['n_outs'],
        'activation' : math.sigmoid
    });
};

DBN.prototype.pretrain = function (settings) {
    var self = this;
    var lr = 0.6, k = 1, epochs = 2000;
    if(typeof settings['lr'] !== 'undefined')
        lr = settings['lr'];
    if(typeof settings['k'] !== 'undefined')
        k = settings['k'];
    if(typeof settings['epochs'] !== 'undefined')
        epochs = settings['epochs'];

    var i,j;
    for(i=0; i<self.nLayers ; i++) {
        var layerInput ,rbm;
        if (i==0)
            layerInput = self.x;
        else
            layerInput = self.sigmoidLayers[i-1].sampleHgivenV(layerInput);
        rbm = self.rbmLayers[i];
        rbm.set('log level',0);
        rbm.train({
            'lr' : lr,
            'k' : k,
            'input' : layerInput,
            'epochs' : epochs
        });

        if(self.settings['log level'] > 0) {
            console.log("DBN RBM",i,"th Layer Final Cross Entropy: ",rbm.getReconstructionCrossEntropy());
            console.log("DBN RBM",i,"th Layer Pre-Training Completed.");
        }

        // Synchronization between RBM and sigmoid Layer
        self.sigmoidLayers[i].W = rbm.W;
        self.sigmoidLayers[i].b = rbm.hbias;
    }
    if(self.settings['log level'] > 0)
        console.log("DBN Pre-Training Completed.")
};

DBN.prototype.finetune = function (settings) {
    var self = this;
    var lr = 0.2, epochs = 1000;
    if(typeof settings['lr'] !== 'undefined')
        lr = settings['lr'];
    if(typeof settings['epochs'] !== 'undefined')
        epochs = settings['epochs'];

    //Fine-Tuning Using MLP (Back Propagation)
    var i;
    var pretrainedWArray = [], pretrainedBArray = []; // HiddenLayer W,b values already pretrained by RBM.
    for(i=0; i<self.nLayers ; i++) {
        pretrainedWArray.push(self.sigmoidLayers[i].W);
        pretrainedBArray.push(self.sigmoidLayers[i].b);
    }
    // W,b of Final Output Layer are not involved in pretrainedWArray, pretrainedBArray so they will be treated as undefined at MLP Constructor.
    var mlp = new MLP({
        'input' : self.x,
        'label' : self.y,
        'n_ins' : self.nIns,
        'n_outs' : self.nOuts,
        'hidden_layer_sizes' : self.hiddenLayerSizes,
        'w_array' : pretrainedWArray,
        'b_array' : pretrainedBArray
    });
    mlp.set('log level',self.settings['log level']);
    mlp.train({
        'lr' : lr,
        'epochs' : epochs
    });
    for(i=0; i<self.nLayers ; i++) {
        self.sigmoidLayers[i].W = mlp.sigmoidLayers[i].W;
        self.sigmoidLayers[i].b = mlp.sigmoidLayers[i].b;
    }
    self.outputLayer.W = mlp.sigmoidLayers[self.nLayers].W;
    self.outputLayer.b = mlp.sigmoidLayers[self.nLayers].b;

};

DBN.prototype.getReconstructionCrossEntropy = function() {
    var self = this;
    var reconstructedOutput = self.predict(self.x);
    var a = math.activateTwoMat(self.y,reconstructedOutput,function(x,y){
        return x*Math.log(y);
    });

    var b = math.activateTwoMat(self.y,reconstructedOutput,function(x,y){
        return (1-x)*Math.log(1-y);
    });

    var crossEntropy = -math.meanVec(math.sumMatAxis(math.addMat(a,b),1));
    return crossEntropy
};

DBN.prototype.predict = function (x) {
    var self = this;
    var layerInput = x, i;
    for(i=0; i<self.nLayers ; i++) {
        layerInput = self.sigmoidLayers[i].output(layerInput);
    }
    var output = self.outputLayer.output(layerInput);
    return output;
};

DBN.prototype.set = function(property,value) {
    var self = this;
    self.settings[property] = value;
}