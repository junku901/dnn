/**
 * Created by joonkukang on 2014. 1. 13..
 */
var math = require('./utils').math;
LogisticRegression = require('./LogisticRegression');
HiddenLayer = require('./HiddenLayer');
RBM = require('./RBM');
CRBM = require('./CRBM');
DBN = require('./DBN');


CDBN = module.exports = function (settings) {
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

        var rbmLayer;
        if(i==0) {
            rbmLayer = new CRBM({
                'input' : layerInput,
                'n_visible' : inputSize,
                'n_hidden' : settings['hidden_layer_sizes'][i],
            });
        } else {
            rbmLayer = new RBM({
                'input' : layerInput,
                'n_visible' : inputSize,
                'n_hidden' : settings['hidden_layer_sizes'][i]
            });
        }
        self.rbmLayers.push(rbmLayer);
    }
    self.outputLayer = new HiddenLayer({
        'input' : self.sigmoidLayers[self.sigmoidLayers.length-1].sampleHgivenV(),
        'n_in' : settings['hidden_layer_sizes'][settings['hidden_layer_sizes'].length - 1],
        'n_out' : settings['n_outs'],
        'activation' : math.sigmoid
    });
};

CDBN.prototype.__proto__ = DBN.prototype;