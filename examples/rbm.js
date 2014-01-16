/**
 * Created by joonkukang on 2014. 1. 15..
 */
var dnn = require('../lib/dnn');
var data = [[1,1,1,0,0,0],
    [1,0,1,0,0,0],
    [1,1,1,0,0,0],
    [0,0,1,1,1,0],
    [0,0,1,1,0,0],
    [0,0,1,1,1,0]];

var rbm = new dnn.RBM({
    input : data,
    n_visible : 6,
    n_hidden : 2
});

rbm.set('log level',1);
var trainingEpochs = 500;

rbm.train({
    lr : 0.6,
    k : 1,
    epochs : trainingEpochs
});



var v = [[1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0]];

console.log(rbm.reconstruct(v));
console.log(rbm.sampleHgivenV(v)[0]);