# dnn

Simple deep learning library for node.js.

Includes Logistic-Regression, MLP, RBM, DBN, CRBM, CDBN. (Deep Neural Network)

RBM is using contrastive-divergence for its training algorithm.

## Installation
```
$ npm install dnn
```

## Features

  * Logistic Regression
  * MLP (Multi-Layer Perceptron)
  * RBM (Restricted Boltzmann Machine)
  * DBN (Deep Belief Network)
  * CRBM (Restricted Boltzmann Machine with continuous-valued inputs)
  * CDBN (Deep Belief Network with continuous-valued inputs)

## Logistic Regression
```javascript
var dnn = require('dnn');
var x = [[1,1,1,0,0,0],
         [1,0,1,0,0,0],
         [1,1,1,0,0,0],
         [0,0,1,1,1,0],
         [0,0,1,1,0,0],
         [0,0,1,1,1,0]];
var y = [[1, 0],
         [1, 0],
         [1, 0],
         [0, 1],
         [0, 1],
         [0, 1]];

var lrClassifier = new dnn.LogisticRegression({
    'input' : x,
    'label' : y,
    'n_in' : 6,
    'n_out' : 2
});

lrClassifier.set('log level',1); // 0 : nothing, 1 : info, 2 : warning.

var training_epochs = 800, lr = 0.01;

lrClassifier.train({
    'lr' : lr,
    'epochs' : training_epochs
});

x = [[1, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 0],
     [1, 1, 1, 1, 1, 0]];

console.log("Result : ",lrClassifier.predict(x));
```

## MLP (Multi-Layer Perceptron)
```javascript
var dnn = require('dnn');
var x = [[0.4, 0.5, 0.5, 0.,  0.,  0.],
         [0.5, 0.3,  0.5, 0.,  0.,  0.],
         [0.4, 0.5, 0.5, 0.,  0.,  0.],
         [0.,  0.,  0.5, 0.3, 0.5, 0.],
         [0.,  0.,  0.5, 0.4, 0.5, 0.],
         [0.,  0.,  0.5, 0.5, 0.5, 0.]];
var y =  [[1, 0],
          [1, 0],
          [1, 0],
          [0, 1],
          [0, 1],
          [0, 1]];

var mlp = new dnn.MLP({
    'input' : x,
    'label' : y,
    'n_ins' : 6,
    'n_outs' : 2,
    'hidden_layer_sizes' : [4,4,5]
});

mlp.set('log level',1); // 0 : nothing, 1 : info, 2 : warning.

mlp.train({
    'lr' : 0.6,
    'epochs' : 20000
});

a = [[0.5, 0.5, 0., 0., 0., 0.],
     [0., 0., 0., 0.5, 0.5, 0.],
     [0.5, 0.5, 0.5, 0.5, 0.5, 0.]];

console.log(mlp.predict(a));
```

## RBM (Restricted Boltzmann Machine)
```javascript
var dnn = require('dnn');
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

rbm.set('log level',1); // 0 : nothing, 1 : info, 2 : warning.

var trainingEpochs = 500;

rbm.train({
    lr : 0.6,
    k : 1, // CD-k.
    epochs : trainingEpochs
});

var v = [[1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0]];

console.log(rbm.reconstruct(v));
console.log(rbm.sampleHgivenV(v)[0]); // get hidden layer probabilities from visible unit.
```

## DBN (Deep Belief Network)
```javascript
var dnn = require('dnn');
var x = [[1,1,1,0,0,0],
         [1,0,1,0,0,0],
         [1,1,1,0,0,0],
         [0,0,1,1,1,0],
         [0,0,1,1,0,0],
         [0,0,1,1,1,0]];
var y = [[1, 0],
         [1, 0],
         [1, 0],
         [0, 1],
         [0, 1],
         [0, 1]];

var pretrain_lr = 0.6, pretrain_epochs = 900, k = 1, finetune_lr = 0.6, finetune_epochs = 500;

var dbn = new dnn.DBN({
    'input' : x,
    'label' : y,
    'n_ins' : 6,
    'n_outs' : 2,
    'hidden_layer_sizes' : [10,12,11,8,6,4]
});

dbn.set('log level',1); // 0 : nothing, 1 : info, 2 : warning.

// Pre-Training using using RBM
dbn.pretrain({
    'lr' : pretrain_lr,
    'k' : k, // RBM CD-k.
    'epochs' : pretrain_epochs
});

// Fine-Tuning dbn using mlp backpropagation.
dbn.finetune({
    'lr' : finetune_lr,
    'epochs' : finetune_epochs
});

/*
for(var i =0;i<6;i++) {
    console.log(i+1,"th layer W : ",dbn.sigmoidLayers[i].W);
}
*/

x = [[1, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 0],
     [1, 1, 1, 1, 1, 0]];

console.log(dbn.predict(x));
```

## CRBM (Restricted Boltzmann Machine with continuous-valued inputs)
```javascript
var dnn = require('dnn');
var data = [[0.4, 0.5, 0.5, 0.,  0.,  0.7],
            [0.5, 0.3,  0.5, 0.,  1,  0.6],
            [0.4, 0.5, 0.5, 0.,  1,  0.9],
            [0.,  0.,  0., 0.3, 0.5, 0.],
            [0.,  0.,  0., 0.4, 0.5, 0.],
            [0.,  0.,  0., 0.5, 0.5, 0.]];

var crbm = new dnn.CRBM({
    input : data,
    n_visible : 6,
    n_hidden : 5
});

crbm.set('log level',1); // 0 : nothing, 1 : info, 2 : warning.

crbm.train({
    lr : 0.6,
    k : 1, // CD-k.
    epochs : 1500
});

var v = [[0.5, 0.5, 0., 0., 0., 0.],
         [0., 0., 0., 0.5, 0.5, 0.]];

console.log(crbm.reconstruct(v));
console.log(crbm.sampleHgivenV(v)[0]); // get hidden layer probabilities from visible unit.
```

## CDBN (Deep Belief Network with continuous-valued inputs)
```javascript
var dnn = require('dnn')

var x = [[0.4, 0.5, 0.5, 0.,  0.,  0.],
         [0.5, 0.3,  0.5, 0.,  0.,  0.],
         [0.4, 0.5, 0.5, 0.,  0.,  0.],
         [0.,  0.,  0.5, 0.3, 0.5, 0.],
         [0.,  0.,  0.5, 0.4, 0.5, 0.],
         [0.,  0.,  0.5, 0.5, 0.5, 0.]];

var y = [[1, 0],
         [1, 0],
         [1, 0],
         [0, 1],
         [0, 1],
         [0, 1]];

var cdbn = new dnn.CDBN({
    'input' : x,
    'label' : y,
    'n_ins' : 6,
    'n_outs' : 2,
    'hidden_layer_sizes' : [10,12,11,8,6,4]
});

cdbn.set('log level',1); // 0 : nothing, 1 : info, 2 : warning.

var pretrain_lr = 0.8, pretrain_epochs = 1600, k= 1, finetune_lr = 0.84, finetune_epochs = 10000;

// Pre-Training using using RBM, CRBM.
cdbn.pretrain({
    'lr' : pretrain_lr,
    'k' : k, // RBM CD-k.
    'epochs' : pretrain_epochs
});

// Fine-Tuning dbn using mlp backpropagation.
cdbn.finetune({
    'lr' : finetune_lr,
    'epochs' : finetune_epochs
});

/*
for(var i =0;i<6;i++) {
    console.log(i+1,"th layer W : ",cdbn.sigmoidLayers[i].W);
}
*/

a = [[0.5, 0.5, 0., 0., 0., 0.],
     [0., 0., 0., 0.5, 0.5, 0.],
     [0.1,0.2,0.4,0.4,0.3,0.6]];

console.log(cdbn.predict(a));
```

##License

(The MIT License)

Copyright (c) 2014 Joon-Ku Kang &lt;junku901@gmail.com&gt;

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
