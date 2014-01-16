/**
 * Created by joonkukang on 2014. 1. 12..
 */
utils = module.exports;

utils.math = require('./math');
/*

utils.log = function(message,logLevel,type) {
    // log level. 0 : nothing, 1 : info, 2 : warning
    if(logLevel == 1 && type === 'info') {
        console.log(message);
    } else if(logLevel == 2) {
        if(type === 'info')
            console.log(message);
        else if (type === 'warning')
            console.warn(message);
    }

}*/