const fs = require('fs');

let TempArray={};

fs.readFile("./CountryLocation.json",function(err,data){
    if(err){
        throw err;
    }else{
        TempStr=data;
        TempArray=JSON.parse(data);
    }
});

console.log(TempArray[0]);

