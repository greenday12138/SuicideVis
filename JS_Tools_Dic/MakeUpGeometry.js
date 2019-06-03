const fs = require('fs');

var TempArray={};
var TargetGeo=null;
//"Amsterdam": [4.895168,52.370216],

//地理数据预处理

fs.readFile("./CountryLocation.json",function(err,data){
    if(err){
        throw err;
    }else{
        TempArray=JSON.parse(data);
        console.log(TempArray[0]);
        TargetGeo={};
        for(i=0;i<TempArray.length;i++){
            var x=TempArray[i];
            //console.log(x.country);
            TargetGeo[x.country]=[x.longitude,x.latitude];
        }
        console.log(TargetGeo["Antigua and Barbuda"]);
        var TargetStr=JSON.stringify(TargetGeo);
        fs.writeFile("../webs/static/GeoData.data",TargetStr,function(err){
            if(err){
                console.log(err);
            }
        });
    }
});

