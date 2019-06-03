const fs = require('fs');

var TempArray = {};
var TargetArray =new Array();

fs.readFile("./master.json", function (err, data) {
    if (err) {
        console.log(err);
    } else {
        TempArray=JSON.parse(data);
//----------------------------------------------
        //制作世界地图数据
        // ["年份"：死亡人数]
        

    }

});