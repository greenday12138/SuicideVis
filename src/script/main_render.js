/*
 * 本文件处理和页面渲染相关的脚本，定义了一些渲染页面的组建和接口
*/

M.AutoInit();

//获取图表渲染组件

var maychart=echarts.init(document.getElementById("left-earth"));

var riverchart=echarts.init(document.getElementById("river"));

var ScatterChart=echarts.init(document.getElementById("Scatter"));

var ScatterChartUp=echarts.init(document.getElementById("ScatterUP"));
document.getElementById("ScatterUP").style.display="none";
//

var ScatterMaxBTN=document.getElementById("Max_Scatter");
ScatterMaxBTN.addEventListener('click',ShowScatterUp);

var isShow=false;
function ShowScatterUp(){
    if(isShow){
        document.getElementById("ScatterUP").style.display="none";
        isShow=false;
    }else{
        
        document.getElementById("ScatterUP").style.width="100%";
        document.getElementById("ScatterUP").style.height="100%";
        document.getElementById("ScatterUP").style.display="block";
       // ScatterChartUp=echarts.init(document.getElementById("ScatterUP"));
        isShow=true;
    }
}