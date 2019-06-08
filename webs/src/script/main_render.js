/*
 * 本文件处理和页面渲染相关的脚本，定义了一些渲染页面的组建和接口
*/

M.AutoInit();

//获取图表渲染组件

var maychart=echarts.init(document.getElementById("left-earth"));

var riverchart=echarts.init(document.getElementById("river"));

var ScatterChart=echarts.init(document.getElementById("Scatter"));

//