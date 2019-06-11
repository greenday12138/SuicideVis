/* 
这个文件主要处理页面的点击逻辑和app.js主进程交互
*/

//调取的electron模块，用于开启本地浏览器
const shell = require('electron').shell

//在github查看按钮
const exLinksBtn = document.getElementById("link-git")
//点击本地浏览器开启本项目地址
exLinksBtn.addEventListener('click', function (event) {
  shell.openExternal('https://github.com/greenday12138/SuicideVis')
})