const electron = require('electron')

const { app, BrowserWindow } = require('electron')

function createWindow () {   
  // 创建浏览器窗口
  let win = new BrowserWindow({ width: 1280, height: 780 , frame: true})

  // 然后加载 app 的 index.html.
  win.loadFile('./webs/main.html')
  win.on('closed', () => {
    win = null
  })
}

app.on('ready', createWindow)
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      app.quit()
    }
  })
  
  app.on('activate', () => {
    if (win === null) {
      createWindow()
    }
  })