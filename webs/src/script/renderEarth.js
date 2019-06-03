//渲染世界地图和下方图表
const fs = require('fs');

function GetGeoData() {
    fs.readFile("../static/GeoData.data", function (err, data) {
        if (err) {
            console.log(err);
        } else {
            var tempData = JSON.parse(data);
            console.log(tempData["Albania"]);
            return tempData;
        }
    });
}

function RenderEarth() {
    var geoCoordMap = GetGeoData();
    var schema = [
        "1985", "1986", "1987", "1988",
        "1989", "1990", "1991", "1992",
        "1993", "1994", "1995", "1996",
        "1997", "1998", "1999", "2000",
        "2001", "2002", "2003", "2004",
        "2005", "2006", "2007", "2008",
        "2009", "2010", "2011", "2012", 
        "2013", "2014", "2015","2016"
    ];
}