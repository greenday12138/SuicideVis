

var schema = [
    "1985",
    "1986",
    "1987",
    "1988",
    "1989",
    "1990",
    "1991",
    "1992",
    "1993",
    "1994",
    "1995",
    "1996",
    "1997",
    "1998",
    "1999",
    "2000",
    "2001",
    "2002",
    "2003",
    "2004",
    "2005",
    "2006",
    "2007",
    "2008",
    "2009",
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015"
];


function makeMapData(rawData) {
    var mapData = [];
    for (var i = 0; i < rawData.length; i++) {
        var geoCoord = geoCoordMap[rawData[i][0]];
        if (geoCoord) {
            console.log({
                name: rawData[i][0],
                value: geoCoord.concat(rawData[i][1])
            });
            mapData.push({
                name: rawData[i][0],
                value: geoCoord.concat(rawData[i][1])
            });
        }
    }
    return mapData;
};

function makeMapColor(rawData) {
    var mapData = [];
    for (var i = 0; i < rawData.length; i++) {
        var geoCoord = geoCoordMap[rawData[i][0]];
        if (geoCoord) {
            var temp = {
                name: rawData[i][0],
                value: rawData[i][1]
            }
            mapData.push(temp);
        }
    }
    return mapData;
};

function makeParallelAxis(schema) {
    var parallelAxis = [];
    for (var i = 1; i < schema.length; i++) {
        parallelAxis.push({ dim: i, name: schema[i] });
    }
    return parallelAxis;
}


option = {
    backgroundColor: new echarts.graphic.RadialGradient(0.5, 0.5, 0.4, [{
        offset: 0,
        color: '#4b5769'
    }, {
        offset: 1,
        color: '#404a59'
    }]),
    //biaoti
    title: {
        text: 'World Suicide Vis',
        subtext: 'data from kaggle',
        left: 'center',
        top: 5,
        itemGap: 0,
        textStyle: {
            color: '#fff'
        },
        z: 200
    },
    tooltip: {
        trigger: 'item',
        formatter: function (params) {
            return params.seriesName + '<br/>' + params.name + ' : ' + params.value[2];
        }
    },
    //gong ju kuang
    toolbox: {
        show: true,
        left: 'right',
        iconStyle: {
            normal: {
                borderColor: '#ddd'
            }
        },
        feature: {
        },
        z: 202
    },
    brush: {
        geoIndex: 0,
        brushLink: 'all',
        inBrush: {
            opacity: 1,
            symbolSize: 14
        },
        outOfBrush: {
            color: '#000',
            opacity: 0.2
        },
        z: 10
    },
    geo: {
        map: 'world',
        silent: true,
        label: {
            emphasis: {
                show: false,
                areaColor: '#2a333d'
            }
        },
        itemStyle: {
            normal: {
                borderColor: 'rgba(0, 0, 0, 0.2)'
            },
            emphasis: {
                areaColor: null,
                shadowOffsetX: 0,
                shadowOffsetY: 0,
                shadowBlur: 20,
                borderWidth: 0,
                shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
        },
        left: '4%',
        top: 40,
        bottom: '40%',
        right: '4%',
        roam: true
        // itemStyle: {
        //     normal: {
        //         areaColor: '#323c48',
        //         borderColor: '#111'
        //     },
        //     emphasis: {
        //         areaColor: '#2a333d'
        //     }
        // }
    },
    parallelAxis: makeParallelAxis(schema),
    grid: [{
        show: true,
        left: 0,
        right: 0,
        top: '63%',
        bottom: 0,
        borderColor: 'transparent',
        backgroundColor: '#404a59',
        z: 99
    }, {
        show: true,
        left: 0,
        right: 0,
        top: 0,
        height: 28,
        borderColor: 'transparent',
        backgroundColor: '#404a59',
        z: 199
    }],
    parallel: {
        top: '65%',
        left: 20,
        right: 20,
        bottom: 50,
        axisExpandable: true,
        axisExpandCenter: 15,
        axisExpandCount: 10,
        axisExpandWidth: 60,
        axisExpandTriggerOn: 'mousemove',

        z: 100,
        parallelAxisDefault: {
            type: 'value',
            nameLocation: 'start',
            nameRotate: 25,
            // nameLocation: 'end',
            nameTextStyle: {
                fontSize: 12
            },
            nameTruncate: {
                maxWidth: 170
            },
            nameGap: 20,
            splitNumber: 3,
            tooltip: {
                show: true
            },
            axisLine: {
                // show: false,
                lineStyle: {
                    width: 1,
                    color: 'rgba(255,255,255,0.3)'
                }
            },
            axisTick: {
                show: false
            },
            splitLine: {
                show: false
            },
            z: 100
        }
    },

    visualMap: {
        min: 10,
        max: 450,
        left: 'left',
        top: 'bottom',
        text: ['High', 'Low'],
        seriesIndex: [1],
        inRange: {
            color: [

                "#1a9641",
                "#a6d96a",
                "#ffffbf",
                "#fdae61",
                "#d7191c"
            ]
        },
        calculable: true
    },

    series: [
        {
            name: 'gdp and suicide',
            type: 'scatter',
            coordinateSystem: 'geo',
            //symbolSize: 8,
            data: makeMapData(rawData),
            activeOpacity: 1,
            label: {
                normal: {
                    formatter: '{b}',
                    position: 'right',
                    show: false
                },
                emphasis: {
                    show: true
                }
            },
            //symbolSize: 10,
            symbolSize: function (data) {
                return Math.max(20, data[2] / 200000);
            },
            itemStyle: {
                normal: {
                    borderColor: '#fff',
                    color: '#577ceb',
                }
            }
        },
        {
            name: 'categoryA',
            type: 'map',
            geoIndex: 0,
            // tooltip: {show: false},
            data: makeMapColor(rawData),
        },
        {
            name: 'parallel',
            type: 'parallel',
            smooth: true,
            lineStyle: {
                normal: {
                    color: '#21f9ff',
                    width: 0.5,
                    opacity: 0.6
                }
            },
            z: 100,
            blendMode: 'lighter',
            data: Multiline
        }
    ]
};



maychart.setOption(option);