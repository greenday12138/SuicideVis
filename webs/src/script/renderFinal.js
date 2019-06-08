var dataBJ = [
    [1,55,9,56,0.46,18,6,"良"],
    [2,25,11,21,0.65,34,9,"优"],
    [3,56,7,63,0.3,14,5,"良"],
    [4,33,7,29,0.33,16,6,"优"],
    [5,42,24,44,0.76,40,16,"优"],
    [6,82,58,90,1.77,68,33,"良"],
    [7,74,49,77,1.46,48,27,"良"],
    [8,78,55,80,1.29,59,29,"良"],
    [9,80,216,280,4.8,108,64,"重度污染"],
    [10,75,127,216,2.52,61,27,"中度污染"],
    [11,39,19,38,0.57,31,15,"优"],
    [12,41,11,40,0.43,21,7,"优"],
    [13,64,38,74,1.04,46,22,"良"],
    [14,50,79,120,1.7,75,41,"轻度污染"],
    [15,61,63,116,1.48,44,26,"轻度污染"],
    [16,33,6,29,0.34,13,5,"优"],
    [17,94,66,110,1.54,62,31,"良"],
    [18,77,142,192,3.88,93,79,"中度污染"],
    [19,57,31,54,0.96,32,14,"良"],
    [20,22,8,17,0.48,23,10,"优"],
    [21,39,15,36,0.61,29,13,"优"],
    [22,94,69,114,2.08,73,39,"良"],
    [23,99,73,110,2.43,76,48,"良"],
    [24,31,12,30,0.5,32,16,"优"],
    [25,42,27,43,1,53,22,"优"],
    [26,45,117,157,3.05,92,58,"中度污染"],
    [27,67,185,230,4.09,123,69,"重度污染"],
    [28,87,120,186,2.77,91,50,"中度污染"],
    [29,90,96,165,2.76,83,41,"轻度污染"],
    [30,52,24,60,1.03,50,21,"良"],
    [31,46,5,49,0.28,10,6,"优"]
];

var dataGZ = [
    [1,26,37,27,1.163,27,13,"优"],
    [2,85,62,71,1.195,60,8,"良"],
    [3,78,38,74,1.363,37,7,"良"],
    [4,21,21,36,0.634,40,9,"优"],
    [5,41,42,46,0.915,81,13,"优"],
    [6,56,52,69,1.067,92,16,"良"],
    [7,64,30,28,0.924,51,2,"良"],
    [8,55,48,74,1.236,75,26,"良"],
    [9,76,85,113,1.237,114,27,"良"],
    [10,91,81,104,1.041,56,40,"良"],
    [11,84,39,60,0.964,25,11,"良"],
    [12,64,51,101,0.862,58,23,"良"],
    [13,70,69,120,1.198,65,36,"良"],
    [14,77,105,178,2.549,64,16,"良"],
    [15,90,68,87,0.996,74,29,"轻度污染"],
    [16,73,68,97,0.905,51,34,"良"],
    [17,54,27,47,0.592,53,12,"良"],
    [18,51,61,97,0.811,65,19,"良"],
    [19,91,71,121,1.374,43,18,"良"],
    [20,73,102,182,2.787,44,19,"良"],
    [21,73,50,76,0.717,31,20,"良"],
    [22,84,94,140,2.238,68,18,"良"],
    [23,93,77,104,1.165,53,7,"良"],
    [24,99,130,227,3.97,55,15,"良"],
    [25,80,84,139,1.094,40,17,"轻度污染"],
    [26,75,108,137,1.481,48,15,"轻度污染"],
    [27,81,48,62,1.619,26,3,"良"],
    [28,56,48,68,1.336,37,9,"良"],
    [29,82,92,174,3.29,0,13,"良"],
    [30,78,116,188,3.628,101,16,"轻度污染"],
    [31,94,50,0,1.383,76,11,"轻度污染"]
];



var schema2 = [
    {name: 'date', index: 0, text: '日'},
    {name: 'AQIindex', index: 1, text: 'AQI指数'},
    {name: 'PM25', index: 2, text: 'PM2.5'},
    {name: 'PM10', index: 3, text: 'PM10'},
    {name: 'CO', index: 4, text: '一氧化碳（CO）'},
    {name: 'NO2', index: 5, text: '二氧化氮（NO2）'},
    {name: 'SO2', index: 6, text: '二氧化硫（SO2）'}
];


var itemStyle = {
    normal: {
        opacity: 0.8,
        shadowBlur: 10,
        shadowOffsetX: 0,
        shadowOffsetY: 0,
        shadowColor: 'rgba(0, 0, 0, 0.5)'
    }
};

option3 = {
    backgroundColor: '#404a59',
    color: [
        '#dd4444', '#fec42c', '#80F1BE'
    ],
    legend: {
        y: 'top',
        data: ['男', '女'],
        textStyle: {
            color: '#fff',
            fontSize: 16
        }
    },
    grid: {
        x: '10%',
        x2: 150,
        y: '18%',
        y2: '10%'
    },
    tooltip: {
        padding: 10,
        backgroundColor: '#222',
        borderColor: '#777',
        borderWidth: 1,
        formatter: function (obj) {
            var value = obj.value;
            return '<div style="border-bottom: 1px solid rgba(255,255,255,.3); font-size: 18px;padding-bottom: 7px;margin-bottom: 7px">'
                + obj.seriesName + ' ' + value[0] + '日：'
                + value[7]
                + '</div>'
                + schema2[1].text + '：' + value[1] + '<br>'
                + schema2[2].text + '：' + value[2] + '<br>'
                + schema2[3].text + '：' + value[3] + '<br>'
                + schema2[4].text + '：' + value[4] + '<br>'
                + schema2[5].text + '：' + value[5] + '<br>'
                + schema2[6].text + '：' + value[6] + '<br>';
        }
    },
    xAxis: {
        type: 'value',
        name: 'GDP',
        nameGap: 16,
        nameTextStyle: {
            color: '#fff',
            fontSize: 14
        },
        max: 31,
        splitLine: {
            show: false
        },
        axisLine: {
            lineStyle: {
                color: '#eee'
            }
        }
    },
    yAxis: {
        type: 'value',
        name: '年龄',
        nameLocation: 'end',
        nameGap: 20,
        nameTextStyle: {
            color: '#fff',
            fontSize: 16
        },
        axisLine: {
            lineStyle: {
                color: '#eee'
            }
        },
        splitLine: {
            show: false
        }
    },
    visualMap: [
        {
            left: 'right',
            top: '10%',
            dimension: 2,
            min: 0,
            max: 250,
            itemWidth: 30,
            itemHeight: 60,
            calculable: true,
            precision: 0.1,
            text: ['圆形大小：死亡率'],
            textGap: 30,
            textStyle: {
                color: '#fff'
            },
            inRange: {
                symbolSize: [10, 70]
            },
            outOfRange: {
                symbolSize: [10, 70],
                color: ['rgba(255,255,255,.2)']
            },
            controller: {
                inRange: {
                    color: ['#c23531']
                },
                outOfRange: {
                    color: ['#444']
                }
            }
        },
        {
            left: 'right',
            bottom: '5%',
            dimension: 6,
            min: 0,
            max: 50,
            itemHeight: 60,
            calculable: true,
            precision: 0.1,
            text: ['明暗：死亡人数'],
            textGap: 30,
            textStyle: {
                color: '#fff'
            },
            inRange: {
                colorLightness: [1, 0.5]
            },
            outOfRange: {
                color: ['rgba(255,255,255,.2)']
            },
            controller: {
                inRange: {
                    color: ['#c23531']
                },
                outOfRange: {
                    color: ['#444']
                }
            }
        }
    ],
    series: [
        {
            name: '男',
            type: 'scatter',
            itemStyle: itemStyle,
            data: dataBJ
        },
        
        {
            name: '女',
            type: 'scatter',
            itemStyle: itemStyle,
            data: dataGZ
        }
    ]
};
/*--------------------------------------------- */
ScatterChart.setOption(option3);