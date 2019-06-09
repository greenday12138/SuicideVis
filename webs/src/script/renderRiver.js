/*
文件名：ScatterPointData.json*/
var optionDefault = {
    title:{
        text:"United States",//predefined first page
        padding:25
    },

    tooltip: {
        trigger: 'axis',
        axisPointer: {
            type: 'line',
            lineStyle: {
                color: 'rgba(0,0,0,0.2)',
                width: 1,
                type: 'solid'
            }
        }
    },
    
    legend: {
        data: ['MALE', 'FEMALE']
    },

    singleAxis: {
        top: 50,
        bottom: 50,
        axisTick: {},

        axisLabel: {
            show: false
        },
        type: 'time',
        axisPointer: {
            animation: true,
            label: {
                show: true
            }
        },
        splitLine: {
            show: true,
            lineStyle: {
                type: 'dashed',
                opacity: 0.2
            }
        }
    },

    series: [
        {
            type: 'themeRiver',
            itemStyle: {
                emphasis: {
                    shadowBlur: 20,
                    shadowColor: 'rgba(0, 0, 0, 0.8)'
                }
            },
            data: RiverData["United States"]

        }
    ]
};

riverchart.setOption(optionDefault);

function MakeUpRiverOption(r_data,m_name){
    var temp = {
        title:{
            text:m_name,
            padding:25 
        },

        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'line',
                lineStyle: {
                    color: 'rgba(0,0,0,0.2)',
                    width: 1,
                    type: 'solid'
                }
            }
        },
    
        legend: {
            data: ['MALE', 'FEMALE']
        },
    
        singleAxis: {
            top: 50,
            bottom: 50,
            axisTick: {},
            axisLabel: {
                show: true
            },
            type: 'time',
            axisPointer: {
                animation: true,
                label: {
                    show: true
                }
            },
            splitLine: {
                show: true,
                lineStyle: {
                    type: 'dashed',
                    opacity: 0.2
                }
            }
        },
    
        series: [
            {
                type: 'themeRiver',
                itemStyle: {
                    emphasis: {
                        shadowBlur: 20,
                        shadowColor: 'rgba(0, 0, 0, 0.8)'
                    }
                },
                data: r_data
    
            }
        ]
    };
    return temp;
}

function RefreshRiverChart(r_data,m_name){
    riverchart.setOption(MakeUpRiverOption(r_data,m_name));
}


