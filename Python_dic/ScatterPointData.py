import json
from copy import deepcopy

with open(r'E:\Project\Python\test\master.json', 'r') as fr:
    data = json.load(fr)

ScatterPointData = dict()
for item in data:
    templist = list()
    templist.append(item["gdp_per_capita($)"])
    templist.append(item["age"])
    templist.append(item["suicides/100k pop"])
    templist.append(item["suicides_no"])
    templist.append(item["gdp_for_year($)"])
    templist.append(item["generation"])
    if item["sex"] == "male":
        try:
            ScatterPointData[item["country"]]["dataMale"].append(templist)
        except (AttributeError, KeyError):
            try:
                ScatterPointData[item["country"]]["dataMale"] = []
                ScatterPointData[item["country"]]["dataMale"].append(templist)
            except KeyError:
                ScatterPointData[item["country"]] = {"dataMale": [templist]}
    elif item["sex"] == "female":
        try:
            ScatterPointData[item["country"]]["dataFemale"].append(templist)
        except (AttributeError, KeyError):
            try:
                ScatterPointData[item["country"]]["dataFemale"] = []
                ScatterPointData[item["country"]]["dataFemale"].append(templist)
            except KeyError:
                ScatterPointData[item["country"]] = {"dataFemale": [templist]}
# print(ScatterPointData)
# print(ScatterPointData["Albania"]["dataFemale"])
with open(r'.\ScatterPointData.json', 'w') as fw:
    json.dump(ScatterPointData, fw, sort_keys=False, indent=4)
