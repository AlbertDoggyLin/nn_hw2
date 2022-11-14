from model import runDataOnMLP, runDataOnRBFN
def readData(dirPos:str)->list:
    dataList=[]
    with open(dirPos) as data:
        for line in data:
            dataList.append(line.strip())
    return dataList
        
if __name__=="__main__":
    from UI import App
    # track=readData('resource/track.txt')
    # dataset=readData('resource/train6DAll.txt')
    # runDataOnRBFN(track, dataset)
    print('hello exe file')
    app=App(readData, runDataOnRBFN)
    #app=App(readData, runDataOnMLP)
    app.startAppSync()
