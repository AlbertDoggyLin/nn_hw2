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
    import inquirer

    questions = [
        inquirer.List(
            "model",
            message="What model do you want to use when training data?",
            choices=["MLP", "RBFN"],
        ),
    ]
    answers = inquirer.prompt(questions)
    if answers['model']=='RBFN':
        app=App(readData, runDataOnRBFN)
    else:
        app=App(readData, runDataOnMLP)
    app.startAppSync()
