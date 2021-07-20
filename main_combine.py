import para_pre_process
import model
import json

def main(text):
    paraInput = ""
    paraInput = text
    usedFeatures = para_pre_process.main(paraInput)
    finalAns = model.main(usedFeatures)

    print(finalAns)
    return finalAns

if __name__ == '__main__':
    paraInput = input("Enter STT paragraph:")
    main(paraInput)