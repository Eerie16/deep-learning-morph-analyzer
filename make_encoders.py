encoders_file = open('feature-encoders.pickle','rb')
import pickle

# encoders = pickle.load(encoders_file)
# print (*[encoders[i].classes_ for i in range((len(encoders)))], sep="\n")

from sklearn.preprocessing import LabelEncoder
file = open('treebank-train.conllu','r')
encoders = [ LabelEncoder() for x in range(6)]
pos=[]
gender=[]
number=[]
person=[]
case=[]
tam=[]
for line in file.readlines():
    inp = line.split('\t')
    if len(inp)<5:
        continue
    pos.append(inp[3])
    features = inp[5]
    if len(features)>1:
        features = features.split('|')
    for f in features:
        # print(f)
        keyval = f.split('=')
        if len(keyval)!=2:
            continue
        key,val=keyval
        if key=='Case':
            case.append(val)
        if key == 'Number':
            number.append(val)
        if key=='Person':
            person.append(val)
        if key=="Gender":
            gender.append(val)
    if len(inp)>9:
        fts = inp[9].split('|')
        for f in fts:
            if f[:3]=='Tam':
                tam.append(f[4:])

encoders[0].fit(pos)
encoders[1].fit(gender)
encoders[2].fit(number)
encoders[3].fit(person)
encoders[4].fit(case)
encoders[5].fit(tam)
print(encoders[5].classes_)
pickle.dump(encoders, encoders_file)
encoders_file.close()
file.close()