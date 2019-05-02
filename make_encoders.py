encoders_file = open('tag_encoders.pickle','wb')
import pickle
import os
# encoders = pickle.load(encoders_file)
# print (*[encoders[i].classes_ for i in range((len(encoders)))], sep="\n")

from sklearn.preprocessing import LabelEncoder

DATASET_DIR = "datasets/HDTB_pre_release_version-0.05/IntraChunk/CoNLL/utf/news_articles_and_heritage/Training/"
encoders = [ LabelEncoder() for x in range(6)]
pos=[]
gender=[]
number=[]
person=[]
case=[]
tam=[]
files = [f for f in os.listdir(DATASET_DIR) if os.path.isfile(os.path.join(DATASET_DIR, f))]
BASE_DIR = os.path.join(os.path.dirname(__file__),DATASET_DIR)
for xyz in files:
    file = open(os.path.join(BASE_DIR, xyz), 'r')
    for line in file.readlines():
        inp = line.split('\t')
        if len(inp)<3:
            continue
        pos.append(inp[3])
        features = inp[5]
        if len(features)>1:
            features = features.split('|')
        for f in features:
            keyval = f.split('-')
            if keyval[1]=='':
                continue
            key,val=keyval
            if key=='case':
                case.append(val)
            if key == 'num':
                number.append(val)
            if key=='pers':
                person.append(val)
            if key=="gen":
                gender.append(val)
            if key=="tam":
                tam.append(val)
    file.close()
pos.append("Unk")
gender.append("Unk")
number.append("Unk")
person.append("Unk")
case.append("Unk")
tam.append("Unk")
encoders[0].fit(pos)
encoders[1].fit(gender)
encoders[2].fit(number)
encoders[3].fit(person)
encoders[4].fit(case)
encoders[5].fit(tam)
print(encoders[5].classes_)
pickle.dump(encoders, encoders_file)
encoders_file.close()
