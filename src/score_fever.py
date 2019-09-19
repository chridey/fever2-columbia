import argparse
import json
import sys
import collections
from fever.scorer import fever_score
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--predicted_labels",type=str)

parser.add_argument("--predicted_evidence",type=str)
parser.add_argument("--actual",type=str)

args = parser.parse_args()

predicted_labels =[]
predicted_evidence = []
actual = []
attacks = collections.defaultdict(list)

with open(args.predicted_labels,"r") as predictions_file:
    for line in predictions_file:
        j = json.loads(line)
        if 'predicted' in j:
            predicted_labels.append(j['predicted'])
        elif 'predicted_label' in j:
            predicted_labels.append(j['predicted_label'])
        else:
            predicted_labels.append('NOT ENOUGH INFO')

pages = False
with open(args.predicted_evidence,"r") as predictions_file:
    for line in predictions_file:
        j = json.loads(line)
        if 'predicted_pages' in j:
            predicted_evidence.append([[i[0],0] for i in j['predicted_pages'][:5]])
            pages = True
        elif 'predicted_evidence' in j:
            predicted_evidence.append(j['predicted_evidence'])
        else:
            predicted_evidence.append(j['predicted_sentences'])
        
with open(args.actual, "r") as actual_file:
    for idx,line in enumerate(actual_file):
        j = json.loads(line)
        j['label'] = j['label'].upper()
        if pages and j['label'] != 'NOT ENOUGH INFO':
            new_evidence = []
            for evidence_set in j['evidence']:
                unique_evidence = set()
                for e in evidence_set:
                    unique_evidence.add((None, None, e[2], 0))
                new_evidence.append([list(i) for i in unique_evidence])
            j['evidence'] = new_evidence
        actual.append(j)
        if 'attack' in j:
            attacks[j['attack']].append(idx)


#for pe, j in zip(predicted_evidence, actual):
#    print(pe, j['evidence'])
            
predictions = []
for ev,label in zip(predicted_evidence,predicted_labels):
    predictions.append({"predicted_evidence":ev,"predicted_label":label})

score,acc,precision,recall,f1 = fever_score(predictions,actual)

tab = PrettyTable()
tab.field_names = ["FEVER Score", "Label Accuracy", "Evidence Precision", "Evidence Recall", "Evidence F1"]
tab.add_row((round(score,4),round(acc,4),round(precision,4),round(recall,4),round(f1,4)))

print(tab)

actually = [i['label'] for i in actual]
predicted = [i['predicted_label'] for i in predictions]
print(classification_report(actually, predicted))
print(confusion_matrix(actually, predicted))

tab = PrettyTable()
for attack,idxs in sorted(attacks.items(), key=lambda x:len(x[1]), reverse=True):
    attack_predictions = [predictions[i] for i in idxs]
    attack_actual = [actual[i] for i in idxs]
    try:
        score,acc,precision,recall,f1 = fever_score(attack_predictions,attack_actual)
    except ZeroDivisionError:
        score,acc,precision,recall,f1 = 0,0,0,0,0
    tab.field_names = ["Attack", "Count", "FEVER Score", "Label Accuracy", "Evidence Precision", "Evidence Recall", "Evidence F1"]
    tab.add_row((attack, len(attack_actual), round(score,4),round(acc,4),round(precision,4),round(recall,4),round(f1,4)))

print(tab)

for attack,idxs in sorted(attacks.items(), key=lambda x:len(x[1]), reverse=True):
    attack_predictions = [predictions[i]['predicted_label'] for i in idxs]
    attack_actual = [actual[i]['label'] for i in idxs]
    print(attack)
    print(classification_report(attack_predictions, attack_actual))
