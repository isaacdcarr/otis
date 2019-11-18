import csv
from os import path

def main():
    print("... obtaining second dataset ")
    with open('data/rsna-dataset/stage_2_train_labels.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        num_there = 0
        num_nothere = 0
        num_goodtar = 0
        num_badtar = 0
        num_dup = 0
        seen = []
        dup = {}
        for row in csv_reader:
            if row["patientId"] not in seen:
                seen.append(row["patientId"])
            else:
                num_dup += 1
                if row["patientId"] not in dup.keys():
                    dup[row["patientId"]] = 1
                else:
                    dup[row["patientId"]] += 1
            if path.exists('data/rsna-dataset/train_img/' + row["patientId"] + '.png'):
                num_there += 1
            else:
                num_nothere += 1
            print(f"there: {num_there}, nothere: {num_nothere}, dup: {num_dup}",end='\r')
    print(f"there: {num_there}, nothere: {num_nothere}, dup: {num_dup}")
    for i in dup.keys():
        print(f"{i}: {dup[i]}")
if __name__ == '__main__':
    main()
    