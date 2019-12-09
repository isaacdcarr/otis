import csv

def main():
  results = {}
  with open('2019-11-19_11:49_21_newmodel_alldata_100epochs_300size.csv') as file:
    csv_reader = csv.reader(file,delimiter=',')
    count = 0
    for row in csv_reader:
      if count != 0:
        results[int(row[0])+1] =[]
        results[int(row[0])+1].append(float(row[3])*100)
      count += 1

  with open('2019-11-21_09:54_24_newnew_alldata_split_adddropout_100epochs_300size.csv') as file:
    csv_reader = csv.reader(file,delimiter=',')
    count = 0
    for row in csv_reader:
      if count != 0:
        results[int(row[0])+1].append(float(row[3])*100)
      count += 1

  for k, v in results.items():
    print(f"{k} & {v[0]} & {v[1]} \\\\")

if __name__ == '__main__':
  main()