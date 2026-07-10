# Relative Reproduction number for viral subject to baseline variants

#-----------------Package Setup--------------------*
import csv
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt


#--------------------Data INPUT--------------------#
print("Data Parsing")
datafile = "japan.csv"  # input("Data CSV file name (assuming same directory as program)")
baselinev = "Alpha" # input("Baseline variant")
debug = True

with open(datafile, newline='') as csvfile:
    parser = np.array(list(csv.reader(csvfile)))

header = parser[0]

datef = list(header).index("date_from")
datet = list(header).index("date_till")

# Automatically detect all variant columns
variant_cols = {
    name: i
    for i, name in enumerate(header)
    if name not in ("date_from", "date_till")
}

# Date parsing

datesn.append((
    dt.datetime.strptime(row[datef], "%Y-%m-%d").date(),
    dt.datetime.strptime(row[datet], "%Y-%m-%d").date()
))
if(debug):
    print("------------INPUT CSV FILE READING----------")

    print("Variants found:")
    for variant in variant_cols:
        print(variant)

    values = []
    values2 = []
    dates = []

    for row in parser[1:]:
        dates.append(row[datef])

        try:
            values.append(float(row[3]))
        except ValueError:
            values.append(0)

        try:
            values2.append(float(row[4]))
        except ValueError:
            values2.append(0)

    print("Graphing input data")
    #print(values)
    #print(values2)
    plt.plot(dates, values)
    plt.plot(dates, values2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    #plt.show()

#----------------Processing------------------------#
for date in datesn[1:]:
    values.append((date[0] - date[1]).days)
    print(date)


#--------------------Data OUTPUT-------------------#