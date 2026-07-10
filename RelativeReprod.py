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
debug = False

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
datesn = []
for row in parser[1:]:
    date_from = dt.datetime.strptime(row[datef], "%Y-%m-%d").date()
    date_till = dt.datetime.strptime(row[datet], "%Y-%m-%d").date()
    datesn.append([date_from, date_till, int((date_till - date_from).days)+1])

reproday=np.delete(parser[1:], [datef, datet], axis=1)
#remove NA values, floats
reproday=np.char.replace(reproday, 'NA', '0').astype(float)
# reproday = reproday_array.tolist()    #Make to list for reproday if needed

if(debug):
#---------------random debug stuff-------------#
    for row in datesn:
        print(row)
        print(int(row[2]))

    for i in reproday:
        print(i)
    values = []
    values2 = []
    dates = []

    print("------------INPUT CSV FILE READING----------")

    print("Variants found:")
    for variant in variant_cols:
        print(variant)

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
#---------------random debug stuff-------------#

print("Graphing input data")
# if(debug):
#---------------random debug stuff-------------#
        # plt.plot(dates, values)
        # plt.plot(dates, values2)
        # plt.bar([row[0] for row in datesn], [row[2] for row in datesn])
        # plt.xticks(rotation=45)
        # plt.show()

#----------------Processing------------------------#
xval=[row[0] for row in datesn]    #x values for the plot
ploty=np.transpose(reproday)    # 2D array by variant then time
relploty=[]
percentresult=[]
# Relative reproduction nr. calculation
for i in reproday:
    daytotal=np.sum(i)
    if daytotal==0:
        daytotal=1
    row_percents = [(val / daytotal) for val in i]
    percentresult.append(row_percents)
for i in percentresult:
    print(i,"\n")

#---------------random debug stuff (plot a graph)-------------#
plot=True
ploty=np.transpose(np.array(percentresult))
if(plot):
    # set up variables in a standard way for plotting
    for name, y in zip(variant_cols.keys(), ploty):
        plt.scatter(xval[:len(y)], y, s=12, label=name)

    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


#--------------------Data OUTPUT-------------------#