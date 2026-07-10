# Relative Reproduction number for viral subject to baseline variants

#-----------------Package Setup--------------------*
import csv
import matplotlib.pyplot as plt
import numpy as np

#--------------------Data INPUT--------------------#
print("Data Parsing")
datafile="japan.csv" #input("Data CSV file name \\(assuming same directory as program\\)")
debug=False
with open(datafile, newline='') as csvfile:
    parser=list(csv.reader(csvfile))
    #Identify data rows
    i=0
    for row in parser:
            if column[n] == "date_from":
                datef=n

            if column[n] == "date_till":
                datet=n

            if column[n] == "Alpha":
                alpha=n

            if column[n] == "R.1":
                r1=n

            if column[n] == "Delta":
                delta=n

            if column[n] == "other":
                other=n


    if(debug):
        print("------------INPUT CSV FILE READING----------")
        for row in parser:                    #Debug for data parsing
            print(row)

        print("Graphing input data")

        

#----------------Processing------------------------#

#--------------------Data OUTPUT-------------------#