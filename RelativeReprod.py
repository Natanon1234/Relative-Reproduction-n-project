# Relative Reproduction number for viral subject to baseline variants

#-----------------Package Setup--------------------*
import csv
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from scipy.optimize import curve_fit # Log regression
#--------------------Data INPUT--------------------#
print("Data Parsing")
datafile="japan.csv"  # input("Data CSV file name (assuming same directory as program)")
baselinev="Alpha" # input("Baseline variant")
debug=False

with open(datafile, newline='') as csvfile:
    parser=np.array(list(csv.reader(csvfile)))

header=parser[0]

datef=list(header).index("date_from")
datet=list(header).index("date_till")

# Automatically detect all variant columns
variant_cols={
    name: i
    for i, name in enumerate(header)
    if name not in ("date_from", "date_till")
}

# Date parsing
datesn=[]
for row in parser[1:]:
    date_from=dt.datetime.strptime(row[datef], "%Y-%m-%d").date()
    date_till=dt.datetime.strptime(row[datet], "%Y-%m-%d").date()
    datesn.append([date_from, date_till, int((date_till - date_from).days)+1])

reproday=np.delete(parser[1:], [datef, datet], axis=1)
#remove NA values, floats
reproday=np.char.replace(reproday, 'NA', '0').astype(float)
# reproday=reproday_array.tolist()    #Make to list for reproday if needed

if(debug):
#---------------random debug stuff-------------#
    for row in datesn:
        print(row)
        print(int(row[2]))

    for i in reproday:
        print(i)
    values=[]
    values2=[]
    dates=[]

    print("------------INPUT CSV FILE Reading----------")

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
    row_percents=[(val / daytotal) for val in i]
    percentresult.append(row_percents)
for i in percentresult:
    print(i,"\n")

#---------------random debug stuff (plot a graph)-------------#
plot=False
ploty=np.transpose(np.array(percentresult))
if(plot):
    # set up variables in a standard way for plotting
    for name, y in zip(variant_cols.keys(), ploty):
        plt.scatter(xval[:len(y)], y, s=12, label=name)

    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

#--------------Simple quadratic regressions (Chebyshev Extrapolation)-------#

# Create an empty list to store the regression models
plotregressquad=False
regression_models=[]

# This is a mini program for a quadratic regression written by AI
if plotregressquad:
    # xval already contains datetime.date objects, convert to ordinals for the math
    x_ord=np.array([d.toordinal() for d in xval])
    
    for name, y in zip(variant_cols.keys(), ploty):
        
        # 1. Plot original data using native datetime objects
        plt.scatter(xval, y, s=12, label=f"{name} (Data)")
        
        if len(x_ord) > 3:
            # 2. Fit Chebyshev polynomial using the class method.
            # This automatically maps the large ordinal dates to a [-1, 1] window,
            # which prevents numerical instability/crashing during extrapolation.
            cheb_model=np.polynomial.Chebyshev.fit(x_ord, y, 3)
            
            # Store the variant name and its corresponding model into our list
            regression_models.append({'variant': name, 'model': cheb_model})
            
            # 3. Create extended timeline (historical + 182 days)
            last_date=xval[-1]
            future_dates=[last_date + dt.timedelta(days=i) for i in range(1, 183)]
            extended_x_dates=xval + future_dates
            extended_x_ord=np.array([d.toordinal() for d in extended_x_dates])
            
            # 4. Evaluate the model on the extended dates
            extended_y=cheb_model(extended_x_ord)
            
            # 5. Plot the line using the SAME datetime object type as the scatter plot
            plt.plot(extended_x_dates, extended_y, linestyle='--', label=f"{name} (Extrap.)")

    # Move legend outside the plot so it doesn't cover the 6-month extrapolation
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.ylim(bottom=0, top=1)
    plt.tight_layout() # Ensures the rotated dates and legend fit in the window
    plt.show()

#--------------Multinomial Comparison (Linear vs Quadratic) -------# #Multinomial linear makes more sense, but apparently the quadratic work better?
plot_multinomial_comparison = True
days = 183

if plot_multinomial_comparison:
    # Convert dates to ordinals and normalize
    x_ord = np.array([d.toordinal() for d in xval])
    x_min = x_ord.min()
    x_norm = x_ord - x_min
    
    # 1. Create the extended timeline (historical + days)
    last_date = xval[-1]
    future_dates = [last_date + dt.timedelta(days=i) for i in range(1, days)]
    extended_x_dates = xval + future_dates
    extended_x_norm = np.array([d.toordinal() - x_min for d in extended_x_dates])
    
    # Dictionaries to store your separate coefficients
    polynomial_coeffs = {}
    linear_coeffs = {}
    
    # Temporary storage for unnormalized log-predictions
    raw_log_poly = {}
    raw_log_linear = {}
    
    # 2. Fit both regressions in a single loop
    for name, y in zip(variant_cols.keys(), ploty):
        log_y = np.log(y + 1e-4) # Safety constant for 0%
        
        # Polynomial (Quadratic - degree 2)
        poly_fit = np.polyfit(x_norm, log_y, 2)
        polynomial_coeffs[name] = poly_fit
        raw_log_poly[name] = np.polyval(poly_fit, extended_x_norm)
        
        # Linear (degree 1)
        linear_fit = np.polyfit(x_norm, log_y, 1)
        linear_coeffs[name] = linear_fit
        raw_log_linear[name] = np.polyval(linear_fit, extended_x_norm)
    
    # 3. Reusable Softmax transformation function to prevent code duplication
    def apply_softmax(raw_predictions):
        all_preds = np.array(list(raw_predictions.values()))
        max_preds = np.max(all_preds, axis=0) # Prevent exponent overflow
        exp_preds = {name: np.exp(pred - max_preds) for name, pred in raw_predictions.items()}
        total_exp = np.sum(list(exp_preds.values()), axis=0)
        return {name: exp_preds[name] / total_exp for name in raw_predictions}
        
    final_poly = apply_softmax(raw_log_poly)
    final_linear = apply_softmax(raw_log_linear)
    
    # 4. Plot Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    for name, y in zip(variant_cols.keys(), ploty):
        # Scatter actual data on both plots (using alpha to keep it visually clean)
        ax1.scatter(xval, y, s=12, alpha=0.4)
        ax2.scatter(xval, y, s=12, alpha=0.4)
        
        # Plot models
        ax1.plot(extended_x_dates, final_poly[name], linestyle='-', label=f"{name}")
        ax2.plot(extended_x_dates, final_linear[name], linestyle='-', label=f"{name}")

    # Format the Visuals
    ax1.set_title("Multinomial Log-Quadratic Fit (Polynomial)")
    ax2.set_title("Multinomial Log-Linear Fit")
    
    for ax in (ax1, ax2):
        ax.set_ylim(bottom=0, top=1)
        ax.tick_params(axis='x', rotation=45)
        
    # Single legend outside the second plot
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    
    plt.tight_layout()
    plt.show()

    # 5. Output the separate coefficient lists for you to work with
    print("\n" + "="*40)
    print("POLYNOMIAL (QUADRATIC) COEFFICIENTS [a, b, c]")
    print("="*40)
    for variant, coeffs in polynomial_coeffs.items():
        print(f"{variant:<15}: {coeffs}")
        
    print("\n" + "="*40)
    print("LINEAR COEFFICIENTS [m, b]")
    print("="*40)
    for variant, coeffs in linear_coeffs.items():
        print(f"{variant:<15}: {coeffs}")

    # 6. Evaluate Models using Akaike Information Criterion (AIC)
    print("\n" + "="*40)
    print("MODEL EVALUATION (AIC)")
    print("="*40)
    
    # Total data points = number of timepoints * number of variants
    n_points = len(xval) * len(variant_cols)
    
    # Number of parameters (k): Linear has 2 per variant, Poly has 3 per variant
    k_linear = 2 * len(variant_cols)
    k_poly = 3 * len(variant_cols)
    
    rss_linear = 0
    rss_poly = 0
    
    for name, y_actual in zip(variant_cols.keys(), ploty):
        # Slice the extended predictions back to just the historical timeframe to match y_actual
        y_pred_linear = final_linear[name][:len(xval)]
        y_pred_poly = final_poly[name][:len(xval)]
        
        # Add to total Residual Sum of Squares
        rss_linear += np.sum((y_actual - y_pred_linear)**2)
        rss_poly += np.sum((y_actual - y_pred_poly)**2)
        
    # AIC Formula using RSS: n * ln(RSS/n) + 2k
    aic_linear = n_points * np.log(rss_linear / n_points) + 2 * k_linear
    aic_poly = n_points * np.log(rss_poly / n_points) + 2 * k_poly
    
    print(f"Linear Model AIC:     {aic_linear:.2f}")
    print(f"Polynomial Model AIC: {aic_poly:.2f}")

#-----------Relative Reproduction based model------#



#--------------------Data OUTPUT-------------------#