# Relative Reproduction number for viral subject to baseline variants

#-----------------Package Setup--------------------*
import csv
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from scipy.optimize import curve_fit # Log regression
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
plotregressquad = False
regression_models = []

# This is a mini program for a quadratic regression written by AI
if plotregressquad:
    # xval already contains datetime.date objects, convert to ordinals for the math
    x_ord = np.array([d.toordinal() for d in xval])
    
    for name, y in zip(variant_cols.keys(), ploty):
        
        # 1. Plot original data using native datetime objects
        plt.scatter(xval, y, s=12, label=f"{name} (Data)")
        
        if len(x_ord) > 3:
            # 2. Fit Chebyshev polynomial using the class method.
            # This automatically maps the large ordinal dates to a [-1, 1] window,
            # which prevents numerical instability/crashing during extrapolation.
            cheb_model = np.polynomial.Chebyshev.fit(x_ord, y, 3)
            
            # Store the variant name and its corresponding model into our list
            regression_models.append({'variant': name, 'model': cheb_model})
            
            # 3. Create extended timeline (historical + 182 days)
            last_date = xval[-1]
            future_dates = [last_date + dt.timedelta(days=i) for i in range(1, 183)]
            extended_x_dates = xval + future_dates
            extended_x_ord = np.array([d.toordinal() for d in extended_x_dates])
            
            # 4. Evaluate the model on the extended dates
            extended_y = cheb_model(extended_x_ord)
            
            # 5. Plot the line using the SAME datetime object type as the scatter plot
            plt.plot(extended_x_dates, extended_y, linestyle='--', label=f"{name} (Extrap.)")

    # Move legend outside the plot so it doesn't cover the 6-month extrapolation
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.ylim(bottom=0, top=1)
    plt.tight_layout() # Ensures the rotated dates and legend fit in the window
    plt.show()

plotregresslog=False
if plotregresslog:
    # Convert dates to ordinals
    x_ord = np.array([d.toordinal() for d in xval])
    
    # Normalize X starting at 0 so the exponent math doesn't overflow
    x_min = x_ord.min()
    x_norm = x_ord - x_min 
    
    # Define the Logistic model function
    def logistic_model(x, L, k, x0):
        """
        L: Maximum value (bounded to 1 max)
        k: Growth rate (steepness)
        x0: The x-value of the curve's midpoint
        """
        # Clip x to prevent math overflow errors on large exponent calculations
        x_safe = np.clip(x - x0, -500, 500)
        return L / (1.0 + np.exp(-k * x_safe))

    for name, y in zip(variant_cols.keys(), ploty):
        
        plt.scatter(xval, y, s=12, label=f"{name} (Data)")
        
        if len(x_norm) > 3:
            try:
                # p0 is our initial guess: max y, slight slope, median date
                p0 = [max(y), 0.1, np.median(x_norm)]
                
                # Bounds ensure the curve's maximum (L) never exceeds 1 (100%)
                bounds = ([0, -np.inf, -np.inf], [1.0, np.inf, np.inf])
                
                # Fit the logistic curve to the data
                popt, _ = curve_fit(logistic_model, x_norm, y, p0=p0, bounds=bounds)
                
                # Store the variant name and its fitted parameters (L, k, x0)
                regression_models.append({'variant': name, 'params': popt})
                
                # Create extended timeline (historical + 182 days)
                last_date = xval[-1]
                future_dates = [last_date + dt.timedelta(days=i) for i in range(1, 183)]
                extended_x_dates = xval + future_dates
                
                # Convert future dates to the same normalized X scale
                extended_x_norm = np.array([d.toordinal() - x_min for d in extended_x_dates])
                
                # Evaluate the model on the extended dates
                extended_y = logistic_model(extended_x_norm, *popt)
                
                # Plot the line
                plt.plot(extended_x_dates, extended_y, linestyle='--', label=f"{name} (Logistic Extrap.)")
                
            except RuntimeError:
                print(f"Could not find a stable logistic fit for {name}. It may lack enough data points.")

    # Move legend outside the plot
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xticks(rotation=45)
    
    # Lock the Y-axis between 0 and 1
    plt.ylim(bottom=0, top=1) 
    
    plt.tight_layout()
    plt.show()

plotregressgaus=False
if plotregressgaus:
    # Convert dates to ordinals
    x_ord = np.array([d.toordinal() for d in xval])
    
    # Normalize X starting at 0 to keep the math stable
    x_min = x_ord.min()
    x_norm = x_ord - x_min 
    
    # Define the Gaussian (Bell Curve) model
    def gaussian_model(x, A, mu, sigma):
        """
        A: Peak height (maximum proportion, bounded to 1)
        mu: The x-value (date) where the peak occurs
        sigma: The width of the wave (how fast it rises and falls)
        """
        # Prevent division by zero if sigma gets too small
        sigma = np.maximum(sigma, 1e-5)
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    for name, y in zip(variant_cols.keys(), ploty):
        
        plt.scatter(xval, y, s=12, label=f"{name} (Data)")
        
        if len(x_norm) > 3 and max(y) > 0:
            try:
                # Initial guesses:
                # A: The highest percentage seen so far
                # mu: The date that highest percentage occurred
                # sigma: Assume a default wave width of about 30 days
                guess_A = max(y)
                guess_mu = x_norm[np.argmax(y)]
                guess_sigma = 30.0
                
                p0 = [guess_A, guess_mu, guess_sigma]
                
                # Bounds: 
                # A must be between 0 and 1 (0% to 100%)
                # mu can be anywhere (past or future)
                # sigma must be positive
                bounds = ([0.0, -np.inf, 0.1], [1.0, np.inf, np.inf])
                
                # Fit the Gaussian curve to the data
                popt, _ = curve_fit(gaussian_model, x_norm, y, p0=p0, bounds=bounds, maxfev=5000)
                
                # Store the variant name and its fitted parameters (A, mu, sigma)
                regression_models.append({'variant': name, 'params': popt})
                
                # Create extended timeline (historical + 182 days)
                last_date = xval[-1]
                future_dates = [last_date + dt.timedelta(days=i) for i in range(1, 183)]
                extended_x_dates = xval + future_dates
                
                # Convert future dates to the same normalized X scale
                extended_x_norm = np.array([d.toordinal() - x_min for d in extended_x_dates])
                
                # Evaluate the model on the extended dates
                extended_y = gaussian_model(extended_x_norm, *popt)
                
                # Plot the line
                plt.plot(extended_x_dates, extended_y, linestyle='--', label=f"{name} (Gaussian)")
                
            except RuntimeError:
                print(f"Could not find a stable Gaussian fit for {name}. It may lack a clear curve.")

    # Move legend outside the plot
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xticks(rotation=45)
    
    # Lock the Y-axis between 0 and 1
    plt.ylim(bottom=0, top=1) 
    
    plt.tight_layout()
    plt.show()

plotmultinomial=True
days=183
if plotmultinomial:
    # Convert dates to ordinals and normalize
    x_ord = np.array([d.toordinal() for d in xval])
    x_min = x_ord.min()
    x_norm = x_ord - x_min
    
    # 1. Create the extended timeline (historical + days)
    last_date = xval[-1]
    future_dates = [last_date + dt.timedelta(days=i) for i in range(1, days)]
    extended_x_dates = xval + future_dates
    extended_x_norm = np.array([d.toordinal() - x_min for d in extended_x_dates])
    
    # We will temporarily store the unnormalized log-predictions for every variant here
    raw_log_predictions = {}
    
    # 2. Fit a quadratic regression to the log-space of each variant
    for name, y in zip(variant_cols.keys(), ploty):
        # Use a small constant (1e-4) to safely handle 0% values without math errors
        log_y = np.log(y + 1e-4)
        
        # Fit a simple quadratic (degree 2) polynomial
        coeffs = np.polyfit(x_norm, log_y, 2)
        
        # Save the coefficients for your list
        regression_models.append({'variant': name, 'coefficients': coeffs})
        
        # Generate raw predictions across the entire extended timeline
        raw_log_predictions[name] = np.polyval(coeffs, extended_x_norm)
    
    # 3. Apply the Softmax transformation to make variants compete and sum to 1.0
    # First, find the maximum raw value at each time step to prevent exponent overflow
    all_preds = np.array(list(raw_log_predictions.values()))
    max_preds = np.max(all_preds, axis=0)
    
    # Calculate exponents
    exp_predictions = {name: np.exp(pred - max_preds) for name, pred in raw_log_predictions.items()}
    total_exp_per_day = np.sum(list(exp_predictions.values()), axis=0)
    
    # 4. Plot the results
    for name, y in zip(variant_cols.keys(), ploty):
        # Plot original historical scatter data
        plt.scatter(xval, y, s=12, label=f"{name} (Data)")
        
        # Calculate the final normalized competitive proportion
        final_extrapolated_y = exp_predictions[name] / total_exp_per_day
        
        # Plot smoothed competitive curve
        plt.plot(extended_x_dates, final_extrapolated_y, linestyle='--', label=f"{name} (Model)")

    # Move legend outside the plot
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xticks(rotation=45)
    
    # Strict Y boundaries since percentages can only live between 0% and 100%
    plt.ylim(bottom=0, top=1) 
    
    plt.tight_layout()
    plt.show()

# Example of how you can access the saved models later:
# for item in regression_models:
#     print(f"Model for {item['variant']}: {item['model']}")


#-----------Relative Reproduction based model------#


#--------------------Data OUTPUT-------------------#