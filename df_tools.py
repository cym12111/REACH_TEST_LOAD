import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# define Python user-defined exceptions
class FitNotFound(Exception):
    "Raised when the input value is less than 18"
    pass


def nominal_vals_mask(df, varname, nominal_vals, tolerance=0.01):
    """ Function to obtain a mask for a DataFrame object that classifies
    dataframe indeces into nominal values of a selected variable.
    """
    nominal_names =[]

    for item in nominal_vals:
        nominal_names.append(varname+' = %2.2f' % (item))

    mask_df = pd.DataFrame(columns=nominal_names)

    for item in nominal_names:
        mask_df[item] = df[varname]

    for item, val in zip(nominal_names,nominal_vals):
        mask_df[item]=mask_df[item]-val
        mask_df[item]=mask_df[item].abs()
        mask_df[item]=np.where(mask_df[item]<=tolerance, mask_df[item], np.nan)

    nominal_idxmin = mask_df.iloc[:,:].idxmin(1)

    for item in nominal_names:
        mask_df[item]=nominal_idxmin==item

    return mask_df

def combine_masks(mask_df1, mask_df2):
    """ Function to combine masks for Dataframes.
    """

    combined_mask_df = pd.DataFrame()

    for name1 in mask_df1.columns:
        for name2 in mask_df2.columns:
            combined_name = name1+'<br />'+name2
            combined_mask_df[combined_name]=mask_df1[name1]*mask_df2[name2]
    
    return combined_mask_df

def line_fit(x, y):
    """ Least squares line between two columns in a DataFrame.
    """

    def func(x, a, b):
        return a * x + b
    
    try:
        popt,_ = curve_fit(func, x, y)
        xfit = x.sort_values()
        yfit = func(xfit, *popt) # the asterix expands the variables in popt
        return xfit, yfit
    except:
        pass

    

def exponential_fit(x, y):
    """ Least squares exponential between two columns in a DataFrame.
    """

    def func(x, a, b, c):
        return -a * np.exp(-b * x) + c
    
    try:
        popt,_ = curve_fit(func, x, y)
        xfit = x.sort_values()
        yfit = func(xfit, *popt) # the asterix expands the variables in popt
        return xfit, yfit
    except:
        pass
    
    