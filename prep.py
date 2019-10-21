import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def tax_data_clean(df):
    df['county']=df['fips']
    df['county'] = np.where(df['fips']== 6037,'Los Angles',(np.where(df['fips']== 6059,'Orange',(np.where(df['fips']==6111,'Ventura',"")))))
    return df

#Tax distribution by county.

def tax_dist(df):
    from matplotlib import pyplot as plt
    g = sns.FacetGrid(df, col = "county")
    g.map(plt.hist, "tax_rate")
    plt.xlim(0,.1)
    plt.xticks(np.arange(0, .07, step=0.01))
    return plt.show()

def base_clean(df):
    df = df.filter(['bedrooms','bathrooms','sqft', 'tax_value'], axis=1)
    return df