#%%
from scipy import stats
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pingouin import friedman
import scikit_posthocs as sp

import Settings as settings

def scientific_to_decimal(value, n):
    return format(value, '.'+str(n)+'f')


# Shapiro
# %%
dfPowers = pd.read_pickle('./pickles/dfPowers.pickle')


for y in range(len(settings.channel_groups)):
    for i in range(len(settings.methods)):
        print("ALPHA POWER Group " , y , " method ", settings.methods[i])
        x = dfPowers.loc[(dfPowers['Group'] == y) & (dfPowers['Method'] == settings.methods[i])]
        shapiro_test = stats.shapiro(x['Alpha'])
        print("statistic: ", shapiro_test.statistic)
        print("P value: ", scientific_to_decimal(shapiro_test.pvalue, 30))
        
for y in range(len(settings.channel_groups)):
    for i in range(len(settings.methods)):
        print("THETA POWER Group " , y , " method ", settings.methods[i])
        x = dfPowers.loc[(dfPowers['Group'] == y) & (dfPowers['Method'] == settings.methods[i])]
        shapiro_test = stats.shapiro(x['Theta'])
        print("statistic: ", shapiro_test.statistic)
        print("P value: ", scientific_to_decimal(shapiro_test.pvalue, 30))
        
for y in range(len(settings.channel_groups)):
    for i in range(len(settings.methods)):
        print("ALPHA THETA Group " , y , " method ", settings.methods[i])
        x = dfPowers.loc[(dfPowers['Group'] == y) & (dfPowers['Method'] == settings.methods[i])]
        shapiro_test = stats.shapiro(x['Alpha/Theta'])
        print("statistic: ", shapiro_test.statistic)
        print("P value: ", scientific_to_decimal(shapiro_test.pvalue, 30))

# Friedman
# %%

dfPowers = pd.read_pickle('./pickles/dfPowers.pickle')

for y in range(len(settings.channel_groups)):
    for i in range(len(settings.methods)):
        print("ALPHA Group " , y , " method ", settings.methods[i])
        tt = dfPowers.loc[(dfPowers['Group'] == y) & (dfPowers['Method'] == settings.methods[i])]
        x1 = tt["Alpha"].loc[(tt['BlockNumber'] == 1)]
        x2 = tt["Alpha"].loc[(tt['BlockNumber'] == 2)]
        x3 = tt["AlphaPow"].loc[(tt['BlockNumber'] == 3)]
        x4 = tt["Alpha"].loc[(tt['BlockNumber'] == 4)]
        x5 = tt["Alpha"].loc[(tt['BlockNumber'] == 5)]
        x6 = tt["Alpha"].loc[(tt['BlockNumber'] == 6)]
        x7 = tt["Alpha"].loc[(tt['BlockNumber'] == 7)]
        
        data = np.array([x1.values, x2.values, x3.values, x4.values, x5.values])
        
        #Group1 = tt.loc[(dfPowers['BlockNumber'] == 1)]
        stat, p = friedmanchisquare(x1.values, x2.values, x3.values, x4.values, x5.values, x6.values, x7.values)
        print("SCIPI" , stat,p)
        
  
        ######################
        df = pd.DataFrame(data=np.column_stack((x1,x2,x3,x4,x5,x6,x7)),columns=['b1','b2', 'b3', 'b4', 'b5', 'b6', 'b7'])
        df.insert(len(df.columns), 'id',  range(1,len(df)+1))
        longDf = pd.melt(df, id_vars='id', var_name='item', value_name='score')
        pgRes = friedman(data=longDf, dv='score', within='item', subject='id')
        print("PENGOUIN", pgRes['Q'].at['Friedman'], pgRes['p-unc'].at['Friedman'])
        
    
        #data = np.array([x1.values, x2.values, x3.values, x4.values, x5.values])
        #result = sp.posthoc_conover_friedman(data)
        result = sp.posthoc_conover_friedman(a=longDf, y_col="score", group_col="item", block_col="id", 
                                 p_adjust="fdr_bh", melted=True)
        
        
        #print(result)
        
        al1 = tt["Alpha"].loc[(tt['BlockNumber'] == 1)].values.mean()
        te1 = tt["Theta"].loc[(tt['BlockNumber'] == 1)].values.mean()
        at1 = tt["Alpha/Theta"].loc[(tt['BlockNumber'] == 1)].values.mean()
        al2 = tt["Alpha"].loc[(tt['BlockNumber'] == 2)].values.mean()
        te2 = tt["Theta"].loc[(tt['BlockNumber'] == 3)].values.mean()
        at2 = tt["Alpha/Theta"].loc[(tt['BlockNumber'] == 2)].values.mean()
        al3 = tt["Alpha"].loc[(tt['BlockNumber'] == 3)].values.mean()
        te3 = tt["Theta"].loc[(tt['BlockNumber'] == 3)].values.mean()
        at3 = tt["Alpha/Theta"].loc[(tt['BlockNumber'] == 3)].values.mean()
        al4 = tt["Alpha"].loc[(tt['BlockNumber'] == 4)].values.mean()
        te4 = tt["Theta"].loc[(tt['BlockNumber'] == 4)].values.mean()
        at4 = tt["Alpha/Theta"].loc[(tt['BlockNumber'] == 4)].values.mean()
        al5 = tt["Alpha"].loc[(tt['BlockNumber'] == 5)].values.mean()
        te5 = tt["Theta"].loc[(tt['BlockNumber'] == 5)].values.mean()
        at5 = tt["Alpha/Theta"].loc[(tt['BlockNumber'] == 5)].values.mean()
        al6 = tt["Alpha"].loc[(tt['BlockNumber'] == 6)].values.mean()
        te6 = tt["Theta"].loc[(tt['BlockNumber'] == 6)].values.mean()
        at6 = tt["Alpha/Theta"].loc[(tt['BlockNumber'] == 6)].values.mean()
        al7 = tt["Alpha"].loc[(tt['BlockNumber'] == 7)].values.mean()
        te7 = tt["Theta"].loc[(tt['BlockNumber'] == 7)].values.mean()
        at7 = tt["Alpha/Theta"].loc[(tt['BlockNumber'] == 7)].values.mean()
        data2 = np.array([[al1, al2, al3,al4, al4, al6,al7],[te1,te2, te3,te4,te5, te6,te7],[at1, at2, at3, at4, at5, at6, at7]])
        stat1, p1 = friedmanchisquare(data2[0], data2[1], data2[2])
        print("SECOND TEST", stat1, p1)

        
