# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 17:14:17 2022

@author: zaratejo
"""

from fitter import Fitter ,get_common_distributions, get_distributions

import pandas as pd


RawData = pd.read_csv('Data_OilCompany.csv')
#RawData = pd.read_csv('Data_OilCompany10records.csv')
#RawData = pd.read_csv('PriceHistory.csv')

#data=RawData

#Index(['  Year 1  ', '  Year 2  ', '  Year 3  ', '  Year 4  ', '  Year 5  '], dtype='object')

#Test Year 2
data=RawData.iloc[:,0]
#data=RawData.iloc[:-1,4]
#data=float(data.replace("-",0))

#data=pd.DataFrame(b[:,7])



dist= ['gamma','lognorm', "beta", "burr", "norm"]
dist= ['uniform', "norm"]

dist=get_common_distributions()  #get_distributions()


f = Fitter(data,distributions=dist)

f.fit()

f.summary()
#f.summary(f, Nbest=5, lw=2, plot=True, method="sumsquare_error", clf=True)

print(f.get_best())