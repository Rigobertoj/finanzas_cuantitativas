import pandas as pd
import numpy as np

import numpy_financial as npf


class financial_metrics:
    
    def calculate_npv(self, data, discount_rate):
        npv = npf.npv(discount_rate, data)
        return npv
    
    def calculate_irr(self, data):
        irr = npf.irr(data)
        return irr
    

def valuate_investment(data : pd.DataFrame, initial_investment : float, discount_rate : float) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        initial_investment (float): _description_
        discount_rate (float): _description_

    Returns:
        pd.DataFrame: _description_
    """
    financial_data = pd.DataFrame()
    
    initial_investment = pd.Series([initial_investment])
    
    fm = financial_metrics()
    
    for index, row in data.iterrows():
        cash_flows = row.values
        
        net_cash_flows = pd.concat([initial_investment, row]).values        

        npv = fm.calculate_npv(cash_flows, discount_rate)
        irr = fm.calculate_irr(net_cash_flows)
        
        financial_data.at[index, 'NPV'] = npv
        financial_data.at[index, 'IRR'] = irr
        
    return financial_data