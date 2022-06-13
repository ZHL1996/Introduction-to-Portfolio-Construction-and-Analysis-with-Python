import pandas as pd

def drawdown(ret_series : pd.Series):
    a=1000*(1+ret_series).cumprod()
    b= (a-a.cummax())/a.cummax()
    return pd.DataFrame({"wealth":a,
                        "peaks":a.cummax(),
                        "drawdown":b})
def compound(x):
    result = (1+x).cum_prod()
    return(result)

def get_ffme_returns():
    returns = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",header = 0, index_col = 0, parse_dates = True,
                     na_values = -99.99)
    returns = returns[["Lo 10","Hi 10"]]
    returns = returns/100
    returns.columns = ["Small cap","Large Cap"]
    returns.index = pd.to_datetime(returns.index,format="%Y%m").to_period("M")
    return returns

def get_hfi_returns():
    hfi_returns = pd.read_csv("data/edhec-hedgefundindices.csv",header = 0, index_col = 0, parse_dates = True)
    hfi_returns = hfi_returns/100
    hfi_returns.index = hfi_returns.index.to_period('M')
    return hfi_returns

def skewness(r):
    devi = r-r.mean()
    std = r.std(ddof=0)
    x = (devi**3).mean()
    return x/std**3

def kurtosis(r):
    devi = r - r.mean()
    devi_f = devi**4
    edevi = devi_f.mean()
    estd = r.std()**4
    return edevi/estd

import scipy.stats
def is_normal(r , level = 0.01):
    s, p = scipy.stats.jarque_bera(r)
    return p>level

def semideviation(r):
    a =r[r<0].std(ddof=0)
    return a

import numpy as np
def var_historical(r,level=5):
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_historical)
    elif isinstance(r,pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError("r expected to be Series or DataFrame")

from scipy.stats import norm
import scipy.stats as ss
def var_gaussian(r,level=5,modified = False):
    z = norm.ppf(level/100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z + (z**2 -1)*s/6 + (z**3 - 3*z)*(k-3)/24-(2*z**3-5*z)*(s**2)/36)
    return -(r.mean()+z*r.std(ddof=0))
 
def cvar_historical(r,level = 5):
    if isinstance(r,pd.Series):
        is_beyond = r<= -var_historical(r,level)
        return r[is_beyond].mean()
    elif isinstance(r,pd.DataFrame):
        return -r.aggregate(cvar_historical,level = level)
    else:
        raise TypeError("wrong")
    
import pandas as pd
def get_ind_nfirms():
    ind = pd.read_csv("data/ind30_m_nfirms.csv",parse_dates = True, header = 0, index_col = 0)
    ind.index = pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind

import pandas as pd
def get_ind_returns():
    ind = pd.read_csv("data/ind30_m_vw_rets.csv",parse_dates = True, header = 0, index_col = 0)/100
    ind.index = pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind

import pandas as pd
def get_ind_size():
    ind = pd.read_csv("data/ind30_m_size.csv",parse_dates = True, header = 0, index_col = 0)
    ind.index = pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind

def an_ret(r,p):
    s = r.shape[0]
    an_rets = (1+r).prod()**(p/s)-1
    return an_rets 

import numpy as np
def an_vol(r,p):
    v = r.std()*np.sqrt(p)
    return v
        
def sharp_ratio(r,rf,p):
    sh = (an_ret(r,p)-rf)/an_vol(r,p)
    return sh

def portfolio_return(weights,returns):
    return weights.T @ returns


def portfolio_vol(weights,covmat):
    return (weights.T @ covmat @ weights)**(0.5)

import pandas as pd
import numpy as np
def plot_ef2(n_points, er, cov):
    if er.shape[0] !=2 or er.shape[0] !=2:
        raise ValueError("Just for 2 assets")
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    r = [portfolio_return(w,er) for w in weights]
    co = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({"return":r,"vol":co})
    ef.plot(x="vol",y="return",style="*-")

    
from scipy.optimize import minimize
import numpy as np
def target_is_met(w,er):
    return target_return - erk.portfolio_return(w,er)
def minimize_vol(target_return,er,cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    return_is_target={'type':'eq',
                    'args':(er,),
                    'fun':lambda weights,er: target_return - erk.portfolio_return(weights,er)}
    weights_sum_to_1={'type':'eq',
                     'fun':lambda weights: np.sum(weights) - 1}
    results = minimize(erk.portfolio_vol,
                       init_guess,
                       args=(cov,),
                       method = "SLSQP",
                       options={"disp":False},
                       constraints = (return_is_target,weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

    
from scipy.optimize import minimize
import numpy as np
def msr(rf,er,cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    weights_sum_to_1={'type':'eq',
                     'fun':lambda weights: np.sum(weights) - 1}
    
    def neg_sharp_ratio(weights,rf,er,cov):
        r = portfolio_return(weights,er)
        vol = portfolio_vol(weights,cov)
        sh = -(r-rf)/vol
        return sh
    
    results = minimize(neg_sharp_ratio,
                       init_guess,
                       args=(rf,er,cov,),
                       method = "SLSQP",
                       options={"disp":False},
                       constraints = (weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

def optimal_weights(n_points,er,cov):
    target_rs = np.linspace(er.min(),er.max(),n_points)
    weights = [minimize_vol(t,er,cov) for t in target_rs]
    return weights

import pandas as pd
import numpy as np
def gmv(cov):
    n = cov.shape[0]
    return msr(0,np.repeat(1,n),cov)
def plot_ef(n_points, er, cov,show_cml=False,rf=0,show_ew=False,show_gmv = False):
    weights = optimal_weights(n_points,er,cov)
    r = [portfolio_return(w,er) for w in weights]
    co = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({"return":r,"vol":co})
    ax = ef.plot(x="vol",y="return",style=".-")
    ax.plot()
    if show_ew:
        n = er.shape[0]
        ew = np.repeat(1/n,n)
        r_ew = portfolio_return(r_er,er)
        vol_ew = portfolio_vol(ew,cov)
        ax.plot([vol_ew],[r_ew],color="goldenrod",marker = "o",markersize=12)
    if show_gmv:
        w_gmv= gmv(cov)
        r_gmv = portfolio_return(w_gmv,er)
        vol_gmv = portfolio_vol(w_gmv,cov)
        ax.plot([vol_gmv],[r_gmv],color="goldenrod",marker = "o",markersize=12)      
    if show_cml:
        rf=0.1
        w_msr = msr(rf,er,cov)
        r_msr=erk.portfolio_return(w_msr,er)
        vol_msr = erk.portfolio_vol(w_msr,cov)
        cml_x = (0,vol_msr)
        cml_y = (rf,r_msr)
        ax.plot(cml_x,cml_y,color="green",marker="*",markersize=12,linewidth=2,linestyle="dashed")

def run_cppi(risky_r,safe_r=None, start=1000,floor=0.8,risk_free_rate=0.03, drawdown = None):
    dates = risky_r.index 
    n_steps = len(dates)
    account_value = start
    m = 3
    floor_value = floor * start
    
    peak=start
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])    
    if safe_r is None:
        safe_r=pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] =  risk_free_rate/12     
    account_history = pd.DataFrame().reindex_like(risky_r)
    cusion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r) 
    for step in range(n_steps):
        if drawdown is not None:
            peak=np.maximum(peak,account_value)
            floor_value = (1-drawdown)*peak
            
        cusion = (account_value - floor_value)/ account_value
        risk_w = m*cusion
        risk_w = np.minimum(risk_w,1)
        risk_w = np.maximum (risk_w,0)
        safe_w = 1-risk_w
        risky_alloc = account_value*risk_w
        safe_alloc = account_value * safe_w
        # update account value for time step
        account_value = risky_alloc*(1+risky_r.iloc[step])+safe_alloc*(1+safe_r.iloc[step])
        # store the result
        cusion_history.iloc[step] = cusion
        risky_w_history.iloc[step] = risk_w
        account_history.iloc[step] = account_value
        risky_wealth = start * (1+risky_r).cumprod()
        brack = {'wealth':account_history,'risky_wealth':risky_wealth,'risk_budget':cusion_history,'risk_allocation':risky_w_history,
            'm':m,'start':start,'floor':floor,'risky_r':risky_r,'safe_r':safe_r}
    return brack

def summary_stats(r,risk_free_rate = 0.03):
    ann_r = r.aggregate(an_ret,p=12)
    ann_vol=r.aggregate(an_vol,p=12)
    ann_sr = r.aggregate(sharp_ratio,rf=risk_free_rate, p=12)
    dd = r.aggregate(lambda r:drawdown(r).drawdown.min())
    skew = r.aggregate(skewness)
    kur = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian,modified = True)
    hist_var5 = r.aggregate(cvar_historical)
    return pd.DataFrame({'Annualized ret':ann_r,'Annualized vol':ann_vol,'Annualized sharp ratio':ann_sr,'Skewness':skew,'Kortosis':kur,'Historical VaR(5%)':hist_var5,'Cornish_Fisher VaR(5%)':cf_var5,'Max Drawdown':dd})

def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val

def show_gbm(n_scenarios,mu,sigma):
    s_0=100
    prices = gbm(n_scenarios=n_scenarios,mu=mu,sigma=sigma,s_0=s_0)
    ax = prices.plot(legend=False,figsize=(12,5),linewidth=2,color="indianred")
    ax.axhline(y=s_0,ls=":",color="black")
    #draw a dot at origin
    ax.plot(0,s_0,marker="o",color = "darkred")
    
def inst_to_ann(r):
    return np.expm1(r)

def ann_to_inst(r):
    return log1p(r)
    
def discount(t,r):
    discount = pd.DataFrame([(1+r)**-i for i in t])
    discount.index = t
    return discount
    
def pv(l,r):
    dates = l.index
    discounts=discount(dates,r)
    return discounts.multiply(l,axis="rows").sum()

import math
def cir(n_years=10,n_scenarios=1,a=0.05,b=0.03,sigma=.05,steps_per_year=12,r_0=None):
    if r_0 is None: r_0 = b
    r_0=inst_to_ann(r_0)
    dt=1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1
    
    shock = np.random.normal(0,scale = np.sqrt(dt),size=(num_steps,n_scenarios))
    rates = np.empty_like(shock)
    rates[0]=r_0
    
    # price generalization
    h=math.sqrt(a**2+2*sigma**2)
    prices = np.empty_like(shock)
    
    def price(ttm,r):
        _A=((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h+(h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0]=price(n_years,r_0)
    
    for step in range(1,num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt+sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs( r_t+d_r_t)
        prices[step]=price(n_years-step*dt,rates[step])
    rates = pd.DataFrame(data = inst_to_ann(rates),index = range(num_steps)) 
    prices = pd.DataFrame(data=prices,index = range(num_steps))
    return rates, prices




def show_cir_prices(n_scenarios=5,a=0.05,b=0.03,sigma=.05,r_0=.03):
    cir(n_scenarios=n_scenarios,a=a,b=b,sigma=sigma,r_0=r_0)[1].plot(legend=False,figsize=(12,6))      

#import ipywidgets as widgets
#from IPython .display import display
#control = widgets.interactive(show_cir_prices,n_scenarios = (1,100),a=(0,1,0.1),b=(0,1,.01),r_0=(0,0.5,0.01),sigma=(0,0.5,.01))
#display(control)




def bond_cash_flow(maturity,principal=1000,coupon_rate=.03,coupons_per_year=12):
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupon_times = np.arange(1,n_coupons+1)
    cash_flows=pd.Series(data=coupon_amt,index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows



def macaulay_duration(flows,discount_rate):
    discount_flows = discount(flows.index,discount_rate)*flows
    weights = discount_flows/discount_flows.sum()
    #macaulay_duration = weights*flows.index
    macaulay_duration = np.average(flows.index, weights=weights)
    return macaulay_duration

def match_durations(cf_t,cf_s,cf_l,discount_rate):
    d_t = macaulay_duration(cf_t,discount_rate)
    d_s = macaulay_duration(cf_s,discount_rate)
    d_l = macaulay_duration(cf_l,discount_rate)
    return (d_l-d_t)/(d_l-d_s)


def bond_price(maturity,principal=1000,coupon_rate=.03,coupons_per_year=12,discount_rate=.03):
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates,columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t]=bond_price(maturity-t/coupons_per_year,principal,coupon_rate,coupons_per_year,discount_rate.loc[t])
        return prices
    else:
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flow(maturity,principal,coupon_rate,coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)

def funding_ratio(assets,l,r):
    return pv(assets,r)/pv(l,r)

def bond_total_return(monthly_prices,principal,coupon_rate,coupons_per_year):
    coupons = pd.DataFrame(data=0,index = monthly_prices.index, columns = monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year,t_max,int(coupons_per_year*t_max/12),dtype=int)
    coupons.iloc[pay_date]=principal * coupon_rate / coupons_per_year
    total_returns = (monthly_prices+coupons)/monthly_prices.shift()-1
    return total_returns .dropna()

def bt_mix(r1,r2,allocator,**keywargs):
    if not r1.shape==r2.shape:
        raise ValueError("r1 and r2 should be in the same shape")
    weights=allocator(r1,r2,**keywargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator's weights don't match r1")
    r_mix = r1*weights+(1-weights)*r2
    return r_mix

def fixedmix_allocator(r1,r2,w1,**kwargs):
    return pd.DataFrame(data = w1, index = r2.index, columns = r1.columns)
    
    
def terminal_values(rets):
    return (rets+1).prod()

def terminal_stats(rets,floor=.8,cap = np.inf,name="Stats"):
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth<floor
    reach = terminal_wealth>=cap
    p_breach = breach.mean() if breach.sum()>0 else np.nan
    p_reach = reach.mean() if reach.sum()>0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum()>0 else np.nan
    e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum()>0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
                   "mean" : terminal_wealth.mean(),
                   "std":terminal_wealth.std(),
                   "p_breach":p_breach,
                   "e_short":e_short,
                   "p_reach":p_reach,
                   "e_surplus":e_surplus
                   },orient="index",columns=[name])
    return sum_stats
                   

def glidepath_allocator(r1,r2,start_glide=1,end_glide=0):
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide,end_glide,num=n_points))
    paths = pd.concat([path]*n_col,axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths
        
 
    