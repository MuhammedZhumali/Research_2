"""Finite-sample paired inference for the prespecified confirmatory family.

Empirical Bernstein: Maurer & Pontil (COLT 2009), Theorems 4 and 11.
The one-sided failure probability is delta/2 for each tail, hence log(4/delta).
Risk differences use two exact Clopper-Pearson discordance intervals and a
union bound; no independence between the discordance categories is assumed.
"""
import math
import numpy as np

def empirical_bernstein(values, *, lower_bound, upper_bound, alpha):
    data=np.asarray(values,dtype=float)
    if len(data)<2 or not 0<alpha<1 or not lower_bound<upper_bound:
        raise ValueError('Invalid interval inputs')
    if not np.all(np.isfinite(data)) or np.any(data<lower_bound-1e-8) or np.any(data>upper_bound+1e-8):
        raise ValueError('Observation outside preregistered support')
    n=len(data);mean=float(data.mean());variance=float(data.var(ddof=1))
    width=upper_bound-lower_bound
    log=math.log(4/alpha)
    half=math.sqrt(2*variance*log/n)+7*width*log/(3*(n-1))
    return dict(n=n,mean_difference=mean,ci_low=max(lower_bound,mean-half),ci_high=min(upper_bound,mean+half),
                unbiased_variance=variance,half_width=half,range_width=width)

def binomial_cdf(k,n,p):
    if k<0:return 0.0
    if k>=n:return 1.0
    if p<=0:return 1.0
    if p>=1:return 0.0
    lp,lq=math.log(p),math.log1p(-p)
    base=math.lgamma(n+1)
    terms=[base-math.lgamma(i+1)-math.lgamma(n-i+1)+i*lp+(n-i)*lq for i in range(k+1)]
    peak=max(terms)
    return min(1.0,math.exp(peak)*math.fsum(math.exp(x-peak) for x in terms))

def _invert_cdf(k,n,target):
    lo,hi=0.0,1.0
    for _ in range(70):
        mid=(lo+hi)/2
        if binomial_cdf(k,n,mid)>target:lo=mid
        else:hi=mid
    return (lo+hi)/2

def clopper_pearson(k,n,alpha=.05):
    if not isinstance(k,(int,np.integer)) or not isinstance(n,(int,np.integer)) or not 0<=k<=n or n<=0 or not 0<alpha<1:
        raise ValueError('Invalid binomial interval inputs')
    tail=alpha/2
    lower=0.0 if k==0 else tail**(1/n) if k==n else _invert_cdf(k-1,n,1-tail)
    upper=1.0 if k==n else -math.expm1(math.log(tail)/n) if k==0 else _invert_cdf(k,n,tail)
    return lower,upper

def paired_binary_difference(candidate,reference,*,alpha):
    a=np.asarray(candidate);b=np.asarray(reference)
    if len(a)!=len(b) or not len(a) or not np.all(np.isin(a,[0,1])) or not np.all(np.isin(b,[0,1])):
        raise ValueError('Paired binary observations required')
    n=len(a);n10=int(np.sum((a==1)&(b==0)));n01=int(np.sum((a==0)&(b==1)))
    n11=int(np.sum((a==1)&(b==1)));n00=n-n10-n01-n11
    l10,u10=clopper_pearson(n10,n,alpha/2)
    l01,u01=clopper_pearson(n01,n,alpha/2)
    return dict(n=n,n10=n10,n01=n01,n11=n11,n00=n00,
                mean_difference=(n10-n01)/n,ci_low=l10-u01,ci_high=u10-l01,
                p10_low=l10,p10_high=u10,p01_low=l01,p01_high=u01,
                marginal_alpha=alpha/2,tail_alpha=alpha/4)

def infer_status(intervals):
    """Two-sided equivalence and one-sided noninferiority are separate claims."""
    ni=all(r['ci_high']<r['margin'] for r in intervals)
    eq=all(r['ci_low']>-r['margin'] and r['ci_high']<r['margin'] for r in intervals)
    ni_bad=any(r['ci_low']>r['margin'] for r in intervals)
    eq_bad=any(r['ci_low']>r['margin'] or r['ci_high']<-r['margin'] for r in intervals)
    return dict(ni_status='supported' if ni else 'inferior' if ni_bad else 'inconclusive',
                eq_status='supported' if eq else 'not_equivalent' if eq_bad else 'inconclusive')
