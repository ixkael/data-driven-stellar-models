import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import pandas as pd
pi=math.pi


#df = pd.read_csv('../Data/magnetic_WDs.csv')
#print df.shape[0]
#print df.columns

df = pd.read_csv('./gaiadr2_maincuts_wds.csv')
print(df.shape[0])
print(df.columns)
df['absG'] = df['phot_g_mean_mag']-5.*(np.log10(1e3/df['parallax'])-1.)
df['vs'] = np.sqrt(df['pmra']**2.+df['pmdec']**2.)*4.74057/df['parallax']
df['BPmag'] = (-5./2.*np.log10(2./5.*df['phot_bp_mean_flux'])+24.356577)
df['RPmag'] = (-5./2.*np.log10(2./5.*df['phot_rp_mean_flux'])+23.767088)
df['umag'] = df['sdssdr9_u_mag']-5.*(np.log10(1e3/df['parallax'])-1.)
df['gmag'] = df['sdssdr9_g_mag']-5.*(np.log10(1e3/df['parallax'])-1.)
df['rmag'] = df['sdssdr9_r_mag']-5.*(np.log10(1e3/df['parallax'])-1.)
df['imag'] = df['sdssdr9_i_mag']-5.*(np.log10(1e3/df['parallax'])-1.)
df['zmag'] = df['sdssdr9_z_mag']-5.*(np.log10(1e3/df['parallax'])-1.)
#df['ugr'] = -0.4925*df['sdssdr9_u_mag']-0.5075*df['sdssdr9_g_mag']+df['sdssdr9_r_mag']
cleanAcut = (df['umag']>4.16) & (df['umag']<21.) & (df['gmag']>4.69) & (df['gmag']<19.4) & (df['rmag']>5.3) & (df['rmag']<18.6) & (df['imag']>5.6) & (df['imag']<18.4) & (df['zmag']>6.) & (df['zmag']<18.3)
#cleanAcut = np.isfinite(df['sdssdr9_u_mag']) & np.isfinite(df['sdssdr9_g_mag']) & np.isfinite(df['sdssdr9_r_mag']) & np.isfinite(df['sdssdr9_i_mag']) & np.isfinite(df['sdssdr9_z_mag'])
cleanBcut = (df['sdssdr9_u_mag_error']<0.2) & (df['sdssdr9_g_mag_error']<0.2) & (df['sdssdr9_r_mag_error']<0.2) & (df['sdssdr9_i_mag_error']<0.2) & (df['sdssdr9_z_mag_error']<0.2)
cleanCcut = (1./df['parallax']<0.2) & (df['parallax']/df['parallax_error']>10.)
cleanDcut = (df['visibility_periods_used']>8) & (df['astrometric_chi2_al']/(df['astrometric_n_good_obs_al']-5.)<1.44*(1.+(df['phot_g_mean_mag']<19.5)*(np.exp(-0.4*(df['phot_g_mean_mag']-19.5))-1.)))
cleancuts = cleanAcut & cleanBcut & cleanCcut & cleanDcut
print(np.shape(df[cleancuts]))

obsmags = np.transpose( [df[cleancuts]['umag'].values,df[cleancuts]['gmag'].values,df[cleancuts]['rmag'].values,df[cleancuts]['imag'].values,df[cleancuts]['zmag'].values] )
print(obsmags[0])
uvar = df[cleancuts]['sdssdr9_u_mag_error'].values**2
gvar = df[cleancuts]['sdssdr9_g_mag_error'].values**2
rvar = df[cleancuts]['sdssdr9_r_mag_error'].values**2
ivar = df[cleancuts]['sdssdr9_i_mag_error'].values**2
zvar = df[cleancuts]['sdssdr9_z_mag_error'].values**2
propparvar = (5./math.log(10.)*df[cleancuts]['parallax_error'].values/df[cleancuts]['parallax'].values)**2
obsmags_covar = [np.diag([uvar[i],gvar[i],rvar[i],ivar[i],zvar[i]])+np.ones((5,5))*propparvar[i] for i in range(len(uvar))]
obsmags_covar_chol = [np.linalg.cholesky(o_c_i) for o_c_i in obsmags_covar]
print(obsmags_covar_chol[0])
obsmags_covar_logdet = [math.log(np.linalg.det(o_c_i)) for o_c_i in obsmags_covar]
print(obsmags_covar_logdet[0])


#np.savez('./WD_data',obsmags=obsmags,obsmags_covar_chol=obsmags_covar_chol,obsmags_covar_logdet=obsmags_covar_logdet)

def multivariate_normal(self,diff,cov):
    dim = len(diff)
    assert dim==np.shape(cov)[0] and dim==np.shape(cov)[1]
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)
    return math.exp(-1./2.*np.dot(np.dot(cov_inv,diff),diff))/math.sqrt((2.*pi)**dim*cov_det)

#exit()


from BergeronInterp import BergeronDAInterp,BergeronDBInterp
umag = df[cleancuts]['sdssdr9_u_mag']-5.*(np.log10(1e3/df[cleancuts]['parallax'])-1.)
gmag = df[cleancuts]['sdssdr9_g_mag']-5.*(np.log10(1e3/df[cleancuts]['parallax'])-1.)
rmag = df[cleancuts]['sdssdr9_r_mag']-5.*(np.log10(1e3/df[cleancuts]['parallax'])-1.)
imag = df[cleancuts]['sdssdr9_i_mag']-5.*(np.log10(1e3/df[cleancuts]['parallax'])-1.)
zmag = df[cleancuts]['sdssdr9_z_mag']-5.*(np.log10(1e3/df[cleancuts]['parallax'])-1.)
col1 = df[cleancuts]['sdssdr9_u_mag']-df[cleancuts]['sdssdr9_g_mag']
col2 = df[cleancuts]['sdssdr9_g_mag']-df[cleancuts]['sdssdr9_r_mag']
col3 = df[cleancuts]['sdssdr9_r_mag']-df[cleancuts]['sdssdr9_i_mag']
col4 = df[cleancuts]['sdssdr9_i_mag']-df[cleancuts]['sdssdr9_z_mag']

def plot_grid(xax,yax):
    Teff=np.array([3000.,4000.,5000.,6000.,7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0, 13000.0, 14500.0, 16000.0, 18000.0, 20000.0, 22500.0, 25000.0, 30000.0, 40000.0, 50000.0, 70000.0, 90000.0, 110000.0, 125000.])
    logg=np.linspace(7.,9.5,6)
    def internal_f(colors,whatax):
        if whatax=='col1':
            res=colors[:,0]-colors[:,1]
        elif whatax=='col2':
            res=colors[:,1]-colors[:,2]
        elif whatax=='col3':
            res=colors[:,2]-colors[:,3]
        elif whatax=='col4':
            res=colors[:,3]-colors[:,4]
        elif whatax=='umag':
            res=colors[:,0]
        elif whatax=='gmag':
            res=colors[:,1]
        elif whatax=='rmag':
            res=colors[:,2]
        elif whatax=='imag':
            res=colors[:,3]
        elif whatax=='zmag':
            res=colors[:,4]
        return res
    for i in range(len(Teff)):
        colors=BergeronDAInterp([Teff[i]]*len(logg),logg)
        xx = internal_f(colors,xax)
        yy = internal_f(colors,yax)
        plt.plot(xx,yy,'b')
        colors=BergeronDBInterp([Teff[i]]*len(logg),logg)
        xx = internal_f(colors,xax)
        yy = internal_f(colors,yax)
        plt.plot(xx,yy,'r')
    for i in range(len(logg)):
        colors=BergeronDAInterp(Teff,[logg[i]]*len(Teff))
        xx = internal_f(colors,xax)
        yy = internal_f(colors,yax)
        plt.plot(xx,yy,'b')
        colors=BergeronDBInterp(Teff,[logg[i]]*len(Teff))
        xx = internal_f(colors,xax)
        yy = internal_f(colors,yax)
        plt.plot(xx,yy,'r')


plot_grid('col1','col2')
plt.scatter(col1,col2,s=5.,alpha=0.1)
plt.show()

plot_grid('col1','umag')
plt.ylim([18.,7.])
plt.scatter(col1,umag,s=5.,alpha=0.1)
plt.show()

plot_grid('col3','gmag')
plt.scatter(col3,gmag,s=5.,alpha=0.1)
#plt.ylim([18.,7.])
plt.show()

plot_grid('col3','rmag')
#plt.tight_layout()
#plt.ylim([18.,7.])
plt.scatter(col3,rmag,s=5.,alpha=0.1)
plt.show()

plot_grid('col3','imag')
#plt.tight_layout()
#plt.ylim([18.,7.])
plt.scatter(col3,imag,s=5.,alpha=0.1)
plt.show()


plot_grid('col4','imag')
#plt.tight_layout()
#plt.ylim([18.,7.])
plt.scatter(col4,imag,s=5.,alpha=0.1)
plt.show()
