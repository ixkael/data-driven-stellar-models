import numpy as np
#from scipy.interpolate import griddata
#from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline


#### DA
#DAtable=open("/Users/axelwidmark/Documents/Fysik/WD/BergeronModels/AllTables/Table_DA")
DAtable=open("../BergeronModels/GaiaTables/Table_DA")
TeffDA=[1500.0, 1750.0, 2000.0, 2250.0, 2500.0, 2750.0, 3000.0, 3250.0, 3500.0, 3750.0, 4000.0, 4250.0, 4500.0, 4750.0, 5000.0, 5250.0, 5500.0, 6000.0, 6500.0, 7000.0, 7500.0, 8000.0, 8500.0, 9000.0, 9500.0, 10000.0, 10500.0, 11000.0, 11500.0, 12000.0, 12500.0, 13000.0, 13500.0, 14000.0, 14500.0, 15000.0, 15500.0, 16000.0, 16500.0, 17000.0, 20000.0, 25000.0, 30000.0, 35000.0, 40000.0, 45000.0, 50000.0, 55000.0, 60000.0, 65000.0, 70000.0, 75000.0, 80000.0, 85000.0, 90000.0, 100000.0, 110000.0, 120000.0]
loggDA=[7.0,7.5,8.0,8.5,9.0,9.5]
massDA=[]
MbolDA=[]
uDA=[]
gDA=[]
rDA=[]
iDA=[]
zDA=[]
ageDA=[]
GmagDA=[]
BPmagDA=[]
RPmagDA=[]
for line in DAtable:
    line = line.strip()
    columns = line.split()
    if columns[0]!='6' and columns[0]!='Teff':
        massDA.append(float(columns[2]))
        MbolDA.append(float(columns[3]))
        uDA.append(float(columns[13]))
        gDA.append(float(columns[14]))
        rDA.append(float(columns[15]))
        iDA.append(float(columns[16]))
        zDA.append(float(columns[17]))
        GmagDA.append(float(columns[21]))
        BPmagDA.append(float(columns[22]))
        RPmagDA.append(float(columns[23]))
        ageDA.append(float(columns[24]))
bbox=[1500.,120000.0,6.5,9.5]
deg=1
arrange=lambda vec: np.transpose(np.reshape(vec,(len(loggDA),len(TeffDA))))
massDAf=RectBivariateSpline(TeffDA,loggDA,arrange(massDA))
MbolDAf=RectBivariateSpline(TeffDA,loggDA,arrange(MbolDA),bbox=bbox,kx=deg,ky=deg)
uDAf=RectBivariateSpline(TeffDA,loggDA,arrange(uDA),bbox=bbox,kx=deg,ky=deg)
gDAf=RectBivariateSpline(TeffDA,loggDA,arrange(gDA),bbox=bbox,kx=deg,ky=deg)
rDAf=RectBivariateSpline(TeffDA,loggDA,arrange(rDA),bbox=bbox,kx=deg,ky=deg)
iDAf=RectBivariateSpline(TeffDA,loggDA,arrange(iDA),bbox=bbox,kx=deg,ky=deg)
zDAf=RectBivariateSpline(TeffDA,loggDA,arrange(zDA),bbox=bbox,kx=deg,ky=deg)
GmagDAf=RectBivariateSpline(TeffDA,loggDA,arrange(GmagDA),bbox=bbox,kx=deg,ky=deg)
BPmagDAf=RectBivariateSpline(TeffDA,loggDA,arrange(BPmagDA),bbox=bbox,kx=deg,ky=deg)
RPmagDAf=RectBivariateSpline(TeffDA,loggDA,arrange(RPmagDA),bbox=bbox,kx=deg,ky=deg)
ageDAf=RectBivariateSpline(TeffDA,loggDA,arrange(ageDA))
lnageDAf=RectBivariateSpline(TeffDA,loggDA,arrange(np.log(ageDA)))
def BergeronDAInterp(Teff_in,logg_in):
    try:
        output=[]
        for i in range(len(Teff_in)):
            output.append([uDAf.ev(Teff_in[i],logg_in[i]),gDAf.ev(Teff_in[i],logg_in[i]),rDAf.ev(Teff_in[i],logg_in[i]),iDAf.ev(Teff_in[i],logg_in[i]),zDAf.ev(Teff_in[i],logg_in[i])])
        return np.array(output)
    except TypeError:
        #return [uDAf.ev(Teff_in,logg_in)[0],gDAf.ev(Teff_in,logg_in)[0],rDAf.ev(Teff_in,logg_in)[0],iDAf.ev(Teff_in,logg_in)[0],zDAf.ev(Teff_in,logg_in)[0]]
        return np.array( [uDAf.ev(Teff_in,logg_in),gDAf.ev(Teff_in,logg_in),rDAf.ev(Teff_in,logg_in),iDAf.ev(Teff_in,logg_in),zDAf.ev(Teff_in,logg_in)] )

#### DB
#DBtable=open("/Users/axelwidmark/Documents/Fysik/WD/BergeronModels/AllTables/Table_DB")
DBtable=open("../BergeronModels/GaiaTables/Table_DB")
TeffDB=[3500.0, 3750.0, 4000.0, 4250.0, 4500.0, 4750.0, 5000.0, 5250.0, 5500.0, 6000.0, 6500.0, 7000.0, 7500.0, 8000.0, 8500.0, 9000.0, 9500.0, 10000.0, 11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0, 23000.0, 24000.0, 25000.0, 26000.0, 27000.0, 28000.0, 29000.0, 30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0, 36000.0, 37000.0, 38000.0, 39000.0, 40000.0]
loggDB=[7.0,7.5,8.0,8.5,9.0]
massDB=[]
MbolDB=[]
uDB=[]
gDB=[]
rDB=[]
iDB=[]
zDB=[]
GmagDB=[]
BPmagDB=[]
RPmagDB=[]
ageDB=[]
for line in DBtable:
    line = line.strip()
    columns = line.split()
    if columns[0]!='5' and columns[0]!='Teff':
        massDB.append(float(columns[2]))
        MbolDB.append(float(columns[3]))
        uDB.append(float(columns[13]))
        gDB.append(float(columns[14]))
        rDB.append(float(columns[15]))
        iDB.append(float(columns[16]))
        zDB.append(float(columns[17]))
        GmagDB.append(float(columns[21]))
        BPmagDB.append(float(columns[22]))
        RPmagDB.append(float(columns[23]))
        ageDB.append(float(columns[24]))
bbox=[3500.,120000.0,6.5,9.5]
arrange=lambda vec: np.transpose(np.reshape(vec,(len(loggDB),len(TeffDB))))
massDBf=RectBivariateSpline(TeffDB,loggDB,arrange(massDB))
MbolDBf=RectBivariateSpline(TeffDB,loggDB,arrange(MbolDB),bbox=bbox,kx=deg,ky=deg)
uDBf=RectBivariateSpline(TeffDB,loggDB,arrange(uDB),bbox=bbox,kx=deg,ky=deg)
gDBf=RectBivariateSpline(TeffDB,loggDB,arrange(gDB),bbox=bbox,kx=deg,ky=deg)
rDBf=RectBivariateSpline(TeffDB,loggDB,arrange(rDB),bbox=bbox,kx=deg,ky=deg)
iDBf=RectBivariateSpline(TeffDB,loggDB,arrange(iDB),bbox=bbox,kx=deg,ky=deg)
zDBf=RectBivariateSpline(TeffDB,loggDB,arrange(zDB),bbox=bbox,kx=deg,ky=deg)
GmagDBf=RectBivariateSpline(TeffDB,loggDB,arrange(GmagDB),bbox=bbox,kx=deg,ky=deg)
BPmagDBf=RectBivariateSpline(TeffDB,loggDB,arrange(BPmagDB),bbox=bbox,kx=deg,ky=deg)
RPmagDBf=RectBivariateSpline(TeffDB,loggDB,arrange(RPmagDB),bbox=bbox,kx=deg,ky=deg)
ageDBf=RectBivariateSpline(TeffDB,loggDB,arrange(ageDB))
lnageDBf=RectBivariateSpline(TeffDB,loggDB,arrange(np.log(ageDB)))
def BergeronDBInterp(Teff_in,logg_in):
    try:
        output=[]
        for i in range(len(Teff_in)):
            output.append([uDBf.ev(Teff_in[i],logg_in[i]),gDBf.ev(Teff_in[i],logg_in[i]),rDBf.ev(Teff_in[i],logg_in[i]),iDBf.ev(Teff_in[i],logg_in[i]),zDBf.ev(Teff_in[i],logg_in[i])])
        return np.array(output)
    except TypeError:
        return np.array( [uDBf.ev(Teff_in,logg_in),gDBf.ev(Teff_in,logg_in),rDBf.ev(Teff_in,logg_in),iDBf.ev(Teff_in,logg_in),zDBf.ev(Teff_in,logg_in)] )





print(BergeronDAInterp(1e4,8.)[0])

grid_len = 500
temp_vec = np.logspace(np.log10(3000.),np.log10(120000.),grid_len)
loggDA_vec=np.linspace(6.5,9.5,grid_len)
loggDB_vec=np.linspace(6.5,9.,grid_len)

DA_grid = np.array( [[[BergeronDAInterp(t,g)[i] for g in loggDA_vec] for t in temp_vec] for i in range(5)] )
DB_grid = np.array( [[[BergeronDAInterp(t,g)[i] for g in loggDB_vec] for t in temp_vec] for i in range(5)] )
model_grid = np.array( [[[[BergeronDAInterp(t,loggDA_vec[g_i])[i],BergeronDBInterp(t,loggDB_vec[g_i])[i]] for t in temp_vec] for g_i in range(len(loggDA_vec))] for i in range(5)] )

for i in range(5):
    print(np.min(DA_grid[i]),np.max(DA_grid[i]))

print(np.shape(DA_grid))
print(np.shape(DB_grid))
print(np.shape(model_grid))

diffA = 1.
diffB = 1.
for i in range(grid_len-1):
    for j in range(grid_len):
        for k in range(5):
            diff = abs(DA_grid[k][i][j]-DA_grid[k][i+1][j])
            if diff<diffA:
                diffA = diff
            diff = abs(DA_grid[k][j][i]-DA_grid[k][j][i+1])
            if diff<diffA:
                diffA = diff
            diff = abs(DB_grid[k][i][j]-DB_grid[k][i+1][j])
            if diff<diffB:
                diffB = diff
            diff = abs(DB_grid[k][j][i]-DB_grid[k][j][i+1])
            if diff<diffB:
                diffB = diff
print(diffA,diffB)

np.savez('./model_grids',model_grid=model_grid,DA_grid=DA_grid,DB_grid=DB_grid,temp_vec=temp_vec,loggDA_vec=loggDA_vec,loggDB_vec=loggDB_vec)