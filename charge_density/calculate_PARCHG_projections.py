import numpy as np
import os

def calc_PARCHG_projections(ref,fp):
    os.chdir(ref)
    listdirref=os.listdir(ref)
    os.chdir(fp)
    listdirfp=os.listdir(fp)
    for j in listdirref:
        if 'PARCHG' in j:
            os.chdir(ref)
            print(j)
            energy=[]
            energy.append(float('.'.join([(j.split('.')[3]),(j.split('.')[4])])))
            eref,lv=parse_CHGCAR(j)[:2]
            os.chdir(fp)
            npts=np.shape(eref)
            bands=[]
            overlaps=[]
            overlaps_norm=[]
            average_zpos=[]
            zmask=np.zeros(npts)
            energies=[]
            energy_shift=[]          
            for i in range(npts[2]):
                zmask[:,:,i]=i/(npts[2]-1)*np.linalg.norm(lv[2])
            for i in listdirfp:
                if 'PARCHG' in i:
                    #print(i)
                    bands.append(int(i.split('.')[1])-1)
                    etemp=parse_CHGCAR(i)[0]
                    energies.append(float('.'.join([(i.split('.')[3]),(i.split('.')[4])])))
                    energy_shift.append(float(energies[-1])-float(energy[0]))
                    overlaps.append(np.linalg.norm(eref*etemp)/np.linalg.norm(etemp))
                    average_zpos.append(np.linalg.norm(zmask*etemp/np.linalg.norm(etemp)))
                    
            overlaps=np.array(overlaps)
            overlaps_norm=np.array(overlaps)
            overlaps_norm-=np.min(overlaps)
            overlaps_norm/=np.max(overlaps)
            bands=np.array(bands)        
            
            #return bands,overlaps,average_zpos,energies
            for i in range(len(energies)):
                if overlaps_norm[i]>0.9:
                    print(overlaps_norm[i],overlaps[i],energies[i],energy[0],energy_shift[i])

def parse_CHGCAR(ifile, **args):
    if 'scale' in args:
        rescale=False
    else:
        rescale=True
    #reads atomic positions and lattice vectors
    lv=np.zeros((3,3))
    with open(ifile,'r') as chgcar:
        for i in range(8):
            line=chgcar.readline().split()
            if i==1:
                sf=float(line[0])
            if i>1 and i<5:
                for j in range(3):
                    lv[i-2][j]=float(line[j])*sf
            if i==5:
                atomtypes=line
            if i==6:
                atomnums=line
                for j in range(len(atomnums)):
                    atomnums[j]=int(atomnums[j])
            if i==7:
                mode=line[0]
        coord=np.zeros((sum(atomnums),3))
        for i in range(sum(atomnums)):
            line=chgcar.readline().split()
            for j in range(3):
                coord[i][j]=float(line[j])
            if mode[0]=='D':
                coord[i]=np.dot(coord[i],lv)
        line=chgcar.readline()
        #starts reading charge density info
        line=chgcar.readline().split()
        x=0
        y=0
        z=0
        dim=[int(i) for i in line]
        e=np.zeros((dim[0],dim[1],dim[2]))
        searching=True
        while searching:
            line=chgcar.readline().split()
            for i in line:
                try:
                    e[x][y][z] = float(i)
                except ValueError:
                    float('E-'.join(i.split('-')))
                x+=1
                if x==dim[0]:
                    x=0
                    y+=1
                if y==dim[1]:
                    y=0
                    z+=1
                if z==dim[2]:
                    searching=False
                    break
    if rescale:
        #print('charge density values rescaled to electrons per cubic Angstrom')
        vol=np.dot(np.cross(lv[0],lv[1]),lv[2])
        e/=vol
    
    return e, lv, coord, atomtypes, atomnums