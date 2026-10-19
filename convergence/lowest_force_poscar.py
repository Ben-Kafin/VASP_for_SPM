from numpy import array,dot
from numpy.linalg import inv,norm
import sys
import os

def lowest_force_poscar(outcar,poscar,ofile,metric):
    with open(poscar,'r') as file:
        title=file.readline().rstrip('\n')
    try:
        lv,coord,atomtypes,atomnums,seldyn=parse_poscar(poscar)
    except ValueError:
        lv,coord,atomtypes,atomnums=parse_poscar(poscar)
        seldyn='none'
    if seldyn=='none':
        gates=['TTT' for i in range(sum(atomnums))]
    else:
        gates=seldyn

    positions,magnitudes,lattices,tol=parse_forces(outcar,gates)
    nsteps=len(positions)
    if nsteps==0:
        print('zero ionic steps read from OUTCAR')
        sys.exit(1)

    values=[]
    for i in magnitudes:
        if metric=='max':
            values.append(max(i))
        elif metric=='average':
            values.append(sum(i)/len(i))
        else:
            values.append(sum(i))
    best=values.index(min(values))

    if lattices[best] is not None:
        lv=lattices[best]

    print('read {} ionic steps from {}'.format(nsteps,outcar))
    if tol is not None:
        print('lowest {} force of {:.6f} eV/A at ionic step {} of {} | EDIFFG tolerance: {:.6f}'.format(metric,values[best],best+1,nsteps,tol))
    else:
        print('lowest {} force of {:.6f} eV/A at ionic step {} of {}'.format(metric,values[best],best+1,nsteps))

    if seldyn=='none':
        write_poscar(ofile,lv,positions[best],atomtypes,atomnums,title=title)
    else:
        write_poscar(ofile,lv,positions[best],atomtypes,atomnums,seldyn=seldyn,title=title)

def parse_forces(ifile,seldyn):
    #reads every ionic step from the OUTCAR: Cartesian positions, seldyn-gated force
    #magnitudes (frozen components contribute zero, as in forces.py), and the lattice
    #vectors in effect at that step
    positions=[]
    magnitudes=[]
    lattices=[]
    lv=None
    tol=None
    atomnum=len(seldyn)
    try:
        with open(ifile,'r') as file:
            while True:
                line=file.readline()
                if not line:
                    break
                if 'EDIFFG' in line:
                    line=line.split()
                    try:
                        tol=abs(float(line[line.index('EDIFFG')+2]))
                    except (ValueError,IndexError):
                        pass
                    continue
                if 'direct lattice vectors' in line:
                    #tolerate a partially written table when reading a live OUTCAR
                    templv=[file.readline().split()[:3] for i in range(3)]
                    if all(len(i)==3 for i in templv):
                        lv=array([[float(j) for j in i] for i in templv])
                if 'NIONS' in line:
                    line=line.split()
                    atomnum=int(line[line.index('NIONS')+2])
                    if atomnum!=len(seldyn):
                        print('atom count mismatch: {} ions in OUTCAR, {} in POSCAR'.format(atomnum,len(seldyn)))
                        sys.exit(1)
                elif 'TOTAL-FORCE' in line:
                    file.readline()
                    temp_pos=[]
                    temp_mag=[]
                    complete=True
                    for i in range(atomnum):
                        line=file.readline().split()
                        if len(line)<6:
                            complete=False
                            break
                        try:
                            temp_pos.append([float(line[j]) for j in range(3)])
                            tempvar=[]
                            for j in range(3,6):
                                if seldyn[i][j-3]=='T':
                                    tempvar.append(float(line[j]))
                                else:
                                    tempvar.append(0.0)
                            temp_mag.append(norm(array(tempvar)))
                        except ValueError:
                            complete=False
                            break
                    if not complete:
                        #a truncated block means the OUTCAR ends mid-write; keep what was read
                        break
                    positions.append(array(temp_pos))
                    magnitudes.append(temp_mag)
                    lattices.append(lv)
    except OSError as err:
        print('error reading OUTCAR: {}'.format(err))
        sys.exit(1)

    return positions,magnitudes,lattices,tol

def parse_poscar(ifile):
    with open(ifile, 'r') as file:
        lines=file.readlines()
        sf=float(lines[1])
        latticevectors=[float(lines[i].split()[j])*sf for i in range(2,5) for j in range(3)]
        latticevectors=array(latticevectors).reshape(3,3)
        atomtypes=lines[5].split()
        atomnums=[int(i) for i in lines[6].split()]
        if 'Direct' in lines[7] or 'Cartesian' in lines[7]:
            start=8
            mode=lines[7].split()[0]
        else:
            mode=lines[8].split()[0]
            start=9
            seldyn=[''.join(lines[i].split()[-3:]) for i in range(start,sum(atomnums)+start)]
        coord=array([[float(lines[i].split()[j]) for j in range(3)] for i in range(start,sum(atomnums)+start)])
        if mode!='Cartesian':
            for i in range(sum(atomnums)):
                for j in range(3):
                    while coord[i][j]>1.0 or coord[i][j]<0.0:
                        if coord[i][j]>1.0:
                            coord[i][j]-=1.0
                        elif coord[i][j]<0.0:
                            coord[i][j]+=1.0
                coord[i]=dot(coord[i],latticevectors)

    #latticevectors formatted as a 3x3 array
    #coord holds the atomic coordinates with shape ()
    try:
        return latticevectors, coord, atomtypes, atomnums, seldyn
    except NameError:
        return latticevectors, coord, atomtypes, atomnums

def write_poscar(ofile, lv, coord, atomtypes, atomnums, **args):
    with open(ofile,'w') as file:
        if 'title' in args:
            file.write(str(args['title']))
        file.write('\n1.0\n')
        for i in range(3):
            for j in range(3):
                file.write(str('{:<018f}'.format(lv[i][j])))
                if j<2:
                    file.write('  ')
            file.write('\n')
        for i in atomtypes:
            file.write('  '+str(i))
        file.write('\n')
        for i in atomnums:
            file.write('  '+str(i))
        file.write('\n')
        if 'seldyn' in args:
            file.write('Selective Dynamics\n')
        file.write('Direct\n')
        for i in range(len(coord)):
            coord[i]=dot(coord[i],inv(lv))
        for i in range(len(coord)):
            for j in range(3):
                file.write(str('{:<018f}'.format(coord[i][j])))
                if j<2:
                    file.write('  ')
            if 'seldyn' in args:
                for j in range(3):
                    file.write('  ')
                    file.write(args['seldyn'][i][j])
            file.write('\n')
    print('new POSCAR written to: '+str(ofile))

if __name__=='__main__':

    calc_dir = 'dir/run'
    metric   = 'max'    #max, average, or sum

    outcar=os.path.join(calc_dir,'OUTCAR')
    poscar=os.path.join(calc_dir,'POSCAR')
    ofile=os.path.join(calc_dir,'POSCAR_lowest_force')

    lowest_force_poscar(outcar,poscar,ofile,metric)
