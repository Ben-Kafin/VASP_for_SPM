import matplotlib.pyplot as plt
import sys
import getopt
from numpy import array,dot,percentile,average
from numpy.linalg import norm,inv
import os

def main(outcar,poscar,**args):
    if 'quiet' in args and args['quiet']:
        quiet=True
    else:
        quiet=False

    try:
        seldyn = parse_poscar(poscar)[4]
    except (IndexError,FileNotFoundError):
        seldyn='none'
        
    forces,time,tol,mode=parse_forces(outcar,seldyn=seldyn)
    minima=[[],[],[],[]]
    averages=[[],[],[],[]]
    maxima=[[],[],[],[]]
    upperq=[[],[],[],[]]
    lowerq=[[],[],[],[]]
    for i in range(4):
        if len(forces[i])>0:
            for j in forces[i]:
                if len(j)>0:
                    minima[i].append(min(j))
                    averages[i].append(average(j))
                    maxima[i].append(max(j))
                    upperq[i].append(percentile(j,75))
                    lowerq[i].append(percentile(j,25))
    if not quiet:
        data_labels=['minimum','lower quartile','average','upper quartile','maximum']
        data_sets=[minima,lowerq,averages,upperq,maxima]
    else:
        data_labels=['minimum','average','maximum']
        data_sets=[minima,averages,maxima]
    #each component and the total force are plotted on their own subplot, along with the convergence criteria set by EDIFFG
    fig,axs=plt.subplots(4,1,sharex=True,figsize=(14,8))
    for i,j in zip(range(4),['_x','_y','_z','_{total}']):
        for k,l in zip(data_labels,data_sets):
            try:
                axs[i].scatter(time,l[i],label=k)
                max_range=max(maxima[i])-min(minima[i])
                axs[i].set_ylim(bottom=min(minima[i])-0.05*max_range,top=max(maxima[i])+0.05*max_range)
            except ValueError:
                pass
        if len(time)==1:
            axs[i].plot([-1,1],[tol,tol],linestyle='dashed',label='convergence')
        else:
            axs[i].plot([time[0],time[-1]],[tol,tol],linestyle='dashed',label='convergence')
        axs[i].set(ylabel='$F{}$'.format(j)+' / eV $\AA^{-1}$')
    xlabels={'time':'optimization time / fs','steps':'optimization steps'}
    axs[-1].set(xlabel=xlabels[mode])
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.01,0.5), loc='right')
    plt.show()
    
def parse_forces(ifile,**args):
    if 'seldyn' in args:
        seldyn=args['seldyn']
    else:
        seldyn='none'
    
    dt=[]
    forces=[[],[],[],[]]
    ibrion=0
    lv=None
    prev_pos=None
    last_dnorm=0.0
    cg_scale=None
    pending_tau=None
    try:
        with open(ifile,'r') as file:
            searching=True
            while searching:
                line=file.readline()
                if not line:
                    break
                if line.split()[:2]==['trial','='] and len(line.split())>2:
                    #IBRION=2 line search: the block just read was the trial move of a new
                    #cycle, which took trial*POTIM fs; that also fixes this cycle's time per
                    #distance for the remaining (collinear) line search moves
                    tau=abs(float(line.split()[2]))
                    if len(dt)>1 and last_dnorm>0.0:
                        dt[-1]=tau*abs(potim)
                        cg_scale=dt[-1]/last_dnorm
                if 'trialstep' in line:
                    #older VASP versions print the size of the upcoming trial move instead
                    temptau=line.split('trialstep',1)[1].split('=',1)
                    if len(temptau)>1 and len(temptau[1].split(')')[0].split())>0:
                        pending_tau=abs(float(temptau[1].split(')')[0].split()[0]))
                if 'EDIFFG' in line:
                    line=line.split()
                    tol=abs(float(line[line.index('EDIFFG')+2]))
                if 'IBRION' in line and '=' in line:
                    #a new job header: line search state from a previous job no longer applies
                    ibrion=int(line.split('=',1)[1].split()[0])
                    cg_scale=None
                    pending_tau=None
                if 'POTIM' in line and '=' in line:
                    potim=float(line.split('=',1)[1].split()[0])
                    if potim==0.0:
                        potim=-1.0
                if 'direct lattice vectors' in line:
                    #tolerate a partially written table when reading a live OUTCAR
                    templv=[file.readline().split()[:3] for i in range(3)]
                    if all(len(i)==3 for i in templv):
                        lv=array([[float(j) for j in i] for i in templv])
                if 'NIONS' in line:
                    line=line.split()
                    atomnum=int(line[line.index('NIONS')+2])
                    if seldyn=='none':
                        seldyn=['TTT' for i in range(atomnum)]
                elif 'TOTAL-FORCE' in line:
                    line=file.readline()
                    temp_pos=[]
                    temp_forces=[[],[],[],[]]
                    for i in range(atomnum):
                        line=file.readline().split()
                        temp_pos.append([float(line[j]) for j in range(3)])
                        tempvar=[]
                        for j in range(3,6):
                            if seldyn[i][j-3]=='T':
                                temp_forces[j-3].append(abs(float(line[j])))
                                tempvar.append(abs(float(line[j])))
                        if len(tempvar)>0:
                            temp_forces[3].append(norm(array(tempvar)))
                    for i in range(4):
                        forces[i].append(temp_forces[i])
                    temp_pos=array(temp_pos)
                    if prev_pos is None:
                        dt.append(0.0)
                        last_dnorm=0.0
                    else:
                        if lv is not None:
                            #actual distance moved since the previous geometry (minimum image)
                            dfrac=dot(temp_pos-prev_pos,inv(lv))
                            dfrac-=dfrac.round()
                            last_dnorm=norm(dot(dfrac,lv))
                        else:
                            last_dnorm=0.0
                        if ibrion!=0 and pending_tau is not None:
                            dt.append(pending_tau*abs(potim))
                            if last_dnorm>0.0:
                                cg_scale=dt[-1]/last_dnorm
                            pending_tau=None
                        elif ibrion!=0 and cg_scale is not None and last_dnorm>0.0:
                            dt.append(last_dnorm*cg_scale)
                        else:
                            #no line search information for this step: POTIM is the best estimate
                            dt.append(abs(potim))
                    prev_pos=temp_pos
    except Exception as err:
        print('error reading OUTCAR: {}'.format(err))
        sys.exit(1)

    if len(dt)==0:
        print('zero ionic steps read from OUTCAR')
        sys.exit()

    time=[dt[0]]
    for i in dt[1:]:
        time.append(time[-1]+i)

    if time[-1]>time[0]:
        mode='time'
    else:
        mode='steps'
        time=[float(i) for i in range(len(time))]

    return forces,time,tol,mode

def parse_poscar(ifile):
    with open(ifile, 'r') as file:
        lines=file.readlines()
        sf=float(lines[1])
        latticevectors=[float(lines[i].split()[j])*sf for i in range(2,5) for j in range(3)]
        latticevectors=array(latticevectors).reshape(3,3)
        atomtypes=lines[5].split()
        atomnums=[int(i) for i in lines[6].split()]
        if lines[7].split()[0] == 'Direct':
            start=8
        else:
            start=9
            seldyn=[''.join(lines[i].split()[-3:]) for i in range(start,sum(atomnums)+start)]
        coord=array([[float(lines[i].split()[j]) for j in range(3)] for i in range(start,sum(atomnums)+start)])
        for i in range(sum(atomnums)):
            coord[i]=dot(latticevectors,coord[i])
            
    #latticevectors formatted as a 3x3 array
    #coord holds the atomic coordinates with shape ()
    try:
        return latticevectors, coord, atomtypes, atomnums, seldyn
    except NameError:
        return latticevectors, coord, atomtypes, atomnums

if __name__=='__main__':

    # REPLACE THE PATH BELOW with your actual data directory
    default_dir = r'dir/run'
    
    # These set the initial defaults
    outcar = os.path.join(default_dir, 'OUTCAR')
    poscar = os.path.join(default_dir, 'POSCAR')
    quiet = False
    try:
        opts,args=getopt.getopt(sys.argv[1:],'ho:p:q',['help','outcar=','poscar','quiet'])
    except getopt.GetoptError:
        print('error in command line syntax')
        sys.exit(2)
    for i,j in opts:
        if i in ['-h','--help']:
            print('''
input options:
    -o, --outcar          specify a path to the OUTCAR file other than ./OUTCAR
    -p, --poscar          specify an path to the POTCAR file other than ./POSCAR
    -q, --quiet           suppresses plotting of quartiles for a less crowded output
    
help options:
    -h, --help            display this help message
                  ''')
            sys.exit()
        if i in ['-o','--outcar']:
            outcar=j
        if i in ['-p','--poscar']:
            poscar=j
        if i in ['-q','--quiet']:
            quiet=True
    main(outcar,poscar,quiet=quiet)