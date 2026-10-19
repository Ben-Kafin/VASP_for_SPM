import matplotlib.pyplot as plt
import sys
import getopt
from numpy import array,dot,cross,cos,sin,pi,percentile,average,std,arcsinh
from numpy.linalg import norm,inv
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
import os

#3D arrow geometry constants
ARROW_SEGMENTS=12
HEAD_LENGTH_FRAC=0.25
HEAD_RADIUS_FRAC=0.10

def plot_arrow3d(ax,origin,vec,color):
    #draws an arrow as a line shaft plus a solid cone head built from a Poly3DCollection
    #the cone scales with the arrow: head length and head radius are both fixed fractions of |vec|
    origin=array(origin,dtype=float)
    vec=array(vec,dtype=float)
    L=norm(vec)
    if L==0.0:
        return
    d=vec/L
    head_length=HEAD_LENGTH_FRAC*L
    head_radius=HEAD_RADIUS_FRAC*L
    shaft_end=origin+d*(L-head_length)
    apex=origin+vec
    ax.plot([origin[0],shaft_end[0]],[origin[1],shaft_end[1]],[origin[2],shaft_end[2]],color=color,linewidth=1.5)
    #orthonormal basis perpendicular to the arrow direction
    if abs(d[2])<0.9:
        ref=array([0.0,0.0,1.0])
    else:
        ref=array([1.0,0.0,0.0])
    u=cross(d,ref)
    u=u/norm(u)
    w=cross(d,u)
    rim=[shaft_end+head_radius*(cos(2*pi*i/ARROW_SEGMENTS)*u+sin(2*pi*i/ARROW_SEGMENTS)*w) for i in range(ARROW_SEGMENTS)]
    faces=[[apex,rim[i],rim[(i+1)%ARROW_SEGMENTS]] for i in range(ARROW_SEGMENTS)]
    faces.append(rim)
    cone=Poly3DCollection(faces,facecolors=color,edgecolors=color)
    ax.add_collection3d(cone)

def forces(outcar,poscar,**args):
    if 'show_all' in args and args['show_all']:
        show_all=True
    else:
        show_all=False

    try:
        lv,coord,atomtypes,atomnums,seldyn = parse_poscar(poscar)
    except ValueError:
        lv,coord,atomtypes,atomnums = parse_poscar(poscar)
        seldyn=['TTT' for i in range(sum(atomnums))]

    positions,forces,time,tol,mode=parse_forces(outcar,seldyn=seldyn)
    nsteps=len(time)

    def scaling_for(window_start):
        #single continuous arrow-length mapping: L(f)=Lmax*asinh(f/beta)/asinh(fmax/beta)
        #with the knee beta at mean+std of every plotted vector (f>EDIFFG) in the window;
        #forces near the mean scale almost linearly while outliers are compressed
        #progressively with their distance above the distribution, yet stay the longest
        #arrows since the mapping only ever rises; the window max pins exactly to Lmax
        pop=[f for s in range(window_start,nsteps) for f in forces[3][s] if f>tol]
        if len(pop)==0:
            #fully converged window: fall back to all nonzero forces so -a still scales
            pop=[f for s in range(window_start,nsteps) for f in forces[3][s] if f>0.0]
        if len(pop)==0:
            return (1.0,1.0)
        return (average(pop)+std(pop),max(pop))

    def arrow_length_for(scaling,f):
        beta,fmax=scaling
        return 5.0*arcsinh(f/beta)/arcsinh(fmax/beta)

    #axis limits: x/y cover the full unit cell footprint; the z top is conditional:
    #if any plotted arrow tip (under the initial full-trajectory scaling) reaches above the
    #tallest atom of any step, the plot extends one head-cone length above the highest tip;
    #otherwise it extends 2 A above the tallest atom
    #atoms sitting on the top cell face (periodic images of the bottom) do not count
    corners=[i*lv[0]+j*lv[1]+k*lv[2] for i in range(2) for j in range(2) for k in range(2)]
    corners=array(corners)
    lo=[min(corners[:,i]) for i in range(3)]
    hi=[max(corners[:,i]) for i in range(3)]
    ilv=inv(lv)
    zmax_atoms=lo[2]
    for p in positions:
        for q in p:
            if dot(q,ilv)[2]<0.999 and q[2]>zmax_atoms:
                zmax_atoms=q[2]
    scal0=scaling_for(0)
    tip_z=None
    tip_head=0.0
    for s in range(nsteps):
        for i in range(len(forces[3][s])):
            fmag=forces[3][s][i]
            if (fmag>tol or show_all) and fmag>0.0:
                L=arrow_length_for(scal0,fmag)
                z=positions[s][i][2]+forces[2][s][i]*L/fmag
                if tip_z is None or z>tip_z:
                    tip_z=z
                    tip_head=HEAD_LENGTH_FRAC*L
    if tip_z is not None and tip_z>zmax_atoms:
        hi[2]=tip_z+tip_head
    else:
        hi[2]=zmax_atoms+2.0
    if hi[2]<=lo[2]:
        hi[2]=max(corners[:,2])

    fig=plt.figure(figsize=(14,9))
    gs=fig.add_gridspec(3,1,height_ratios=[5,1,0.15])
    ax=fig.add_subplot(gs[0],projection='3d')
    axf=fig.add_subplot(gs[1])
    ax_slider=fig.add_subplot(gs[2])
    fig.subplots_adjust(left=0.06,right=0.94,top=0.99,bottom=0.05,hspace=0.25)
    #mouse bindings on the 3D panel: left drag rotates, right drag pans in the plane of the
    #plot, scroll wheel zooms; the default right-drag zoom is disabled
    ax.mouse_init(rotate_btn=1,pan_btn=3,zoom_btn=[])

    marker_size=2000/max([norm(lv[j]) for j in range(3)])

    state={'step':0,'pressed':False,'window':0,'scaling':(1.0,1.0),'lims':[[lo[i],hi[i]] for i in range(3)]}

    def compute_scaling(window_start):
        #arrow-length mapping, recomputed whenever the window changes
        state['scaling']=scaling_for(window_start)

    def arrow_length(f):
        return arrow_length_for(state['scaling'],f)

    def apply_limits():
        #zoomed limits persist in state so redraws (stepping, window changes) keep the current zoom
        ax.set_xlim(state['lims'][0][0],state['lims'][0][1])
        ax.set_ylim(state['lims'][1][0],state['lims'][1][1])
        ax.set_zlim(state['lims'][2][0],state['lims'][2][1])
        ax.set_box_aspect(tuple(state['lims'][i][1]-state['lims'][i][0] for i in range(3)))

    def draw_step(step):
        ax.cla()
        #plots atom coordinates at this ionic step
        for i in range(len(atomtypes)):
            tempvar=[[],[],[]]
            for j in range(atomnums[i]):
                for k in range(3):
                    tempvar[k].append(positions[step][j+sum(atomnums[:i])][k])
            ax.scatter(tempvar[0],tempvar[1],tempvar[2],s=marker_size,label=atomtypes[i])
        #plots force vectors at this ionic step
        for i in range(len(atomtypes)):
            tempo=[[],[],[]]
            tempv=[[],[],[]]
            for j in range(atomnums[i]):
                fmag=forces[3][step][j+sum(atomnums[:i])]
                if (fmag>tol or show_all) and fmag>0.0:
                    scale=arrow_length(fmag)/fmag
                    for k in range(3):
                        tempo[k].append(positions[step][j+sum(atomnums[:i])][k])
                        tempv[k].append(forces[k][step][j+sum(atomnums[:i])]*scale)
            for j in range(len(tempo[0])):
                plot_arrow3d(ax,[tempo[0][j],tempo[1][j],tempo[2][j]],[tempv[0][j],tempv[1][j],tempv[2][j]],'k')
        apply_limits()
        if mode=='time':
            ax.set_title('step {} of {} | {:.3f} fs'.format(step+1,nsteps,time[step]))
        else:
            ax.set_title('step {} of {}'.format(step+1,nsteps))

    #convergence subplot: statistics of the total force per ionic step, as in plot_forces
    minima=[]
    averages=[]
    maxima=[]
    upperq=[]
    lowerq=[]
    for j in forces[3]:
        if len(j)>0:
            minima.append(min(j))
            averages.append(average(j))
            maxima.append(max(j))
            upperq.append(percentile(j,75))
            lowerq.append(percentile(j,25))
    data_labels=['minimum','lower quartile','average','upper quartile','maximum']
    data_sets=[minima,lowerq,averages,upperq,maxima]
    for k,l in zip(data_labels,data_sets):
        try:
            axf.scatter(time,l,label=k)
            max_range=max(maxima)-min(minima)
            axf.set_ylim(bottom=min(minima)-0.05*max_range,top=max(maxima)+0.05*max_range)
        except ValueError:
            pass
    if len(time)==1:
        axf.plot([-1,1],[tol,tol],linestyle='dashed',label='convergence')
    else:
        axf.plot([time[0],time[-1]],[tol,tol],linestyle='dashed',label='convergence')
    axf.set(ylabel='$F_{total}$'+' / eV $\\AA^{-1}$')
    xlabels={'time':'optimization time / fs','steps':'optimization steps'}
    axf.set(xlabel=xlabels[mode])
    handles,labels=axf.get_legend_handles_labels()
    fig.legend(handles,labels,bbox_to_anchor=(1.01,0.5),loc='right')

    #time marker on the convergence subplot acts as the slider:
    #clicking or dragging within the subplot snaps it to the nearest ionic step
    #and redraws the 3D panel at that step
    marker=axf.axvline(time[0],color='k')

    def set_step(step):
        if step!=state['step']:
            state['step']=step
            marker.set_xdata([time[step],time[step]])
            draw_step(step)
            fig.canvas.draw_idle()

    def nearest_step(x):
        return min(range(state['window'],nsteps),key=lambda i: abs(time[i]-x))

    def on_press(event):
        if event.inaxes is axf and event.xdata is not None:
            state['pressed']=True
            set_step(nearest_step(event.xdata))

    def on_motion(event):
        if state['pressed'] and event.inaxes is axf and event.xdata is not None:
            set_step(nearest_step(event.xdata))

    def on_release(event):
        state['pressed']=False
        #matplotlib's built-in 3D pan (right-button drag) shifts the axis limits internally:
        #capture them so redraws keep the panned view
        state['lims']=[list(ax.get_xlim3d()),list(ax.get_ylim3d()),list(ax.get_zlim3d())]

    def on_key(event):
        #arrow keys step through frames one at a time, so consecutive steps that share
        #an x value (zero displacement between geometries) all remain reachable
        if event.key=='right':
            set_step(min(state['step']+1,nsteps-1))
        elif event.key=='left':
            set_step(max(state['step']-1,state['window']))

    def on_scroll(event):
        #scroll wheel zooms the 3D panel in and out about the current view center
        #all three axes scale by the same factor, preserving the box aspect
        if event.inaxes is ax:
            if event.button=='up':
                factor=0.9
            else:
                factor=1.1
            for i in range(3):
                center=0.5*(state['lims'][i][0]+state['lims'][i][1])
                half=0.5*(state['lims'][i][1]-state['lims'][i][0])*factor
                state['lims'][i]=[center-half,center+half]
            apply_limits()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event',on_press)
    fig.canvas.mpl_connect('motion_notify_event',on_motion)
    fig.canvas.mpl_connect('button_release_event',on_release)
    fig.canvas.mpl_connect('key_press_event',on_key)
    fig.canvas.mpl_connect('scroll_event',on_scroll)

    def update_ylim():
        #y limits follow the visible window's data, same 5% padding rule as the full-range plot
        if len(maxima)==nsteps and len(minima)==nsteps:
            vis_max=max(maxima[state['window']:])
            vis_min=min(minima[state['window']:])
            vis_range=vis_max-vis_min
            if vis_range==0.0:
                vis_range=vis_max
            if vis_range>0.0:
                axf.set_ylim(bottom=vis_min-0.05*vis_range,top=vis_max+0.05*vis_range)

    def update_window(val):
        idx=min(range(nsteps),key=lambda i: abs(time[i]-val))
        state['window']=idx
        if idx==nsteps-1 or time[-1]-time[idx]<=0.0:
            #degenerate window (single step, or zero displacement across the visible steps):
            #pad xlim to keep a finite axis, falling back to a fraction of the full range
            dt=time[-1]-time[-2]
            if dt<=0.0:
                dt=(time[-1]-time[0])*0.05
            if dt<=0.0:
                dt=1.0
            axf.set_xlim(time[-1]-dt,time[-1]+dt)
        else:
            axf.set_xlim(time[idx],time[-1])
        update_ylim()
        compute_scaling(idx)
        if state['step']<idx:
            state['step']=idx
            marker.set_xdata([time[idx],time[idx]])
        draw_step(state['step'])
        fig.canvas.draw_idle()

    if nsteps>1:
        window_slider=Slider(ax_slider,'window start',time[0],time[-1],valinit=time[0],valstep=array(time))
        window_slider.on_changed(update_window)
        #keep a hard reference on the axes: widget callbacks are weakly referenced and die if the Slider is garbage collected
        ax_slider._window_slider=window_slider
        axf.set_xlim(time[0],time[-1])
    else:
        ax_slider.set_visible(False)

    compute_scaling(0)
    draw_step(0)
    species_handles,species_labels=ax.get_legend_handles_labels()
    fig.legend(species_handles,species_labels)
    plt.show()

def parse_forces(ifile,**args):
    if 'seldyn' in args:
        seldyn=args['seldyn']
    else:
        seldyn='none'
    
    dt=[]
    positions=[]
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
                                temp_forces[j-3].append(float(line[j]))
                                tempvar.append(float(line[j]))
                            else:
                                temp_forces[j-3].append(0.0)
                                tempvar.append(0.0)
                        if len(tempvar)>0:
                            temp_forces[3].append(norm(array(tempvar)))
                    temp_pos=array(temp_pos)
                    positions.append(temp_pos)
                    for i in range(4):
                        forces[i].append(temp_forces[i])
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

    return positions,forces,time,tol,mode

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
            coord[i]=dot(coord[i],latticevectors)
            
    #latticevectors formatted as a 3x3 array
    #coord holds the atomic coordinates with shape ()
    try:
        return latticevectors, coord, atomtypes, atomnums, seldyn
    except NameError:
        return latticevectors, coord, atomtypes, atomnums

if __name__=='__main__':
    default_dir =  r'dir/run'
    # These set the initial defaults
    outcar = os.path.join(default_dir, 'OUTCAR')
    poscar = os.path.join(default_dir, 'POSCAR')
    show_all=False
    try:
        opts,args=getopt.getopt(sys.argv[1:],'ho:p:a',['help','outcar=','poscar','all'])
    except getopt.GetoptError:
        print('error in command line syntax')
        sys.exit(2)
    for i,j in opts:
        if i in ['-h','--help']:
            print('''
input options:
    -o, --outcar          specify a path to the OUTCAR file other than ./OUTCAR
    -p, --poscar          specify an path to the POTCAR file other than ./POSCAR
    -a, --all             all forces are plotted, as opposed to just those larger than EDIFFG
    
help options:
    -h, --help            display this help message
                  ''')
            sys.exit()
        if i in ['-o','--outcar']:
            outcar=j
        if i in ['-p','--poscar']:
            poscar=j
        if i in ['-a','--all']:
            show_all=True
    forces(outcar,poscar,show_all=show_all)