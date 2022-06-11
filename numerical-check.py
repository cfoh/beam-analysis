'''
This Python test program performs testing of our developed forumlas.

The program contains a number of independent tests, please scroll down 
to the bottom to find the main launcher and specify a test to run.

For all tests, we can visually verify the outcome to confirm the correctness.
For some tests, we also perform simulation and compare the numerical and
simultion results to check the correctness.
'''

## the following are packages required for this program
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

###################################################################
# Some useful classes
###################################################################

class TF: # Transformation
    '''
    This is a static class, it collects a number of 
    transformation functions.
    '''
    def rotate(x,y,angle):
        x_rot = x * cos(angle) - y * sin(angle)
        y_rot = x * sin(angle) + y * cos(angle)
        return (x_rot,y_rot)

    def reflect(xy):
        (x_ref,y_ref) = xy
        x_ref *= -1
        return (x_ref,y_ref)

    def move_towards(x,y,dist,angle):
        return (x+dist*cos(angle),y+dist*sin(angle))

    def shift(x,y,dx,dy):
        return (x+dx,y+dy)

class PointXY:
    '''
    This class specifies an (x,y) point object. Use it to create 
    an (x,y) point.
    '''
    def __init__(self, x=0, y=0):
        (self.x,self.y) = (x,y)

    def to_polar(self, beam_center=(0,0), beam_angle=0):
        pt = self.clone()
        pt.x -= beam_center[0]
        pt.y -= beam_center[1]
        r = math.sqrt(pt.x**2+pt.y**2)
        degree = atan2(pt.y,pt.x) - beam_angle
        return (r, degree)

    def from_polar(self, r, degree, beam_center=(0,0), beam_angle=0):
        (self.x,self.y) = beam_center
        self.x += r*cos(degree+beam_angle)
        self.y += r*sin(degree+beam_angle)

    def xy(self):
        return (self.x,self.y)

    def clone(self):
        return PointXY(self.x,self.y)

    def is_inside(self, beam):
        (r,degree) = self.to_polar(beam.center.xy(),beam.theta)
        if r>beam.Range: return False
        if degree>=-beam.Width/2 and degree<=beam.Width/2: return True
        degree+=360
        if degree>=-beam.Width/2 and degree<=beam.Width/2: return True
        degree-=2*360
        if degree>=-beam.Width/2 and degree<=beam.Width/2: return True
        return False


###################################################################
# Some math functions for convenient use
###################################################################

def sin(degree):
    return math.sin(math.radians(degree))

def cos(degree):
    return math.cos(math.radians(degree))

def atan2(y,x):
    return math.degrees(math.atan2(y,x))

def atan(r):
    return math.degrees(math.atan(r))

def tan(degree):
    return math.tan(math.radians(degree))

def abs(x):
    return x if x>0 else -x

def circle_intersection(x1,y1,r1,x2,y2,r2):
    '''It returns the intersection between two circles. The circles have center points at
    (x1,y1) and (x2,y2), with radius r1 and r2, respectively.
    It returns `(px1,py1,px2,py2)` where (px1,py1) and (px2,py2) are the two points.
    If the circles don't intersec, `(None,None,None,None)` is returned.
    See: https://math.stackexchange.com/questions/256100/how-can-i-find-the-points-at-which-two-circles-intersect
    '''
    d = math.sqrt((x1-x2)**2+(y1-y2)**2)
    l = (r1**2 - r2**2 + d**2) / (2*d)
    if (r1**2-l**2)<0:
        return (None,None,None,None)
    h = math.sqrt(r1**2-l**2)
    px1 = (l/d)*(x2-x1) + (h/d)*(y2-y1) + x1
    py1 = (l/d)*(y2-y1) - (h/d)*(x2-x1) + y1
    px2 = (l/d)*(x2-x1) - (h/d)*(y2-y1) + x1
    py2 = (l/d)*(y2-y1) + (h/d)*(x2-x1) + y1
    return (px1,py1,px2,py2)

###################################################################
# Scenario settings
###################################################################

class Beam:
    Width = 60 
    Range = 80

    def __init__(self, x, y, degree):
        self.center = PointXY(x,y)
        self.theta  = degree
        self.start_angle = -Beam.Width/2
        self.end_angle = Beam.Width/2

    def clone(self): return Beam(self.x(),self.y(),self.theta)
    def x(self): return self.center.x
    def y(self): return self.center.y

b_k = Beam(0,0,80)

R_B = 80
beamwidth = 60
theta_k = 80
start_angle_k = -beamwidth/2
end_angle_k = beamwidth/2
(x_k,y_k) = (0,0)   # normalized to origin


###################################################################
# test 1: draw a horizontal line D_k away from the BS center 
#         on the south side
###################################################################
def test1():
    D_k = 25

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(patches.Wedge(b_k.center.xy(),b_k.Range,
                                b_k.theta+b_k.start_angle,
                                b_k.theta+b_k.end_angle,color="g",alpha=0.2))

    for angle in range(int(b_k.start_angle),int(b_k.end_angle)):
        R_k = D_k / sin(b_k.theta+angle)
        pt = PointXY(0,0)
        pt.from_polar(R_k,angle,b_k.center.xy(),b_k.theta)
        if pt.is_inside(b_k):
            ax1.plot(pt.x,pt.y,'ro')

    plt.text(0, -80,  "Show a horizontal line of the highway\n"
                      "edge where all vehicles can only appear\n"
                     f"on top of the line (D_k={D_k})", 
                     ha='center', wrap=True)
    plt.axis([-150, 150, -150, 150])
    plt.show()

###################################################################
# test 2: relative polar location translation between two sectors
##         Defined pt_i_from_k() to calculate (r_{i|k},phi_{i|k}}
###################################################################
def pt_i_from_k(r_k,phi_k,x_k,y_k,theta_k,x_i,y_i,theta_i):

    ## absolute location
    x_hat = x_k + r_k*cos(theta_k+phi_k) 
    y_hat = y_k + r_k*sin(theta_k+phi_k)

    ## convert to (r_i,phi_i) relative to b_i
    r_i = math.sqrt((x_i-x_hat)**2 + (y_i-y_hat)**2)
    phi_i = atan2(y_hat-y_i,x_hat-x_i) - theta_i

    return (r_i,phi_i)

def test2():

    ## another beam, b_i
    b_i = Beam(x=100, y=-10, degree=130)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    for bm,color in [(b_k,"r"),(b_i,"b")]:
        ax1.add_patch(patches.Wedge(bm.center.xy(),bm.Range,
                                    bm.theta+bm.start_angle,
                                    bm.theta+bm.end_angle,color=color,alpha=0.2))

    ## place a point pt_k relative to b_k
    (r_k,phi_k) = (40,10)
    pt_k = PointXY()
    pt_k.from_polar(r_k, phi_k, beam_center=b_k.center.xy(), beam_angle=b_k.theta)
    ax1.plot(pt_k.x,pt_k.y,'ro')

    ## convert (r_k,phi_k) to (r_i,phi_i) where pt_i is relative to b_i
    (x_k,y_k),theta_k = b_k.center.xy(), b_k.theta
    (x_i,y_i),theta_i = b_i.center.xy(), b_i.theta
    (r_i,phi_i) = pt_i_from_k(r_k,phi_k,x_k,y_k,theta_k,x_i,y_i,theta_i)

    ## plot (r_i,phi_i), should overlap with (r_k,phi_k)
    pt_i = PointXY()
    pt_i.from_polar(r_i,phi_i,b_i.center.xy(),b_i.theta)
    ax1.plot(pt_i.x,pt_i.y,'b^')

    ## test description, you may adjust the position & font size of this message
    plt.text(x=0, y=-80, 
                    s="Translate a location relative to one beam\n"
                      "to another. The two markers (before and after\n"
                      "translations) must appear at the same location", 
                      ha='center', wrap=True)
    plt.axis([-150, 150, -150, 150])
    plt.show()

###################################################################
# test 3: trajectory of a movement in a beam
###################################################################
def test3():

    b_k.theta = 20 # set a new pointing angle if needed

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(patches.Wedge(b_k.center.xy(),b_k.Range,
                                b_k.theta+b_k.start_angle,
                                b_k.theta+b_k.end_angle,color="g",alpha=0.2))

    ## trajectory setup
    ## (r_k,phi_k) is the starting point relative to b_k
    ## psi_k is the moving direction
    d = 0
    r_k = 40
    phi_k = 15
    psi_k = 250

    theta_k = b_k.theta
    while True:
        ## calculate point after moving d distance
        ## to test the correctness of the parametric functions
        gamma_k = psi_k-(theta_k+phi_k)
        r_d = math.sqrt(r_k**2 + d**2 + 2*r_k*d*cos(gamma_k))
        phi_d = phi_k + atan2(d*sin(gamma_k),r_k+d*cos(gamma_k))

        ## convert to (x,y) for plotting
        pt = PointXY()
        pt.from_polar(r_d,phi_d,b_k.center.xy(),b_k.theta)
        if not pt.is_inside(b_k): break
        ax1.plot(pt.x,pt.y,'ro')
        d += 1

    ## put a special marker on the starting location
    pt.from_polar(r_k,phi_k,b_k.center.xy(),b_k.theta)
    ax1.plot(pt.x,pt.y,'b^')

    ## test description, you may adjust the position & font size of this message
    plt.text(x=0, y=-80, 
                    s="Show the tracjectory of a vehicle\n"
                     f"moving at psi={psi_k} degree.",
                     ha='center', wrap=True)
    plt.axis([-150, 150, -150, 150])
    plt.show()


###################################################################
## test 4: transformation of a beam, cases 1 & 2
###################################################################
def test4():

    def case1(x,y,angle):
        return TF.reflect(TF.rotate(x,y,angle))

    def case2(x,y,angle):
        return TF.rotate(x,y,angle)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(patches.Wedge(b_k.center.xy(),b_k.Range,
                                b_k.theta+b_k.start_angle,
                                b_k.theta+b_k.end_angle,color="g",alpha=0.2))

    ## vehicle starting point, P2
    r_k = 50
    phi_k = 5
    psi_k = 190 # direction

    ## beam setup
    theta_k = b_k.theta
    beamwidth = b_k.Width
    R_B = b_k.Range
    (x_k,y_k) = b_k.center.xy()

    ## transformation case, pick a case below
    transform = case1 # use case1
    #transform = case2 # use case2

    if transform is case1:
        rot_angle = 180 - theta_k - beamwidth/2  # case 1
    else:
        rot_angle = beamwidth/2 - theta_k        # case 2

    ## setup P2, P3, P4
    pt2 = PointXY()
    pt2.from_polar(r_k,phi_k,(x_k,y_k),theta_k)
    ax1.plot(pt2.x,pt2.y,'ro')
    pt3 = PointXY(x=R_B*cos(theta_k+beamwidth/2),
                  y=R_B*sin(theta_k+beamwidth/2))
    pt4 = PointXY(x=R_B*cos(theta_k-beamwidth/2),
                  y=R_B*sin(theta_k-beamwidth/2))

    ## P2, P3, P4 after transformation (become Q2, Q3, Q4)
    (xp,yp) = transform(pt2.x,pt2.y,rot_angle)
    qt2 = PointXY(xp,yp)
    (xp,yp) = transform(pt3.x,pt3.y,rot_angle)
    qt3 = PointXY(xp,yp)
    (xp,yp) = transform(pt4.x,pt4.y,rot_angle)
    qt4 = PointXY(xp,yp)
    ax1.plot(qt2.x,qt2.y,'b>')
    plt.plot([0,qt3.x], [0,qt3.y], linestyle = 'dotted')
    plt.plot([0,qt4.x], [0,qt4.y], linestyle = 'dotted')
    plt.plot([qt3.x,qt4.x], [qt3.y,qt4.y], linestyle = 'dotted')

    ## calculated x_hat & y_hat for checking, should match with Q2
    if transform is case1:
        alpha0 = beamwidth/2 - phi_k # case 1
    else:
        alpha0 = beamwidth/2 + phi_k # case 2
    pt2hat = PointXY(x=r_k*cos(alpha0),y=r_k*sin(alpha0))
    ax1.plot(pt2hat.x,pt2hat.y,'g^')

    ## transform psi_k to psi_k' (or psip_k)
    if transform is case1: 
        psip_k = beamwidth/2 + theta_k - psi_k # case 1
    else:
        psip_k = beamwidth/2 - theta_k + psi_k # case 2
    if psip_k<0: psip_k+=360
    if psip_k>360: psip_k-=360

    ## alpha values
    alpha1 = atan(abs(pt2hat.x/pt2hat.y))
    alpha2 = atan(abs((R_B-pt2hat.x)/pt2hat.y))
    print("alpha1=%1.2f; alpah2=%1.2f"%(alpha1,alpha2))

    ## psi range
    is_psi_valid = True
    left_lim = 270-alpha1
    right_lim = 270+alpha2
    print("psi' = %1.2f to %1.2f"%(left_lim,right_lim))
    print("psi' is %1.2f"%psip_k)
    if psip_k<left_lim or psip_k>right_lim: 
        print("Warning: psi' is outside of the valid range")
        print("         no trajectory is shown.")
        is_psi_valid = False
    if transform is case1:
        left_lim = beamwidth/2 + theta_k - left_lim      # case 1
        right_lim = beamwidth/2 + theta_k - right_lim    # case 1
    else:
        left_lim = -beamwidth/2 + theta_k + left_lim      # case 2
        right_lim = -beamwidth/2 + theta_k + right_lim    # case 2
    while left_lim<0: left_lim+=360
    while left_lim>360: left_lim-=360
    while right_lim<0: right_lim+=360
    while right_lim>360: right_lim-=360

    ## departure point
    r_out = pt2hat.x - pt2hat.y*tan(270-psip_k)
    tau = math.sqrt(pt2hat.y**2 + (pt2hat.x-r_out)**2)
    phi_out = beamwidth/2
    if tau>R_B: tau=R_B

    ## trajectory
    (xm,ym) = TF.move_towards(pt2.x,pt2.y,tau,psi_k) # before tranformation
    if is_psi_valid:
        ax1.plot([pt2.x,xm],[pt2.y,ym],'r')

    (xm,ym) = TF.move_towards(qt2.x,qt2.y,tau,psip_k) # after tranformation
    if is_psi_valid:
        ax1.plot([qt2.x,xm],[qt2.y,ym],'g')

    case = 1 if transform is case1 else 2
    plt.text(x=0, y=-120, 
                    s=f"Transform a point in a beam using formulas\n"
                      f"from case {case} (orignal RED marker,\n"
                      f"after GREEN marker). BLUE marker & dotted\n"
                      f"lines are transformed using the\n"
                      f"program (not formula).\n"
                      f"BLUE and GREEN must overlap.",
                      ha='center', wrap=True)
    plt.axis([-150, 150, -150, 150])
    plt.show()

###################################################################
## test 5: transformation of a beam for case 3
###################################################################
def test5():

    ## beam setup & plot
    theta_k = b_k.theta
    beamwidth = b_k.Width
    R_B = b_k.Range
    (x_k,y_k) = b_k.center.xy()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(patches.Wedge((x_k,y_k),R_B,theta_k+start_angle_k,theta_k+end_angle_k,color="r",alpha=0.2))

    ## transformation setup
    rot_angle = 270 - theta_k
    dx = R_B*sin(beamwidth/2)
    dy = R_B*cos(beamwidth/2)
    def case3(x,y): # case 3 transformation
        (x,y) = TF.rotate(x,y,rot_angle)
        (x,y) = TF.shift(x,y,dx,dy)
        return (x,y)

    ## vehicle starting point, P2
    r_k = 40
    phi_k = 10
    psi_k = 100 # direction
    l = 43

    ## setup P2 & its transformed point Q2
    pt2 = PointXY()
    pt2.from_polar(r_k,phi_k,(x_k,y_k),theta_k)
    ax1.plot(pt2.x,pt2.y,'ro')
    qt2 = PointXY()
    (qt2.x,qt2.y) = case3(pt2.x,pt2.y)
    ax1.plot(qt2.x,qt2.y,'bo')

    ## calculated x_hat,y_hat
    pt2hat = PointXY()
    pt2hat.x = r_k*sin(phi_k) + R_B*sin(beamwidth/2)
    pt2hat.y = -r_k*cos(phi_k) + R_B*cos(beamwidth/2)
    ax1.plot(pt2hat.x,pt2hat.y,'g^')

    ## transformed sector, P1 is the center
    qt1 = PointXY()
    qt1.x = R_B*sin(beamwidth/2)
    qt1.y = R_B*cos(beamwidth/2)
    ax1.add_patch(patches.Wedge((qt1.x,qt1.y),R_B,270-beamwidth/2,270+beamwidth/2,color="g",alpha=0.2))

    ## P3 before/after the transformation: P3->Q3
    pt3 = PointXY(R_B*cos(theta_k+beamwidth/2),R_B*sin(theta_k+beamwidth/2))
    ax1.plot(pt3.x,pt3.y,'r^')
    qt3 = PointXY()
    (qt3.x,qt3.y) = case3(pt3.x,pt3.y)
    ax1.plot(qt3.x,qt3.y,'r^')

    ## transform psi_k to psi_k' (or psip_k)
    psip_k = 270 - theta_k + psi_k

    ## alpha values
    alpha1 = atan(abs(pt2hat.x/pt2hat.y))
    alpha2 = atan(abs((R_B-pt2hat.x)/pt2hat.y))
    print("alpha1=%1.2f; alpah2=%1.2f"%(alpha1,alpha2))

    ## psi range
    left_lim = 270-alpha1
    right_lim = 270+alpha2
    print("psi' = %1.2f to %1.2f"%(left_lim,right_lim))
    left_lim = -270 + theta_k + left_lim
    right_lim = -270 + theta_k + right_lim
    while left_lim<0: left_lim+=360
    while left_lim>360: left_lim-=360
    while right_lim<0: right_lim+=360
    while right_lim>360: right_lim-=360
    print("psi = %1.2f to %1.2f"%(left_lim,right_lim))

    ## find rho1, rho2 (using circle intersection points)
    ## circle_1 at P1, circle_2 at P2
    rho1 = PointXY()
    rho2 = PointXY()
    (rho1.x,rho1.y,rho2.x,rho2.y) = circle_intersection(qt1.x,qt1.y,R_B,pt2hat.x,pt2hat.y,l)
    if rho1.x is not None:
        ax1.plot(rho1.x,rho1.y,'y<')
        ax1.plot(rho2.x,rho2.y,'y<')
        ax1.add_patch(plt.Circle((pt2hat.x,pt2hat.y), l, color='b', alpha=0.5, fill=False))

    ## find distance, d, to the arc (approach sum of two vectors)
    ## vector 1: P1->P2, ie r_k∠phi_k
    ## vector 2: P2->arc, ie d∠psi_k
    ## sum of vectors: P1->arc, ie R_B∠angle
    ## where |r_k∠phi_k + d∠psi_k| = |R_B∠angle| = R_B
    gamma_k = psi_k-(theta_k+phi_k)
    d1 = -r_k*cos(gamma_k) + math.sqrt((r_k*cos(gamma_k))**2-(r_k**2-R_B**2))
    d2 = -r_k*cos(gamma_k) - math.sqrt((r_k*cos(gamma_k))**2-(r_k**2-R_B**2))
    print("dist1=%1.2f, dist2=%1.2f"%(d1,d2))

    ## trajectory
    tau = max(d1,d2) # accept only the positive solution
    (xm,ym) = TF.move_towards(pt2.x,pt2.y,tau,psi_k) # before tranformation
    ax1.plot([pt2.x,xm],[pt2.y,ym],'r')

    (xm,ym) = TF.move_towards(qt2.x,qt2.y,tau,psip_k) # after tranformation
    ax1.plot([qt2.x,xm],[qt2.y,ym],'g')

    ## test description, you may adjust the position & font size of this message
    plt.text(x=0, y=-120, 
                    s="Transform the beam using formulas from\n"
                     f"case 3 (before in RED, after in GREEN).\n"
                     f"Yellow markers are distance l={l} from the\n"
                     f"vehicle (intersection points with the arc).",
                     ha='center', wrap=True)
    plt.axis([-150, 150, -150, 150])
    plt.show()

###################################################################
## test 6: All 3 cases, with right condition to select
##         appropriate travelled distance
##         Defined J class to compute J(.) and tau
###################################################################
class J:
    def __init__(self, xy, theta_k, R_B, beamwidth):
        (self.x_k,self.y_k) = xy
        self.theta_k = theta_k
        self.beamwidth = beamwidth
        self.R_B = R_B

    def get_tau1(self,r_k,phi_k,psi_k):
        '''If case 1 condition is satisfied, return `tau`, otherwise return `None`.'''
        alpha0 = self.beamwidth/2 - phi_k
        x_hat = r_k*cos(alpha0)
        y_hat = r_k*sin(alpha0)
        psip_k = self.beamwidth/2 + self.theta_k - psi_k
        while psip_k<0: psip_k+=360
        while psip_k>360: psip_k-=360
        alpha1 = atan(abs(x_hat/y_hat))
        alpha2 = atan(abs((self.R_B-x_hat)/y_hat))
        lbound = 270 - alpha1
        rbound = 270 + alpha2
        if psip_k>=lbound and psip_k<=rbound:
            r_out = x_hat - y_hat*tan(270-psip_k)
            tau1 = math.sqrt(y_hat**2 + (x_hat-r_out)**2)
            return tau1
        return None

    def get_tau2(self,r_k,phi_k,psi_k):
        '''If case 2 condition is satisfied, return `tau`, otherwise return `None`.'''
        alpha0 = self.beamwidth/2 + phi_k
        x_hat = r_k*cos(alpha0)
        y_hat = r_k*sin(alpha0)
        psip_k = self.beamwidth/2 - self.theta_k + psi_k
        while psip_k<0: psip_k+=360
        while psip_k>360: psip_k-=360
        alpha1 = atan(abs(x_hat/y_hat)) if y_hat!=0 else 90
        alpha2 = atan(abs((self.R_B-x_hat)/y_hat)) if y_hat!=0 else 90
        lbound = 270 - alpha1
        rbound = 270 + alpha2
        if psip_k>=lbound and psip_k<=rbound:
            r_out = x_hat - y_hat*tan(270-psip_k)
            tau2 = math.sqrt(y_hat**2 + (x_hat-r_out)**2)
            return tau2
        return None

    def get_tau3(self,r_k,phi_k,psi_k):
        '''If case 3 condition is satisfied, return `tau`, otherwise return `None`.'''
        gamma_k = psi_k-(self.theta_k+phi_k)
        d1 = -r_k*cos(gamma_k) + math.sqrt((r_k*cos(gamma_k))**2-(r_k**2-self.R_B**2))
        d2 = -r_k*cos(gamma_k) - math.sqrt((r_k*cos(gamma_k))**2-(r_k**2-self.R_B**2))
        return max(d1,d2)

    def get(self,r_k,phi_k,psi_k):
        tau = self.get_tau1(r_k,phi_k,psi_k)
        if tau is not None: # if case 1?
            return (1, tau)
        tau = self.get_tau2(r_k,phi_k,psi_k)
        if tau is not None: # if case 2?
            return (2, tau)
        tau = self.get_tau3(r_k,phi_k,psi_k)
        if tau is not None: # if case 3?
            return (3, tau)
        return (0, None) # if this point is reached, the formula is wrong!

def test6():

    ## beam setup & plot
    theta_k = b_k.theta = 80
    beamwidth = b_k.Width
    R_B = b_k.Range
    (x_k,y_k) = b_k.center.xy()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(patches.Wedge((x_k,y_k),R_B,theta_k+start_angle_k,theta_k+end_angle_k,color="g",alpha=0.2))

    ## vehicle starting point, P2
    r_k = 60
    phi_k = 10
    psi_k = 100 # direction
    l = 70

    pt2 = PointXY()
    pt2.from_polar(r_k,phi_k,(x_k,y_k),theta_k)

    ## Use J(.) to find which `case` and calculate the corresponding travelling distance `tau`
    j = J((x_k,y_k),theta_k,R_B,beamwidth)
    for psi_k in range(0,360,10):
        case,tau = j.get(r_k,phi_k,psi_k)
        if case==0:
            print("Test failed, something is wrong in the derivation!!!")
            return
        (xm,ym) = TF.move_towards(pt2.x,pt2.y,tau,psi_k)
        ax1.plot([pt2.x,xm],[pt2.y,ym],{1:"r",2:"g",3:"b"}[case])
    ax1.plot(pt2.x,pt2.y,'yo')

    ## test description, you may adjust the position & font size of this message
    plt.text(x=0, y=-80,  
                    s="Show the trajectories of a vehicle with\n"
                     f"various directions. The trajectory is shown in\n"
                     f"RED, GREEN, BLUE for cases 1, 2, 3 respectively.\n"
                     f"Based on the formula, it should pick the right case\n"
                     f"to draw the line with the appropriate color and length.",
                     ha='center', wrap=True, fontsize=8)
    plt.axis([-150, 150, -150, 150])
    plt.show()

###################################################################
## test 7: Test the area calculations
##         Defined `Infinitesimal` class
###################################################################
class Infinitesimal:
    '''This is an infinitesimal object that can be iterated over
    the given `step_size` from `lower_bound` to `upper_bound`.'''
    def __init__(self, lower_bound, upper_bound, step_size):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step_size = step_size
        self.sample_size = int((upper_bound-lower_bound)/step_size)

    def delta(self):
        return self.step_size

    def __iter__(self):
        self.val = self.lower_bound
        return self

    def __next__(self):
        this_val = self.val
        self.val += self.step_size
        if this_val>=self.upper_bound:
            raise StopIteration
        else:
            return this_val

    def arange(self):
        return np.arange(self.lower_bound, self.upper_bound, self.step_size)

def test7():

    ## beam setup
    theta_k = b_k.theta 
    beamwidth = b_k.Width
    R_B = b_k.Range
    (x_k,y_k) = b_k.center.xy()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(patches.Wedge((x_k,y_k),R_B,theta_k+start_angle_k,theta_k+end_angle_k,color="g",alpha=0.2))

    ## monte carlo setup
    D_k = 25
    x_range = (-50,70) # the range must be wider enough to fully cover the beam
    y_range = (-10,85)
    ax1.add_patch(patches.Rectangle((x_range[0],y_range[0]), 
                            x_range[1]-x_range[0], y_range[1]-y_range[0], 
                            linewidth=1, edgecolor='r', facecolor='none'))
    ax1.plot([x_range[0],x_range[1]],[D_k,D_k],'b')

    ## calculate area by formula
    print("Calculating, please wait...",end="")
    A_B = 0.5*(R_B**2)*(beamwidth*math.pi/180) # area
    area1 = A_B

    r = Infinitesimal(0,R_B,0.1)
    phi = Infinitesimal(-beamwidth/2,beamwidth/2,0.1)

    ## calculate area by looping polar coordinate
    area2 = 0
    for r_k in r:
        for phi_k in phi:
            area2 += r_k*(phi.delta()*math.pi/180)*r.delta()

    ## calculate area by looping polar coordinate with D_k condition
    area3 = 0
    phi_k = phi.lower_bound
    while phi_k<phi.upper_bound:
        r_k = min(R_B, D_k/sin(theta_k+phi_k))
        while r_k<r.upper_bound:
            area3 += r_k*(phi.delta()*math.pi/180)*r.delta()
            r_k += r.step_size
        phi_k += phi.step_size

    ## use monte carlo to find area with D_k condition
    inside = 0
    sample_size = 50000
    for i in range(sample_size):
        x = random.random()*(x_range[1]-x_range[0])+x_range[0]
        y = random.random()*(y_range[1]-y_range[0])+y_range[0]
        pt = PointXY(x,y)
        if pt.is_inside(b_k):
            if y>=D_k:
                inside += 1
    area4 = inside/sample_size
    area4 *= (x_range[1]-x_range[0]) * (y_range[1]-y_range[0])
    print("done")

    ## test description, you may adjust the position & font size of this message
    plt.text(x=-140, y=-140, 
                    s="This test calculates the area of the beam\n"
                      "using various methods:\n"
                     f"- Beam area = {area1:1.4f} (standard formula)\n"
                     f"- Beam area = {area2:1.4f} (our formula)\n"
                     f"Set D_k = {D_k}\n"
                     f"- Highway region only = {area3:1.4f} (our formula)\n"
                     f"- Highway region only = {area4:1.4f} (using monte carlo)\n"
                     f"NOTE1: The above pairs must match.\n"
                     f"NOTE2: the red box must cover the entire beam",
                     ha='left', wrap=True, fontsize=8)
    plt.axis([-150, 150, -150, 150])
    plt.show()


###################################################################
## test 8: Calculate F*(l) using J*()
###################################################################
def test8():

    ## beam setup
    theta_k = b_k.theta = 90 # or test with any other value, e.g. 150, 30, 90
    beamwidth = b_k.Width
    R_B = b_k.Range
    (x_k,y_k) = b_k.center.xy()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(patches.Wedge((x_k,y_k),R_B,theta_k+start_angle_k,theta_k+end_angle_k,color="g",alpha=0.2))

    ## monte carlo setup
    D_k = 25
    x_range = (-60,60) # or use this (-60-30,60+50) if wider range is needed to cover the entire beam 
    y_range = (-10,85)
    ax1.add_patch(patches.Rectangle((x_range[0],y_range[0]), 
                            x_range[1]-x_range[0], y_range[1]-y_range[0], 
                            linewidth=1, edgecolor='r', facecolor='none'))
    ax1.plot([x_range[0],x_range[1]],[D_k,D_k],'b')

    r = Infinitesimal(0,R_B,0.5)
    phi = Infinitesimal(-beamwidth/2,beamwidth/2,0.2)

    ## calculate area by looping polar coordinate with D_k condition
    area = 0
    for phi_k in phi.arange():
        r_k = min(R_B, D_k/sin(theta_k+phi_k)) if (theta_k+phi_k)!=0 else R_B
        while r_k<r.upper_bound:
            area += r_k*(phi.delta()*math.pi/180)*r.delta()
            r_k += r.step_size

    ## calculate F*(l) using J*(), the highway case
    def F_star(l):
        nominator = 0
        denominator = area
        for phi_k in phi.arange():
            r_k = min(R_B, D_k/sin(theta_k+phi_k)) if (theta_k+phi_k)!=0 else R_B
            while r_k<r.upper_bound:
                j_0 = 1 if j.get(r_k,phi_k,0)[1]<=l else 0
                j_pi = 1 if j.get(r_k,phi_k,180)[1]<=l else 0
                j_star = 0.5*(j_0+j_pi)
                nominator += r_k*j_star*(phi.delta()*math.pi/180)*r.delta()
                r_k += r.step_size
        return nominator/denominator

    ## calculate F*(l) for a single l
    l = 40
    j = J((x_k,y_k),theta_k,R_B,beamwidth)
    prob_calc = F_star(l)

    ## test with monte carlo simulation
    sample_size = 1000
    count = 0
    sum_prob = 0
    while True:
        x = random.random()*(x_range[1]-x_range[0])+x_range[0]
        y = random.random()*(y_range[1]-y_range[0])+y_range[0]
        if y>=D_k:
            if PointXY(x,y).is_inside(b_k):
                if not PointXY(x-l,y).is_inside(b_k): sum_prob+=0.5
                if not PointXY(x+l,y).is_inside(b_k): sum_prob+=0.5
                count += 1
                if count==sample_size: break
    prob_sim = sum_prob/sample_size

    ## test description, you may adjust the position & font size of this message
    plt.text(x=-140, y=-120, 
                    s="This test calculates F*(l):\n"
                     f"- F*(l={l:1.1f})={prob_calc:1.4f}, our numerical\n"
                     f"- F*(l={l:1.1f})={prob_sim:1.4f}, simulated\n"
                     f"NOTE1: the above must be similar\n"
                     f"NOTE2: the red box must cover the entire beam",
                     ha='left', wrap=True, fontsize=8)
    plt.axis([-150, 150, -150, 150])
    plt.show()

    ## calculate F*(l) over a range of l (if needed)
    ## - controlled by `ls_loop` flag, set to `True` to run the loop, 
    ##   or `False` to skip the calculation
    is_loop = True
    #is_loop = False
    if is_loop: # loop?
        print(" l,   F*(l)")
        print("---,  -----")
        for l in range(0,int(R_B*1.2),2):
            prob = F_star(l)
            print(f"{l:1.1f}, {prob:1.4f}")


###################################################################
## test 9: Calculate d_zeta, service distance (with interference)
##         Defined P_I(.)
###################################################################
def P_I(r_i,phi_i,p_i,R_B=R_B,beamwidth=beamwidth):
    if r_i<=R_B and phi_i>=-beamwidth/2 and phi_i<=beamwidth/2:
        return 1-p_i
    else:
        return 1

def test9():

    ## beam setup
    theta_k = b_k.theta = 90
    beamwidth = b_k.Width
    R_B = b_k.Range
    (x_k,y_k) = b_k.center.xy()

    ## interference beam
    b_1 = Beam(x_k,y_k,theta_k-40)
    b_2 = Beam(80,y_k,130-15)
    p_active = 0.5  # prob that these beams are active when observed

    ## plot the scenario
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    for bm,color in [(b_k,"r"),(b_1,"g"),(b_2,"b")]:
        ax1.add_patch(patches.Wedge(bm.center.xy(),bm.Range,
                                    bm.theta+bm.start_angle,
                                    bm.theta+bm.end_angle,color=color,alpha=0.2))

    ## vehicle & its movement
    r_k = 60
    phi_k = 0
    psi_k = 340 # direction
    l = 70

    ## plot the vehicle & its movement
    pt = PointXY()
    pt.from_polar(r_k,phi_k,(x_k,y_k),theta_k)
    ax1.plot(pt.x,pt.y,'ro')

    j = J((x_k,y_k),theta_k,R_B,beamwidth)
    (_,tau) = j.get(r_k, phi_k, psi_k)
    (xm,ym) = TF.move_towards(pt.x,pt.y,tau,psi_k)
    ax1.plot([pt.x,xm],[pt.y,ym],"r")

    ## create a loop to iterate over the trajectory
    d_integration = Infinitesimal(lower_bound=0, upper_bound=tau, step_size=0.1)
    d_zeta = 0
    for d in d_integration.arange():
        ## calculate r_{i|k} & phi_{i|k}
        gamma_k = psi_k-(theta_k+phi_k)
        r_d = math.sqrt(r_k**2 + d**2 + 2*r_k*d*cos(gamma_k))
        phi_d = phi_k + atan2(d*sin(gamma_k),r_k+d*cos(gamma_k))

        ## calculate \prod_i P_I(r_{i|k},phi_{i|k},p_i)
        prod_P_I = 1
        for b_i in [b_1,b_2]:
            (r_i,phi_i) = pt_i_from_k(r_d,phi_d, x_k,y_k,theta_k,
                                                 b_i.x(),b_i.y(),b_i.theta)
            prod_P_I *= P_I(r_i,phi_i,p_active)
        
        ## calculate d_{zeta} = \int \prod_i P_I(r_{i|k},phi_{i|k},p_i) dd
        d_zeta += prod_P_I * d_integration.delta()

    ## do simulation to check with the numerical results
    d, d_sim = 0, 0
    delta_d = 0.1
    (dx,dy) = (delta_d*cos(psi_k),delta_d*sin(psi_k))
    pt_vehicle = pt.clone()
    while True:
        pt_vehicle.x += dx
        pt_vehicle.y += dy
        interference = False
        for b_i in [b_1,b_2]:
            if pt_vehicle.is_inside(b_i) and random.random()<p_active:
                interference = True
        if not interference:
            d_sim += delta_d
        if not pt_vehicle.is_inside(b_k): 
            break # departing the beam, break the loop & done
        d += delta_d

    ## print outcomes
    print(f"tau = {tau}")
    print(f"d_zeta = {d_zeta}")
    print(f"d_sim = {d_sim}")

    ## test description, you may adjust the position & font size of this message
    plt.text(x=-120, y=-120, 
                    s="This test calculates d_zeta:\n"
                     f"- tau={tau:1.4f}, distance travelled\n"
                     f"- p_active={p_active:1.1f}, beam active prob\n"
                     f"- d_zeta={d_zeta:1.1f}, from our formula\n"
                     f"- d_sim={d_sim:1.1f}, sim result\n"
                     f"NOTE: The above should match closely.",
                     ha='left', wrap=True, fontsize=8)
    plt.axis([-150, 150, -150, 150])
    plt.show()


###################################################################
## test 10: Calculate \breve{F}*(l) for a selected beam in 
##          the highway scenario.
## Note: In this version, vehicles can appear at any point on 
##       the highway
###################################################################
def test10():

    ## all 18 beams at 6 sites for the highway scenario
    ##   (3)   (4)     (5)    <-- north-side beams
    ##   ==================   <-- highway edge
    ##     > >     >          <-- 3 lanes (east moving)
    ##    - - - - - - - -     <-- traffic flow separator
    ##      <     <    <      <-- 3 lanes (west moving)
    ##   ==================   <-- highway edge
    ##   (0)    (1)     (2)   <-- south-side beams
    D_k = 28  # distance from beam origin to the edge of the highway
    D_mid = 0 # traffic flow separation
    H = 24    # width of highway
    beam = [[Beam(100,-40,90+60),Beam(100,-40,90),Beam(100,-40,90-60)],  # 0 (and 3 beams per site)
            [Beam(220,-40,90+60),Beam(220,-40,90),Beam(220,-40,90-60)],  # 1
            [Beam(360,-40,90+60),Beam(360,-40,90),Beam(360,-40,90-60)],  # 2
            [Beam(90,40,270-60),Beam(90,40,270),Beam(90,40,270+60)],     # 3
            [Beam(210,40,270-60),Beam(210,40,270),Beam(210,40,270+60)],  # 4
            [Beam(340,40,270-60),Beam(340,40,270),Beam(340,40,270+60)]]  # 5
    beam_list = [b for sublist in beam for b in sublist]

    ## manually configure the picked beam and specify neighbouring beams
    ## that may cause interference. If non-interfering beams are also
    ## included as neighbouring beams, it won't generate different results, 
    ## but will simply make the calculation run longer unnecessarily
    beam_to_test = beam[1][1]
    beam_neighbours = {}
    beam_neighbours[beam_to_test] = [beam[4][0],beam[4][1],beam[4][2]]

    ## scenario setup
    p_active = 0.8
    l = 30
    print(f"Calculating F_breve(l)...")
    print(f"Note: The calculation is numerically intensive, please be patient.")
    print(f"l={l}; p_active={p_active}")

    ## beam setup
    theta_k = beam_to_test.theta
    beamwidth = beam_to_test.Width
    R_B = beam_to_test.Range
    (x_k,y_k) = beam_to_test.center.xy()

    ## plot the scenario
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    color = "g"
    class Box:
        def __init__(self):
            (self.left,self.top,self.right,self.bottom) = (0,0,0,0)
        def xylim(self,x,y,xbuf,ybuf):
            if x-xbuf<self.left: self.left=x-xbuf
            if x+xbuf>self.right: self.right=x+xbuf
            if y+ybuf>self.top: self.top=y+ybuf
            if y-ybuf<self.bottom: self.bottom=y-ybuf
    box = Box()
    for bm in beam_neighbours[beam_to_test]:
        ax1.add_patch(patches.Wedge(bm.center.xy(),bm.Range,
                                    bm.theta+bm.start_angle,
                                    bm.theta+bm.end_angle,color=color,alpha=0.2))
        box.xylim(bm.x(),bm.y(),int(1.2*bm.Range),int(0.5*bm.Range))
        color = "g" if color=="b" else "b"
    bm = beam_to_test
    box.xylim(bm.x(),bm.y(),int(1.2*bm.Range),int(0.5*bm.Range))
    ax1.add_patch(patches.Wedge(bm.center.xy(),bm.Range,
                                bm.theta+bm.start_angle,
                                bm.theta+bm.end_angle,color="r",alpha=0.2))

    ## add a box to indicate highway section of the beam
    x_range = (170,270)       # the x range must fully cover the beam
    y_range = (bm.y()+D_k,12) # set y range so that it covers from D_k to the far edge of the beam
    #y_range = (-42,bm.y()-D_k) # use this for north site beam
    ax1.add_patch(patches.Rectangle((x_range[0],y_range[0]), 
                            x_range[1]-x_range[0], y_range[1]-y_range[0], 
                            linewidth=1, edgecolor='r', facecolor='none'))

    ## add arrow to indicate D_k
    ax1.add_patch(patches.Arrow(bm.x(), bm.y(), 0, D_k, width=3.0)) # for south site, use this
    #ax1.add_patch(patches.Arrow(bm.x(), bm.y(), 0, -D_k, width=3.0)) # for norht site, use this

    ## simulation
    sample_size = 1000
    count_all = 0
    count_prob = 0
    delta_d = 0.5
    while count_all<sample_size:
        x = random.random()*(x_range[1]-x_range[0])+x_range[0]
        y = random.random()*(y_range[1]-y_range[0])+y_range[0]
        #dx = random.choice([delta_d,-delta_d])
        dx = delta_d if y>D_mid else -delta_d # use proper direction based on lane
        d_sim = 0
        pt_vehicle = PointXY(x,y)
        if not pt_vehicle.is_inside(beam_to_test): continue
        while True:
            pt_vehicle.x += dx
            interference = False
            for b_i in beam_neighbours[beam_to_test]:
                if pt_vehicle.is_inside(b_i) and random.random()<p_active:
                    interference = True
            if not interference:
                d_sim += delta_d
            if not pt_vehicle.is_inside(beam_to_test):
                break # departing the beam, break the loop & done
        if d_sim<=l: count_prob += 1
        count_all += 1

    F_l_sim = count_prob/count_all
    print(f"F_sim(l={l}) = {F_l_sim}")

    ## calculate \breve{F}*(l) = \int r_k \breve{J}star() / area
    ## where \breve{J}star() = 1 if d_zeta<=l else 0
    ##       d_zeta = \int_0^{tau} P_I()
    ## step 1: calculate denominator (area of the sector on the highway)
    ## step 2: calculate tau, P_I(), then d_zeta
    ## step 3: calculate \breve{J}star(), then nominator
    ## step 4: finally, calculate \breve{F}*(l)

    ## (step 1) calculate area by looping polar coordinate with D_k condition
    def beam_area_on_highway(theta_k,D_k,R_B=R_B,beamwidth=beamwidth):
        area = 0
        r = Infinitesimal(0, R_B, step_size=0.5)
        phi = Infinitesimal(-beamwidth/2, beamwidth/2, step_size=0.2)
        for phi_k in phi.arange():
            R_n = min(R_B, D_k/sin(theta_k+phi_k)) if (theta_k+phi_k)!=0 else R_B
            R_f = min(R_B, (D_k+H)/sin(theta_k+phi_k)) if (theta_k+phi_k)!=0 else R_B
            r_k = R_n
            while r_k<R_f:
                area += r_k*(phi.delta()*math.pi/180)*r.delta()
                r_k += r.step_size
        return area

    ## (step 2) calculate tau, P_I(), then d_zeta
    j = J((x_k,y_k),theta_k,R_B,beamwidth)
    def d_zeta(r_k, phi_k, psi_k):
        (_,tau) = j.get(r_k, phi_k, psi_k)
        d_integration = Infinitesimal(lower_bound=0, upper_bound=tau, step_size=0.5)
        d_zeta_val = 0
        for d in d_integration.arange():
            ## calculate r_{i|k} & phi_{i|k}
            gamma_k = psi_k-(theta_k+phi_k)
            r_d = math.sqrt(r_k**2 + d**2 + 2*r_k*d*cos(gamma_k))
            phi_d = phi_k + atan2(d*sin(gamma_k),r_k+d*cos(gamma_k))

            ## calculate \prod_i P_I(r_{i|k},phi_{i|k},p_i)
            prod_P_I = 1
            for b_i in beam_neighbours[beam_to_test]:
                (r_i,phi_i) = pt_i_from_k(r_d,phi_d, x_k,y_k,theta_k,
                                                    b_i.x(),b_i.y(),b_i.theta)
                ## need to condition phi_i based on b_i.theta
                while phi_i>b_i.theta and phi_i-360>b_i.theta: phi_i-=360
                while phi_i<b_i.theta and phi_i+360<b_i.theta: phi_i+=360

                prod_P_I *= P_I(r_i,phi_i,p_active)
            
            ## calculate d_{zeta} = \int \prod_i P_I(r_{i|k},phi_{i|k},p_i) dd
            d_zeta_val += prod_P_I * d_integration.delta()

        return d_zeta_val

    ## (step 3) calculate \breve{J}star()
    def J_breve_star(r_k,phi_k,l,psi_k):
        return 1 if d_zeta(r_k, phi_k, psi_k)<=l else 0

    ## (step 4) calculate \breve{F}*(l)
    def F_breve_star(l, theta_k=theta_k, D_k=D_k, D_mid=D_mid, R_B=R_B, 
                        beamwidth=beamwidth, bm_y=beam_to_test.y()):
        nominator = 0
        denominator = beam_area_on_highway(theta_k, D_k)
        r = Infinitesimal(0, R_B, step_size=0.5)
        phi = Infinitesimal(-beamwidth/2, beamwidth/2, step_size=0.5)
        for phi_k in phi.arange():
            R_n = min(R_B, D_k/sin(theta_k+phi_k)) if (theta_k+phi_k)!=0 else R_B
            R_f = min(R_B, (D_k+H)/sin(theta_k+phi_k)) if (theta_k+phi_k)!=0 else R_B
            r_k = R_n
            while r_k<R_f:
                y_loc = bm_y + r_k*sin(theta_k+phi_k)
                psi_k = 0 if y_loc>D_mid else 180
                nominator += r_k*J_breve_star(r_k,phi_k,l,psi_k)*(math.radians(phi.delta()))*r.delta()
                r_k += r.step_size
        return nominator/denominator

    ## do calculation
    F_l = F_breve_star(l)
    print(f"F(l={l}) = {F_l}")

    ## show outcome
    plt.text(box.left,box.top+6, "Check for interference setup:",
                                 ha='left', wrap=True, fontsize=8)
    plt.text(x=box.left, y=box.top-80,
                            s=f" For l={l}, p_active={p_active}, D_k={D_k}\n"
                              f" - F_sim = {F_l_sim:1.3f}\n"
                              f" - F_num = {F_l:1.3f}\n\n"
                              f"NOTE: The box must cover\n"
                              f" the highway section of the\n"
                              f" ORANGE beam which depends\n"
                              f" on D_k.",
                              ha='left', wrap=True, fontsize=8)
    plt.axis([box.left, box.right, box.bottom, box.top])
    plt.show()

    ## calculate \breve{F}*(l) over a range of l (if needed)
    ## - controlled by `ls_loop` flag, set to `True` to run the loop, 
    ##   or `False` to skip the calculation
    is_loop = True
    #is_loop = False
    if is_loop: # loop?
        print("Running the loop now, it may take a while to complete.")
        print("Use [Ctlr]+[C] to stop the calculation.")
        print(" l,   \breve{F}*(l)")
        print("---,  -------------")
        for l in range(0,int(R_B*1.2),2):
            prob = F_breve_star(l)
            print(f"{l:1.1f}, {prob:1.4f}")


###################################################################
## test 11: Calculate upper bound of \breve{F}*(l) for a selected 
##          beam in the highway scenario.
## Note: In this version, vehicles will always starts from a specific
##       optimal location on the lane, and move east or west following 
##       the lane direction
###################################################################
def test11():

    ## all 18 beams at 6 sites for the highway scenario
    ##   (3)   (4)     (5)    <-- north-side beams
    ##   ==================   <-- highway edge
    ##     > >     >          <-- 3 lanes (east moving)
    ##    - - - - - - - -     <-- traffic flow separator
    ##      <     <    <      <-- 3 lanes (west moving)
    ##   ==================   <-- highway edge
    ##   (0)    (1)     (2)   <-- south-side beams
    D_k = 28 # distance from beam origin to the edge of the highway
    D_mid = 0 # traffic flow separation
    beam = [[Beam(100,-40,90+60),Beam(100,-40,90),Beam(100,-40,90-60)],  # 0 (and 3 beams per site)
            [Beam(220,-40,90+60),Beam(220,-40,90),Beam(220,-40,90-60)],  # 1
            [Beam(360,-40,90+60),Beam(360,-40,90),Beam(360,-40,90-60)],  # 2
            [Beam(90,40,270-60),Beam(90,40,270),Beam(90,40,270+60)],     # 3
            [Beam(210,40,270-60),Beam(210,40,270),Beam(210,40,270+60)],  # 4
            [Beam(340,40,270-60),Beam(340,40,270),Beam(340,40,270+60)]]  # 5
    beam_list = [b for sublist in beam for b in sublist]

    ## manually configure the picked beam and specify neighbouring beams
    ## that may cause interference. If non-interfering beams are also
    ## included as neighbouring beams, it won't generate different results, 
    ## but will simply make the calculation run longer unnecessarily
    beam_to_test = beam[1][1] # mid south BS, north pointing beam
    beam_neighbours = {}
    beam_neighbours[beam_to_test] = [beam[4][0],beam[4][1],beam[4][2]]
    lane_e2w = [30, 34, 38] # distance from south beam origin to each west moving lane
    lane_w2e = [42, 46, 50] # distance from south beam origin to each east moving lane 

    ## scenario setup
    p_active = 0.8
    l = 30
    print(f"Calculating upper bound of F_breve(l)...")
    print(f"l={l}; p_active={p_active}")

    ## beam setup
    theta_k = beam_to_test.theta
    beamwidth = beam_to_test.Width
    R_B = beam_to_test.Range
    (x_k,y_k) = beam_to_test.center.xy()

    ## determine mathcal{L_k} for this beam
    ## - here, apart from r_start & phi_start, we also
    ##   include psi_start, tau, xy for easy calculation
    j = J((x_k,y_k),theta_k,R_B,beamwidth)
    mathcal_L_k = []
    small_epsilon = 0.001 # this is needed to avoid divide-by-zero
    for lane_y in lane_e2w:
        r_start = lane_y/sin(theta_k-beamwidth/2)
        phi_start = -beamwidth/2 + small_epsilon
        psi_start = 180
        _,tau = j.get(r_start,phi_start,psi_start)
        xy = PointXY()
        xy.from_polar(r_start,phi_start,(x_k,y_k),theta_k)
        start_loc = (r_start,phi_start,psi_start,-tau,xy)
        mathcal_L_k.append(start_loc)
    for lane_y in lane_w2e:
        r_start = lane_y/sin(theta_k+beamwidth/2)
        phi_start = beamwidth/2 - small_epsilon
        psi_start = 0
        _,tau = j.get(r_start,phi_start,psi_start)
        xy = PointXY()
        xy.from_polar(r_start,phi_start,(x_k,y_k),theta_k)
        start_loc = (r_start,phi_start,psi_start,tau,xy)
        mathcal_L_k.append(start_loc)

    ## plot the scenario
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    color = "g"
    class Box:
        def __init__(self):
            (self.left,self.top,self.right,self.bottom) = (0,0,0,0)
        def xylim(self,x,y,xbuf,ybuf):
            if x-xbuf<self.left: self.left=x-xbuf
            if x+xbuf>self.right: self.right=x+xbuf
            if y+ybuf>self.top: self.top=y+ybuf
            if y-ybuf<self.bottom: self.bottom=y-ybuf
    box = Box()
    for bm in beam_neighbours[beam_to_test]:
        ax1.add_patch(patches.Wedge(bm.center.xy(),bm.Range,
                                    bm.theta+bm.start_angle,
                                    bm.theta+bm.end_angle,color=color,alpha=0.2))
        box.xylim(bm.x(),bm.y(),int(1.2*bm.Range),int(0.5*bm.Range))
        color = "g" if color=="b" else "b"
    bm = beam_to_test
    box.xylim(bm.x(),bm.y(),int(1.2*bm.Range),int(0.5*bm.Range))
    ax1.add_patch(patches.Wedge(bm.center.xy(),bm.Range,
                                bm.theta+bm.start_angle,
                                bm.theta+bm.end_angle,color="r",alpha=0.2))

    ## add arrow to indicate lanes
    for lane in mathcal_L_k:
        (r_start,phi_start,psi_start,tau,xy) = lane
        ax1.add_patch(patches.Arrow(xy.x, xy.y, tau, 0, width=2.0, color='r'))

    ## simulation
    def sim(l):
        sample_size = 1000
        count_all = 0
        count_prob = 0
        delta_d = 0.5
        while count_all<sample_size:
            (_,_,_,tau,xy) = random.choice(mathcal_L_k)
            x = xy.x + random.random()*tau
            y = xy.y
            #dx = random.choice([delta_d,-delta_d])
            dx = delta_d if y>D_mid else -delta_d # use proper direction based on lane
            d_sim = 0
            pt_vehicle = PointXY(x,y)
            if not pt_vehicle.is_inside(beam_to_test): continue
            while True:
                pt_vehicle.x += dx
                interference = False
                for b_i in beam_neighbours[beam_to_test]:
                    if pt_vehicle.is_inside(b_i) and random.random()<p_active:
                        interference = True
                if not interference:
                    d_sim += delta_d
                if not pt_vehicle.is_inside(beam_to_test):
                    break # departing the beam, break the loop & done
            if d_sim<=l: count_prob += 1
            count_all += 1

        F_l_sim = count_prob/count_all
        return F_l_sim

    F_l_sim = sim(l)
    print(f"F_sim(l={l}) = {F_l_sim}")

    ## calculate \hat\breve{F}*(l) = \sum_(r_s,phi_s) \breve{J}star() / number_lane
    ## where \breve{J}star() = 1 if d_zeta<=l else 0
    ##       d_zeta = \int_0^{tau} P_I()
    ## step 1: calculate denominator (i.e. number of lanes)
    ## step 2: calculate tau, P_I(), then d_zeta
    ## step 3: calculate \breve{J}star(), then nominator
    ## step 4: finally, calculate \hat\breve{F}*(l)

    ## (step 1) number of lanes
    def number_of_lanes(mathcal_L_k):
        return len(mathcal_L_k)

    ## (step 2) calculate tau, P_I(), then d_zeta
    j = J((x_k,y_k),theta_k,R_B,beamwidth)
    def d_zeta(r_k, phi_k, psi_k):
        (_,tau) = j.get(r_k, phi_k, psi_k)
        d_integration = Infinitesimal(lower_bound=0, upper_bound=tau, step_size=0.5)
        d_zeta_val = 0
        for d in d_integration.arange():
            ## calculate r_{i|k} & phi_{i|k}
            gamma_k = psi_k-(theta_k+phi_k)
            r_d = math.sqrt(r_k**2 + d**2 + 2*r_k*d*cos(gamma_k))
            phi_d = phi_k + atan2(d*sin(gamma_k),r_k+d*cos(gamma_k))

            ## calculate \prod_i P_I(r_{i|k},phi_{i|k},p_i)
            prod_P_I = 1
            for b_i in beam_neighbours[beam_to_test]:
                (r_i,phi_i) = pt_i_from_k(r_d,phi_d, x_k,y_k,theta_k,
                                                    b_i.x(),b_i.y(),b_i.theta)
                ## need to condition phi_i based on b_i.theta
                while phi_i>b_i.theta and phi_i-360>b_i.theta: phi_i-=360
                while phi_i<b_i.theta and phi_i+360<b_i.theta: phi_i+=360

                prod_P_I *= P_I(r_i,phi_i,p_active)
            
            ## calculate d_{zeta} = \int \prod_i P_I(r_{i|k},phi_{i|k},p_i) dd
            d_zeta_val += prod_P_I * d_integration.delta()

        return d_zeta_val

    ## (step 3) calculate \breve{J}star()
    def J_breve_star(r_k,phi_k,l,psi_k):
        return 1 if d_zeta(r_k, phi_k, psi_k)<=l else 0

    ## (step 4) calculate \hat\breve{F}*(l)
    def hat_F_breve_star(l, theta_k=theta_k, D_k=D_k, D_mid=D_mid, R_B=R_B, 
                        beamwidth=beamwidth, bm_y=beam_to_test.y()):
        nominator = 0
        denominator = number_of_lanes(mathcal_L_k)
        for lane in mathcal_L_k:
            (r_start,phi_start,psi_start,tau,xy) = lane
            nominator += J_breve_star(r_start,phi_start,l,psi_start)
        return nominator/denominator

    ## do calculation
    F_l = hat_F_breve_star(l)
    print(f"F(l={l}) = {F_l}")

    ## show outcome
    plt.text(box.left,box.top+6, "Check for interference setup:",
                                 ha='left', wrap=True, fontsize=8)
    plt.text(x=box.left, y=box.top-60,
                            s=f" For l={l}, p_active={p_active}, D_k={D_k}\n"
                              f" - F_sim = {F_l_sim:1.3f}\n"
                              f" - F_num = {F_l:1.3f}\n\n"
                              f"NOTE: The arrows are the \n"
                              f" lanes and the traffic direction.\n",
                              ha='left', wrap=True, fontsize=8)
    plt.axis([box.left, box.right, box.bottom, box.top])
    plt.show()

    ## calculate \hat\breve{F}*(l) over a range of l (if needed)
    ## - controlled by `ls_loop` flag, set to `True` to run the loop, 
    ##   or `False` to skip the calculation
    is_loop = True
    #is_loop = False
    if is_loop: # loop?
        print("Running the loop now, it may take a while to complete.")
        print("Use [Ctlr]+[C] to stop the calculation.")
        print(" l,   \breve{F}*(l)")
        print("---,  -------------")
        for l in range(0,int(R_B*1.2),2):
            #prob = hat_F_breve_star(l)
            prob = sim(l)
            print(f"{l:1.1f}, {prob:1.4f}")


###################################################################
## Launcher
## - Pick one of the following tests to run
###################################################################

test1() # draw a horizontal line in the beam (formula for the highway edge)
#test2() # polar location relative to b_i, translated from b_k
#test3() # trajectory of the movement in a beam
#test4() # transformation of a beam for case 1 and case 2
#test5() # transformation of a beam for case 3

## ---------------------------------------------------------------
## tau   = distance travelled
## J(.)  = prob that service distance is shorter than $l$
## J*(.) = J(.) for highway scenario
## F(l)  = CDF, service distance is shorter than $l$
## F*(l) = CDF for highway scenario
## d_zeta   = distance travelled discounting interference area
## \breve{J}*(.) = highway scenario, prob d_zeta <= $l$
## \breve{F}*(l) = CDF of d_zeta <= l
## ---------------------------------------------------------------

#test6() # J(.) and tau calculation
#test7() # calculation of beam areas
#test8() # calculation of F*(l) using J*(.), ie. highway scenario
#test9() # calculation of d_zeta 
#test10() # calculation of \breve{F}*(l)
#test11() # calculation of upper bound \breve{F}*(l)

