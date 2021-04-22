#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 2021

@author: Vladimir Krajnak
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

#constants
#https://physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable1.htm
mH = 1.00794 #Da

D1 = 4.7466
D3 = 1.9668
Re = 1.40083
alpha = 1.04435
beta = 1.000122
delta = 28.2
epsilon = -17.5
kappa = 0.6
lam = 0.65

au = 27.211386 #eV
Emin = -4.746599999999999
Rs = 1.700828745264


def Hamiltonian(y):
    """
    Parameters
    ----------
    y : ndarray, shape(6,)
        y=[r1,pr1,r2,pr2,theta, ptheta].

    Returns
    -------
    H : float
        Total energy for hydrogen exchange.
    """

    T = y[1]*y[1]+y[3]*y[3] - y[1]*y[3]*np.cos(y[4]) \
        + y[5]*np.sin(y[4])*(y[1]/y[0]+y[3]/y[2]) \
        + y[5]*y[5]*(1/(y[0]*y[0]) + 1/(y[2]*y[2]) +np.cos(y[4])/(y[0]*y[2]) )
    return T/mH+pot(y[::2])

def potential(conf):
    """
    Parameters
    ----------
    conf : ndarray, shape(3,)
        conf=[r1,r2,theta].

    Returns
    -------
    pot : float [eV]
        Potential energy for hydrogen exchange (Porter Karplus, JCP 40 (1964)).
        By minimal energy we understand potential(Re,infty,0) in accordance with existing collinear work.
        Davis, JCP 86 (1987) states the energy of saddle point (1.70083,1.70083,0) of 0.396 eV.
    """
    R1 = conf[0]
    R2 = conf[1]
    R3 = np.sqrt( conf[0]**2+conf[1]**2+2*conf[1]*conf[0]*np.cos(conf[2]) )
    R = np.array([R1,R2,R3])
    Rl = np.roll(R,1)
    Rm = np.roll(R,2)

    zeta = 1+kappa*np.exp(-lam*R)
    S = (1+zeta*R+zeta*zeta*R*R/3)*np.exp(-zeta*R)
    oneE = D1*( np.exp(-2*alpha*(R-Re)) - 2*np.exp(-alpha*(R-Re)) )
    threeE = D3*( np.exp(-2*beta*(R-Re)) + 2*np.exp(-beta*(R-Re)) )
    J = 0.5*(oneE-threeE) \
        + S*S*( 0.5*(oneE+threeE) + delta*( (1+1/Rl)*np.exp(-2*Rl)+(1+1/Rm)*np.exp(-2*Rm) ) )
    Qd = 0.5*( oneE + threeE + S*S*(oneE-threeE) )
    Q = np.sum(Qd, axis=0)

    onemS123 = 1-np.prod(S, axis=0)
    S12mS22 = S[0]*S[0]-S[1]*S[1]
    S22mS32 = S[1]*S[1]-S[2]*S[2]
    S12mS32 = S[0]*S[0]-S[2]*S[2]
    J1m2 = J[0]-J[1]
    J2m3 = J[1]-J[2]
    J1m3 = J[0]-J[2]
    J123 = epsilon*np.prod(S, axis=0)
    QmJ123 = Q-J123

    c1 = onemS123**2 - 0.5*( S12mS22**2 + S22mS32**2 + S12mS32**2 )
    c2 = -QmJ123*onemS123 + 0.5*(J1m2*S12mS22 + J2m3*S22mS32 + J1m3*S12mS32)
    c3 = QmJ123*QmJ123 - 0.5*(J1m2**2 + J2m3**2 + J1m3**2)
    pot = (-c2-np.sqrt(c2*c2-c1*c3))/c1
    return pot-Emin

def dpotential(conf):
    """
    Parameters
    ----------
    conf : ndarray, shape(3,)
        conf=[r1,r2,theta].

    Returns
    -------
    pot : ndarray, shape(3,)
        Gradient of the potential energy for hydrogen exchange.
    """
    R1 = conf[0]
    R2 = conf[1]
    R3 = np.sqrt( conf[0]**2+conf[1]**2+2*conf[1]*conf[0]*np.cos(conf[2]) )
    R = np.array([R1,R2,R3])
    Rl = np.roll(R,1)
    Rm = np.roll(R,2)

    dR3dr1 = (conf[0]+conf[1]*np.cos(conf[2]))/R3
    dR3dr2 = (conf[1]+conf[0]*np.cos(conf[2]))/R3
    dR3dtheta = - conf[0]*conf[1]*np.sin(conf[2])/R3

    zeta = 1+kappa*np.exp(-lam*R)
    dzeta = -lam*kappa*np.exp(-lam*R)

    S = (1+zeta*R+zeta*zeta*R*R/3)*np.exp(-zeta*R)
    dS = (-zeta-dzeta*R)*(1+zeta*R+zeta*zeta*R*R/3)*np.exp(-zeta*R) \
        + ( zeta+dzeta*R + 2*dzeta*zeta*R*R/3 + 2*zeta*zeta*R/3 )*np.exp(-zeta*R)

    oneE = D1*( np.exp(-2*alpha*(R-Re)) - 2*np.exp(-alpha*(R-Re)) )
    doneE = -alpha*D1*( 2*np.exp(-2*alpha*(R-Re)) - 2*np.exp(-alpha*(R-Re)) )

    threeE = D3*( np.exp(-2*beta*(R-Re)) + 2*np.exp(-beta*(R-Re)) )
    dthreeE = -beta*D3*( 2*np.exp(-2*beta*(R-Re)) + 2*np.exp(-beta*(R-Re)) )

    J = 0.5*(oneE-threeE) \
        + S*S*( 0.5*(oneE+threeE) + delta*( (1+1/Rl)*np.exp(-2*Rl)+(1+1/Rm)*np.exp(-2*Rm) ) )
    dJdR = 0.5*(doneE-dthreeE) \
        + 2*dS*S*( 0.5*(oneE+threeE) + delta*( (1+1/Rl)*np.exp(-2*Rl)+(1+1/Rm)*np.exp(-2*Rm) ) ) \
        + S*S*( 0.5*(doneE+dthreeE) )
    dJdRl = S*S*delta*(-2-2/Rl-1/Rl**2)*np.exp(-2*Rl)
    dJdRm = S*S*delta*(-2-2/Rm-1/Rm**2)*np.exp(-2*Rm)

    Qd = 0.5*( oneE + threeE + S*S*(oneE-threeE) )
    dQd = 0.5*( doneE + dthreeE + 2*dS*S*(oneE-threeE) + S*S*(doneE-dthreeE) )
    Q = np.sum(Qd, axis=0)

    S123 = np.prod(S, axis=0)
    onemS123 = 1-S123
    S12mS22 = S[0]*S[0]-S[1]*S[1]
    S22mS32 = S[1]*S[1]-S[2]*S[2]
    S12mS32 = S[0]*S[0]-S[2]*S[2]
    J1m2 = J[0]-J[1]
    J2m3 = J[1]-J[2]
    J1m3 = J[0]-J[2]
    J123 = epsilon*S123
    QmJ123 = Q-J123

    c1 = onemS123**2 - 0.5*( S12mS22**2 + S22mS32**2 + S12mS32**2 )
    c2 = -QmJ123*onemS123 + 0.5*(J1m2*S12mS22 + J2m3*S22mS32 + J1m3*S12mS32)
    c3 = QmJ123*QmJ123 - 0.5*(J1m2**2 + J2m3**2 + J1m3**2)

    dS1S2S3 = dS[0]*S[1]*S[2]
    dc1dR1 = 2*onemS123*(-dS1S2S3) - 2*(S12mS22 + S12mS32)*dS[0]*S[0]
    dc2dR1 = -(dQd[0]-epsilon*dS1S2S3)*onemS123 - QmJ123*(-dS1S2S3) + (J1m2 + J1m3)*dS[0]*S[0] \
            + 0.5*((dJdR[0]-dJdRl[1])*S12mS22 + (dJdRl[1]-dJdRm[2])*S22mS32 + (dJdR[0]-dJdRm[2])*S12mS32)
    dc3dR1 = 2*QmJ123*(dQd[0]-epsilon*dS1S2S3) - (J1m2*(dJdR[0]-dJdRl[1]) + J2m3*(dJdRl[1]-dJdRm[2]) + J1m3*(dJdR[0]-dJdRm[2]))
    dpotdR1 = (-dc2dR1-0.5*(2*dc2dR1*c2-dc1dR1*c3-c1*dc3dR1)/np.sqrt(c2*c2-c1*c3))/c1 - (-c2-np.sqrt(c2*c2-c1*c3))*dc1dR1/c1**2

    S1dS2S3 = S[0]*dS[1]*S[2]
    dc1dR2 = 2*onemS123*(-S1dS2S3) - 2*( -S12mS22 + S22mS32)*dS[1]*S[1]
    dc2dR2 = -(dQd[1]-epsilon*S1dS2S3)*onemS123 - QmJ123*(-S1dS2S3) + (-J1m2 + J2m3)*dS[1]*S[1] \
            + 0.5*((dJdRm[0]-dJdR[1])*S12mS22 + (dJdR[1]-dJdRl[2])*S22mS32 + (dJdRm[0]-dJdRl[2])*S12mS32)
    dc3dR2 = 2*QmJ123*(dQd[1]-epsilon*S1dS2S3) - ((dJdRm[0]-dJdR[1])*J1m2 + (dJdR[1]-dJdRl[2])*J2m3 + (dJdRm[0]-dJdRl[2])*J1m3)
    dpotdR2 = (-dc2dR2-0.5*(2*dc2dR2*c2-dc1dR2*c3-c1*dc3dR2)/np.sqrt(c2*c2-c1*c3))/c1 - (-c2-np.sqrt(c2*c2-c1*c3))*dc1dR2/c1**2

    S1S2dS3 = S[0]*S[1]*dS[2]
    dc1dR3 = 2*onemS123*(-S1S2dS3) - 2*(-S22mS32 - S12mS32)*dS[2]*S[2]
    dc2dR3 = -(dQd[2]-epsilon*S1S2dS3)*onemS123 - QmJ123*(-S1S2dS3) + (-J2m3 - J1m3)*dS[2]*S[2] \
            + 0.5*((dJdRl[0]-dJdRm[1])*S12mS22 + (dJdRm[1]-dJdR[2])*S22mS32 + (dJdRl[0]-dJdR[2])*S12mS32)
    dc3dR3 = 2*QmJ123*(dQd[2]-epsilon*S1S2dS3) - ((dJdRl[0]-dJdRm[1])*J1m2 + (dJdRm[1]-dJdR[2])*J2m3 + (dJdRl[0]-dJdR[2])*J1m3)
    dpotdR3 = (-dc2dR3-0.5*(2*dc2dR3*c2-dc1dR3*c3-c1*dc3dR3)/np.sqrt(c2*c2-c1*c3))/c1 - (-c2-np.sqrt(c2*c2-c1*c3))*dc1dR3/c1**2

    dpotdr1 = dpotdR1 + dpotdR3*dR3dr1
    dpotdr2 = dpotdR2 + dpotdR3*dR3dr2
    dpotdtheta = dpotdR3*dR3dtheta

    return np.array([dpotdr1, dpotdr2, dpotdtheta])


def derivs(x,y):
    """
    Parameters
    ----------
    x : float,
        Time.

    y : ndarray, shape(7,),
        y=[r1,pr1,r2,pr2,theta, ptheta, time].

    Returns
    -------
    dydx : ndarray, shape(7,),
        Vector field for hydrogen exchange.
    """
    dydx=np.zeros(len(y))
    dV = dpotential(y[::2])
    dydx[0] = ( 2*y[1] - y[3]*np.cos(y[4]) + y[5]*np.sin(y[4])/y[0] )/mH
    dydx[1] = - dV[0] + (y[5]*np.sin(y[4])*y[1]/y[0]**2 \
                + y[5]*y[5]*(2/(y[0]*y[0]*y[0]) + np.cos(y[4])/(y[0]*y[0]*y[2]) ))/mH
    dydx[2] = ( 2*y[3] - y[1]*np.cos(y[4]) + y[5]*np.sin(y[4])/y[2] )/mH
    dydx[3] = - dV[1] + (y[5]*np.sin(y[4])*y[3]/y[2]**2 \
                + y[5]*y[5]*(2/(y[2]*y[2]*y[2]) + np.cos(y[4])/(y[0]*y[2]*y[2]) ))/mH
    dydx[4] = ( np.sin(y[4])*(y[1]/y[0]+y[3]/y[2]) + 2*y[5]*(1/(y[0]*y[0]) + 1/(y[2]*y[2]) +np.cos(y[4])/(y[0]*y[2])) )/mH
    dydx[5] = - dV[2] - (y[1]*y[3]*np.sin(y[4]) - y[5]*np.cos(y[4])*(y[1]/y[0]+y[3]/y[2]) \
                + y[5]*y[5]*(np.sin(y[4])/(y[0]*y[2]) ))/mH
    dydx[6] = 1
    return dydx

def lin_derivs(x,y):
    """
    Parameters
    ----------
    x : float,
        Time.

    y : ndarray, shape(7,),
        y=[r1,pr1,r2,pr2,theta, ptheta, time].

    Returns
    -------
    lin : ndarray, shape(7,),
        Numerical linearisation of the vector field for hydrogen exchange.
    """
    dydx=derivs(0,y)
    tiny = 1.0e-12
    shift = tiny*np.eye(7)
    y_shifted = y+shift
    f = lambda t :  derivs(0,t)
    dydx_shifted = np.array(list(map(f,y_shifted)))
    lin = (dydx_shifted - dydx)/tiny
    lin = lin[:6,:6]
    return lin

def contoursf(E=0.75):
    nstep=200
    r=np.linspace(1,4,nstep)
    pot_val=np.zeros((len(r),len(r)))
    for i in range(len(r)):
        for j in range(i+1):
            pot_val[j][i]=potential([r[i],r[j],0])
            pot_val[i][j]=pot_val[j][i]


    plt.figure(dpi=200)
    plt.contourf(r,r,pot_val, levels=np.linspace(0,E,11))
    plt.colorbar(format="%.3f")
    plt.xlim(np.min(r),np.max(r))
    plt.ylim(np.min(r),np.max(r))
    plt.xlabel(r'$r_1$')
    plt.ylabel(r'$r_2$')

if __name__ == '__main__':
    contoursf(E=0.75)