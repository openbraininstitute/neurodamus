TITLE Adding vascular coupling to endfoot of astrocyte + SMC model, enhanced model

: PARAMETERS NEEDED -> j_BK, eet  -> values given in hoc file

NEURON {

    SUFFIX vascouplingB
    RANGE  eete, j_KIR, kp, vKIR, gKIR, Vi, Cam, Frm, M, ICam, jCam, pCam, iCam, th, Rad, R0pas, I_BK ,phi_n,n,neq, kp,v3,j_BK ,gKIR ,Vi,I_KIR,vKIR
    USEION ca READ  cao,cai
}

UNITS {

    (mV) = (millivolt)
    (mA) = (milliamp)
    (mM) = (mole/m3)
    (um) = (micron)
    FARADAY = (faraday) (coulomb)
    (uF)    =  (microfarad)
    PI = (pi) (1)
    (uS) = (microsiemens)

}

PARAMETER {

    f   = 96485.33289 (C/mol) : Faraday constant
    Cm = 0.000015 (uF) :using 15pF approximate value inspired by Kreft et al. 2004
    e   = 2.7182818284590452353602874713527 : euler constant used for exponential functions

    : Parameters to initialize in hoc file taken from other mechanism (pointer-like)

    eete = 0.0 : eet concentration

    :: SMOOTH MUSCLE AND ENDOTHELIAL CELL ::
    : PERIVASCULAR SPACE (David 2011)
    VRpa = 0.0005 :VR perivascular/astrocyte
    VRps = 0.1 :VR perivascular/SMC
    Rdecay = 0.00005 (1/ms) :K+ clearance rate
    kpmin = 3 (mM) :K+ equilibrium
    gammaeet = 4 :in (mV/(mM*ms)) was 0.004 V/(uM*s) :Bennet 2008
    keet = 0.0072 (1/ms)

    : CONTRACTION MODEL
    : L-type Ca++ channel of SMC (Bennet 2008)
    Vh = 6.2 (mV)
    kCam = 9.5 (mV)
    Vbar = 1.44 (mV)
    kappa = 0.0000000000041 (mA/mV) :4.1 (pA/V) -> pow(10,-12)
    Ntot = 10000 : estimation
    Cam0 = 0.00005 (mM)
    Caminit = 0.0003 (mM)
    Cmu = 0.000084 (uF)

    : Rate constant for actin myosin cross bridges (Dormanns 2015)
    K2 = 0.0005 (1/ms)
    K3 = 0.0004 (1/ms)
    K4 = 0.0001 (1/ms)
    K5 = 0.0005 (1/ms)
    K7 = 0.0001 (1/ms)
    cross = 17 : Sensitivity to Ca++ (units uM-3 s-1)

    : MECHANICAL MODEL (Dormanns 2015)
    R0pas = 20 (um) :Passive radius
    th0pas = 3 (um) :Passive thickness
    Pt = 4000 (Pa) :Transmural pressure
    Epas = 66000 (Pa) :Youngs moduli passive
    Eact = 233000 (Pa) :Youngs moduli active
    alphar = 0.6
    visc = 10000000 (Pa*ms) :viscosity

    ::: EET and BK subsystem :::
    : BK channel
    gBK = 0.013854 (uS) :Conductance
    Ebk = -95.0 (mV) :Reversal potential
    eet_shift = 2000 (mV/mM) :Voltage shift
    gamma_n = 0.002664 (1/ms) :Characteristic time
    : BK related constants
    : Kenny 2018
:   v4 = 8 (mV)
:   v5 = 15 (mV)
:   v6 = -55 (mV)
:   Ca3 = 0.0004 (mM)
:   Ca4 = 0.00035 (mM)

:from gonzalez-fernandez and ermentrout 1994
    v4 = 14.5 (mV)
    v5 = 8 (mV)
    v6 = -15 (mV)
    Ca3 = 0.0004 (mM)
    Ca4 = 0.00015 (mM)

    dt=0.025 (ms)
}

ASSIGNED {
    v (mV)
    : Geometry
    Vcyt

    :: SMOOTH MUSCLE AND ENDOTHELIAL CELL ::
    : PERIVASCULAR SPACE
    : KIR channel
    j_KIR
    I_KIR
    vKIR
    gKIR (uS)

    : CONTRACTION MODEL
    : L-type Ca++ channel of SMC
    pCam
    pCam0
    iCam
    iCam0
    ICam
    ICam0
    jCam
    : Other model
    Cam
    : Ca++ dependent actin myosin rate constants
    K1
    K6
    Frm

    : MECHANICAL MODEL
    th (um) :wall thickness of the vessel
    EFrm
    R0Frm

    : K+ intracellular concentration
    ike     (mM)

    : BK channel
    j_BK
    I_BK
    neq
    v3
    phi_n
     eet

    cai       :(mM)
    cao       :(mM)
init0
}

STATE {

    :: SMOOTH MUSCLE AND ENDOTHELIAL CELL ::

    : PERIVASCULAR SPACE
    : Potassium in perivascular space
    kp

    : Membrane potential of SMC
    Vi

    : CONTRACTION MODEL
    : free nonphosphorylated cross bridges
    M

    : free phosphorylated cross bridges
    Mp

    : attached phosphorylated cross bridges
    AMp

    : attached dephosphorylated latch bridges
    AM

    : MECHANICAL MODEL
    : Vessel radius in um
    Rad
    n
}

BREAKPOINT{

    rates(kp,Vi,AM,Mp,AMp,Rad)

    :: SMOOTH MUSCLE AND ENDOTHELIAL CELL ::
    : PERIVASCULAR SPACE
    :kp = kp + dt*((1/VRpa)*j_BK+(1/VRps)*(j_KIR)-Rdecay*(kp-kpmin)) :Perivascular K+
    :kp = kp + dt*((1/VRpa)*j_BK+(1/VRps)*(j_KIR)-Rdecay*1000*(kp-kpmin))

:RESTORE NEXT
    kp = kp + dt*((1/VRpa)*j_BK*0.0001+(1/VRps)*(j_KIR)-Rdecay*1000*(kp-kpmin))
    :kp = kp + dt*((1/VRpa)*j_BK*0.0001-Rdecay*1000*(kp-kpmin))

    : Membrane potential of SMC cell. Added mechanism to come back to resting potential, otherwise stays hyperpolarized
    : Via activation of KIR channel
    :Vi = Vi + dt*(1/Cmu)*(I_KIR)*pow(10,7) - dt*keet*5*pow(10,-2)*(Vi+30): with inactivation
    :Vi = Vi + dt*(1/Cmu)*(I_KIR)*pow(10,7)  - dt*keet*5*pow(10,-2)*(Vi+30): with inactivation
    Vi = Vi + dt*(1/Cmu)*(I_KIR)*pow(10,7)  - 0*dt*eete*1.2-0.001*(Vi+30)

    :no eet depedendence
    :Vi = Vi + dt*(1/Cmu)*(I_KIR)*pow(10,7)-0.0001*(Vi+40)

    : CONTRACTION MODEL
    M = M + dt*(-K1*M + K2*Mp +K7*AM)
    Mp = Mp + dt*(K4*AMp + K1*M -(K2+K3)*Mp)
    AMp = AMp + dt*(K3*Mp + K6*AM -(K4+K5)*AMp)
    AM = AM + dt*(K5*AMp -(K7+K6)*AM)

    : MECHANICAL MODEL
    Rad = Rad + dt* (R0pas/visc)* ( (Rad*Pt)/th - EFrm*(Rad-R0Frm)/(R0Frm) -init0)

    n=n+dt*(20000*phi_n*(neq-n))

:VERBATIM
:printf("WWW v: %g kp: %g Vi: %g ICam: %g Cam: %g Frm: %g R0Frm: %g K1:%g M: %g Mp: %g AMp: %g AM: %g %g %g %g %g %g Rad: %g\n",v,kp,Vi,ICam,Cam,Frm,R0Frm,K1,M,Mp,AMp,AM,EFrm,Pt,R0pas,th,R0Frm,Rad);
:ENDVERBATIM


}


INITIAL {
    :: SMOOTH MUSCLE AND ENDOTHELIAL CELL ::

    : PERIVASCULAR SPACE
    kp = kpmin:7
    Vi = -30.0

    : CONTRACTION MODEL

M= 0.434884 
Mp= 0.0679663 
AMp= 0.138629 
AM= 0.358521

  :  M = 0.25
  :  Mp = 0.25
  :  AMp = 0.25
  :  AM = 0.25

    : MECHANICAL MODEL
    :Rad = 20
      Rad = 14.7

    : BK subsystem :
    n = 0.0

    rates(kp,Vi,AM,Mp,AMp,Rad)
init0=(Rad*Pt)/th - EFrm*(Rad-R0Frm)/(R0Frm)
}

UNITSOFF

PROCEDURE rates(kp,Vi,AM,Mp,AMp,Rad) {

    : Assuming Vcyt = 11 um3, read in old version
    Vcyt = 11

    :: SMOOTH MUSCLE AND ENDOTHELIAL CELL ::

    : KIR channel, Dormanns 2015
    :vKIR = 4.5*pow(10,3)*kp*pow(10,3) - 112  : result in mV
    vKIR = 4.5*pow(10,3)*kp*pow(10,-3) - 112  : result in mV, used
    :gKIR = pow(e,(((-7.4*pow(10,-2))*Vi)+(4.2*pow(10,2)*kp*pow(10,3))-12.6)) :most logic, result in uM/s.mV
    gKIR = pow(e,(((-7.4*pow(10,-2))*Vi)+(4.2*pow(10,-1)*kp)-12.6)) :mostly used
    j_KIR = -666*((gKIR*750)/1970)*(Vi - vKIR)*pow(10,-6) :initially given in uM/s, we convert to mM/ms
    I_KIR = j_KIR*pow(10,3)*f*Vcyt*pow(10,-18)*pow(10,3) :in mA, careful using Vcyt is not necessarily accurate

    : CONTRACTION MODEL
    : L-type Ca++ channel of SMC
    pCam = 1/(1+pow(e,(Vh-Vi)/kCam)) : open probability of single Ca channel in SMC
    iCam = kappa*((Vi - Vbar)/(1-pow(e,0.075*(Vi-Vbar)))) : single channel current of Ca into SMC, mA
    ICam = Ntot*pCam*iCam : total current of Ca into SMC, result in mA
    jCam = ICam*pow(10,-3)*(1/f)*(1/(11*pow(10,-18)))*pow(10,-3) : TEST, would be the flux of ion through channel (result in mM/ms)
    pCam0 = 1/(1+pow(e,(Vh-(-30))/kCam))
    iCam0 = kappa*((-30 - Vbar)/(1-pow(e,0.075*(-30-Vbar)))) : initial current for initial Vi
    ICam0 = Ntot*pCam0*iCam0
    Cam = Cam0 + ((Caminit-Cam0)/ICam0)*ICam

    : Ca++ dependent actin myosin rate constants
    K1 = cross*pow(Cam*pow(10,3),3)*pow(10,-3) : result in 1/ms, Cam in mM converted to uM
    K6 = cross*pow(Cam*pow(10,3),3)*pow(10,-3) : result in 1/ms, Cam in mM converted to uM
    M = 1 - AM - AMp - Mp
    Frm = AMp + AM :Fraction of attached cross bridges

    : MECHANICAL MODEL
    EFrm = Epas + Frm*(Eact - Epas) :Frm dependent young modulus
    R0Frm = R0pas + Frm*(alphar - 1)*R0pas :Frm dependent initial radius
    th = -Rad + sqrt(pow(Rad,2)+2*R0pas*th0pas+pow(th0pas,2)) :thickness


    : BK channel, David 2011
    v3 = (-(v5/2)*tanh((cai-Ca3)/Ca4))+v6
    :v3 = (-(v5/2)*tanh((cai/10-Ca3)/Ca4))+v6

    neq = 0.5*(1+tanh((e+(eet_shift*eet)-v3)/v4))
    :neq=   ((e+(eet_shift*eet)-v3)/v4)
    phi_n = gamma_n*cosh((e-v3)/(2*v4))
:    I_BK    = gBK*n*(e-Ebk)*pow(10,-6) : result in mA (ionic current)
    I_BK    = gBK*n*(v-Ebk)*pow(10,-6) : result in mA (ionic current)

    j_BK = I_BK*pow(10,-3)*(1/f)*(1/(11*pow(10,-18)))*pow(10,-3) : result in mM/ms (flux of ion through channel)



}

UNITSON
