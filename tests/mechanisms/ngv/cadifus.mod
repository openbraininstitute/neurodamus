COMMENT
Calcium ion accumulation with radial and longitudinal diffusion, pump, 
and SERCA.

Diffusion geometry based on Ca accumulation models from chapter 9 
of The NEURON Book.

Mechanistic details of calcium pump and SERCA as described by Fink et al. 2000.

alpha = relative abundance of SERCA.

Current implementation assumes that ip3i is uniform across all compartments, 
i.e. that radial diffusion of IP3 is very fast compared to the diffusion of 
Ca and the mechanisms that drive Ca to change with time.
Indeed, simulations reveal this to be the case--IP3 concentration 
remains nearly identical across section diameters.
There are slight differences in the soma during the fast rising phase 
of the IP3 transient, but these resolve quickly.

Consequently, coupling between the shells of the ip3cum mechanism 
and the SERCA channels in the shells of this mechanism 
is a complexity that can be omitted--all shells of this mechanism
can use the concentration of IP3 in the outermost shell of the 
ip3cum mechanism, and discoverable by any mechanism that has a 
USEION ip3 READ ip3i VALENCE 1
statement in its NEURON block, and declares
ip3i (mM)
in its ASSIGNED block.

-------------
SERCA channel
-------------

jchnl = alpha * jmax * (1-(ca/caer)) * ( (ip3/(ip3+Kip3)) * (ca/(ca+Kact)) * h)^3
note:  jchnl is release from SER to cytoplasm
jmax = 3500 uM/s
caer = 400 uM

Kip3 = 0.8 uM
Kact = 0.3 uM

h' = kon * (Kinh - (ca + Kinh)*h)
kon = 2.7 /uM-s
Kinh = 0.2 uM

Recasting h in terms of kinetic scheme--
From RHS of ODE for h'
hinf = Kinh/(ca+Kinh) = alpha/(alpha+beta)
tauh = 1/(kon*(ca+Kinh)) = 1/(alpha+beta)
So alpha = kon*Kinh and beta = kon*ca

----------
SERCA pump
----------

jpump = alpha * vmax*ca^2 / (ca^2 + Kp^2)
note:  jpump is uptake from cytoplasm into SER

vmax = 3.75 uM/s/um2
Kp = 0.27 uM

----------
SERCA leak
----------

jleak = alpha * L*(1 - (Ca/caer))
note:  jleak is leak from SER to cytoplasm

L = 0.1 uM/s nominally,
but adjusted so that
jchnl + jpump + jleak = 0
when
ca = 0.05 uM and h = Kinh/(ca + Kinh)

ENDCOMMENT

NEURON {
    THREADSAFE
    SUFFIX cadifus
    POINTER glu2

:    USEION ca READ cao, cai, ica WRITE cai, ica    
:    USEION k READ ko, ki, ik WRITE ki, ik
:    USEION ip3 READ ip3i  WRITE ip3i VALENCE 1
    NONSPECIFIC_CURRENT i
    RANGE ica_pmp, cai0, cai00,fluo, fluoNew
    RANGE alpha : relative abundance of SERCA
  
RANGE ica_pmp, cai0,caer0, fluo,ikleak ,inka,nai,ko, incx,inaleak,fluoNew,reporter,reporter2 ,trigger ,j_ATP,perimFactor,eet,L,eATP,icaER,icaPLASMA,iNaCa,parea,pumpca00,pumpca,diam1,bound_pump0,bound_pump,ratioERsqrt,caerAll,ERarea_per_um,ip3i,glu3
  GLOBAL vrat, TBufs, TBufm, BufferAlpha,TSerca
  : vrat must be GLOBAL--see INITIAL block
  : however TBufs and TBufm may be RANGE


}


DEFINE Nannuli 4:4

UNITS {
    (mol)   = (1)
    (molar) = (1/liter)
    (uM)    = (micromolar)
    (mM)    = (millimolar)
    (um)    = (micron)
    (mA)    = (milliamp)
    FARADAY = (faraday)  (10000 coulomb)
    PI      = (pi)       (1)

	F = (faraday) (coulombs)
	R 	= (k-mole)	(joule/degC)

}

PARAMETER {

     
:	imax	=1e-2:1.6e3 (mA/cm2) :for inaca
     imax=0.01 :for 10 ?M ms?1 for inaca

	kna	=  87.5     (mM)
	kca	=  1.38     (mM)
	gamma1	= .35		: voltage dependence factor


:baseline conditions
  cai0 = 100e-6 (mM) 
  cai00 = 100e-6 (mM)
  caer0 = 0.4:400 uM
  :external potassium
  ko0=3 (mM)

  eATP0 	= 0.0000031 :Melani 2005
  :external sodium
  nao=145 (mM)
  nao0=145 (mM)

  :external calcium
  cao= 1.8 (mM)
  cao0= 1.8 (mM)

  :internal sodium
  nai0=10 (mM)

  ki0=135

  ip3i0 = 0.00001 (mM)



    
    perimFactor=1

    fluo = 0     (mM) 
    fluoNew = 0  
    DCa   = 0.22 (um2/ms) : Fink et al. 2000 0.22  
    DK   = 1.96 (um2/ms) : find a SOURCE
    :DK   = 1960 (um2/ms) : find a SOURCE
    BufferAlpha = 100


	modelStim = 0 (mM)
	kappa_delta = 1.5e-3 (mM) 
    K_3 = 1e-3 (mM)
    K_pi = 0.6e-3 (mM) 
    K_D = 0.7 (mM) 
   
    K_p = 10e-3 (mM)         
    K_PLCdelta = 0.1e-3 (mM)
    K_R = 1.3e-3       (mM)
    r_bar_5P = 0.04e-3 (/ms) 
  
  
    v_bar_3K = 2e-6 (mM/ms)
    v_bar_beta = 0.2e-6  (mM/ms)
    v_bar_delta = 0.02e-6  (mM/ms)

: Bufs--endogenous, stationary buffer
    TBufs =0.01:0.45:0.45:10:worked for realastrogeom
    :TBufs =0.45:(mM) : total Bufs
    : just make kfs fast, and calculate krs as kfs*KDs
    kfs = 10000 (/mM-ms) : try these for now
    :KDs = 10 (uM)
    KDs = 0.18 (uM)
    :KDs = 0.010 (mM)

    : Bufm--fura2, for bradykinin experiments
    TBufm =0:0.0075: (mM) : total Bufm
    : just make kfm fast, and calculate krm as kfm*KDm
    :kfm = 0.0000001 (/mM-ms) : try these for now
    kfm = 1000 (/mM-ms) : try these for now

    KDm = 0.24 (uM)
    :KDm = 0.00024 (mM)

    DBufm = 0.050 (um2/ms)

: Bufm--calcium green, for uncaging experiments
    :  TBufm = 0.075 (mM) : total Bufm
    : just make kfm fast, and calculate krm as kfm*KDm
    :  kfm = 1000 (/mM-ms) : try these for now
    :  KDm = 0.26 (/ms)
    :  DBufm = 0.0184 (um2/ms)

    : to eliminate ca pump, set gamma to 0 in hoc
    cath = 0.2e-3 (mM) : threshold for ca pump activity
    gamma = 1:8 (um/s) : ca pump flux density

    : SERCA params
    TSerca=1e-6
    alpha = 1:1 (1) : relative abundance of SERCA mechanism as per Fig. 3

: SERCA pump
    : jpump = alpha * vmax*ca^2 / (ca^2 + Kp^2)
    : jpump is uptake from cytoplasm into SER
    vmax = 0.001 (mm/ms/um2)
    : 3.75e-6 (mM/ms/um2)
    Kp = 0.27e-3 (mM)

: SERCA channel
    : jchnl is release from SER to cytoplasm
    jmax = 1.5e8(mM/ms)
    :  caer = 0.400 (mM)
:ip3
    Kip3 = 0.8e-3 (mM)
    Kact = 0.16e-3:0.3e-3 (mM) :Tu 2005 Biophyical Journal
    kon = 2.7 (/mM-ms)

    Kinh = 0.06e-3:0.2e-3 (mM):Tu 2005 Biophyical Journal

: SERCA leak -- no fixed parameter other than caer
: does have an adjustable parameter L

    trigger=0 : trigger for synaptic release


  decay = 80:80 (ms) : rate of removal of calcium, taken from CaDynamics_E24

}

ASSIGNED {
reporter
reporter2
     ina
  :  hc[Nannuli]
   : ho[Nannuli]
   i       (mA/cm2)

   : sercaB20
   sercaBound0[Nannuli]
   sercaUnBound0[Nannuli]
   k
   KB
   Kqa
   q10
   temp
   local_icaER
   allrate
   q101
   Kqa1
   KB1 
   k1
   Tpmca
   pmca_k1
   pmca_k2
   pmca_k3
   serca_k1
   serca_k2
   serca_kp2 

   KNCXmN1 
   KNCXmC1 

   rate0 
   ica1
   rate
   celsius	(degC)
   v	(mV)

   depth (micron)
   conversionFactorCa (1/cm/coulombs)

   diam      (um)
   diam1 (um) 
   L (um)

   ik (mA/cm2)
   ica       (mA/cm2)
   ica_pmp   (mA/cm2)
   ica_pmp_last   (mA/cm2)
   parea     (um)     : pumps per square micron

   :    cai       (mM)
   caerAll (mM)
   :cao       (mM)

   nai
   :ki
   ko

    

   vrat[Nannuli]  (1) : dimensionless

   : numeric value of vrat[i] equals the volume 
                        : of annulus i of a 1um diameter cylinder
                        : multiply by diam^2 to get volume per um length

    bufs_0 (mM)
    bufm_0 (mM)
    base1
val1
    :ip3i   (mM)

    LL[Nannuli] (mM/ms) : 0.1e-6 mM/ms nominally, but adjusted so that
    : jchnl + jpump + jleak = 0  when  ca = 0.05 uM and h = Kinh/(ca + Kinh)
    glu2 :variable for synaptic stimulation
glu3
    icaER
    icaPLASMA
    iNaCa
}

CONSTANT { volo = 1e10 (um2) }

STATE {
ikState
:ikState0
    tempState
    : ca[0] is equivalent to cai
    : ca[] are very small, so specify absolute tolerance
    : let it be ~1.5 - 2 orders of magnitude smaller than baseline level
    cai    
    ki
    ca[Nannuli]       (mM) <1e-7>
    caer[Nannuli]       (mM) <1e-7>
    bufs[Nannuli]    (mM) <1e-3>
    cabufs[Nannuli]  (mM) <1e-7>
    bufm[Nannuli]    (mM) <1e-4>
    cabufm[Nannuli]  (mM) <1e-8>
    hc[Nannuli]
    ho[Nannuli]
    hho[Nannuli]
    sercaU[Nannuli]
    sercaB1[Nannuli]
    sercaB2[Nannuli]
    gluExt
    eet (mM)
    catrack
    ip3i (mM) <1e-10>
    eATP
    bound_pump
    unbound_pump
    ratioERsqrt
    ERarea_per_um
}


BREAKPOINT {
    :    ica=0
    :ica=0
    ik=0
    
    :i=0
    caerAll=caer[0]

    if (nrn_pointing(glu2) && glu2>0){
:        ikState=1e-3
        ikState=1e1
:	ki=150
:        VERBATIM
:                printf("RECXXXXX\n");
:        ENDVERBATIM
        :ki=ki+ 1* 3000 /(6.02e23) *1000 / (L * diam^2 / (1e15)) :need to recalibrate this based on presynaptic k outflux
      
 :     reporter2=1								
      :  ki=ki+0.1 :this works but no voltage change
	glu2=0
:
    }
   : ik=-ikState
    reporter=ki
:    reporter2=ik
    i=(-ikState+ica+ina)
    reporter2=i
    if(0){
        :for frap
    :   FROM i=0 TO Nannuli-1 {
    :     cabufm[i] = 0
    :     bufm[i] = TBufm  *0.7
    :   }


        :glutamate transporter cost
        : 3000 mols glutamate
        : 3 molecules Na per glutamate
        : volume=diam^2 * L (in um2) * liter/1e15 um^3 

        nai=nai+ 3* 3000 /(6.02e23) *1000 / (L * diam^2 / (1e15))

        :metabotropic glutamate receptor
        gluExt=gluExt+1.1  :mM, Clements 1992
        :     glu2=0
        :    trigger=0      
        :trigger=1
    }

    k1 = R*(celsius + 273.14)/(F*1e-3)
    q101 = 3^((celsius - 37)/10 (degC))
    Kqa1 = exp(gamma1*v/k1)
    KNCXmN1 =87.5 
    KNCXmC1 =1.380 

    KB1 = exp( (gamma1 - 1)*v/k1)
   
    SOLVE state METHOD sparse
     


    icaER=33



    rate=-q10*imax*(KB*nao*nao*nao)/((kna*kna*kna + nao*nao*nao)*(kca + cao)*(1 + 0.1*KB))*ca[0]

    ica=- 2*rate / conversionFactorCa :+1.0e1*glu2 

    ica=ica+ 2*rate0 / conversionFactorCa 
    :ica=ica-2*tempState * (q10*imax*(KB*nao*nao*nao)/((kna*kna*kna + nao*nao*nao)*(kca + cao)*(1 + 0.1*KB)))/conversionFactorCa 

:NOT NEEDED ica=ica+tempState *cai0* (( ( 0* q10*imax*(Kqa*nai*nai*nai*cao-Kqa*nai0*nai0*nai0*cao0) + q10*imax*(KB*nao*nao*nao)/((kna*kna*kna + nao*nao*nao)*(kca + cao)*(1 + 0.1*KB))))) /conversionFactorCa 

ica=ica-pmca_k3*bound_pump /conversionFactorCa +pmca_k3*bound_pump0 /conversionFactorCa 
ica=-ica
:glu3 is the number of potassium ions


cai = ca[0]
:if (ik>reporter2){
:reporter2=ik
:}
}
LOCAL volshell,volin,conversionFactor 
LOCAL factors_done, jx , k1

INITIAL {

    ki=ki0
    volin = PI*diam*diam/4 : Surface area (volume per unit length)
    conversionFactor = (1e4)*4/diam/FARADAY
    k = R*(celsius + 273.14)/(F*1e-3)
    KB = exp( (gamma1 - 1)*v/k)
    q10 = 3^((celsius - 37)/10 (degC))
    Kqa = exp(gamma1*v/k)

    tempState=1
    catrack=0
    rate0 = pumprate(v,nai,nao,cai00,cao)
    ica1=0




    if (factors_done == 0) {  : flag becomes 1 in the first segment
        factors_done = 1       :   all subsequent segments will have
        factors()              :   vrat = 0 unless vrat is GLOBAL
    }

    ip3i = ip3i0 
    nai=nai0
    cai = cai0
    caerAll=caer[0]

    bufs_0 = 0.001*KDs*TBufs/(0.001*KDs + (1)*cai0)

    bufm_0 = 0.001*KDm*TBufm/(0.001*KDm + (1)*cai0)

    diam1=diam
    : parea = 2000 :number pmca per square micron
    :    Tpmca=0.05e-3 :Handy 2016
    :Tpmca=0.05e-5
    Tpmca=0.01e-3/0.200 :mM
 
    depth = diam/4/(Nannuli-1)

    ratioERsqrt=0.2
    FROM i=0 TO Nannuli-1 {
        ca[i] = cai0
        caer[i] = caer0 
        :        caer[i] = caer0*ratioERsqrt/(1-ratioERsqrt)
        bufs[i] = bufs_0
        cabufs[i] = TBufs - bufs_0
        bufm[i] = bufm_0
        cabufm[i] = TBufm - bufm_0
    }
 
    volshell = PI*((diam*diam/4) - (diam/2-depth)*(diam/2-depth)) : 
    :volshell = PI*((diam*diam) - (diam-depth)*(diam-depth))/4 : Surface area ca-shell

    serca_kp2 =0.05*0.001 :mM
    serca_k1=500000
    serca_k2=serca_kp2 *serca_k1

    FROM i=0 TO Nannuli-1 {
      sercaB2[i]=TSerca/(1+serca_kp2/cai0+(serca_kp2/cai0)*(serca_kp2/cai0))
      
      sercaU[i]=TSerca/(1+cai0/serca_kp2+(cai0/serca_kp2)*(cai0/serca_kp2))
      sercaB1[i]=TSerca/(1+serca_kp2/cai0+cai0/serca_kp2)
      
      sercaBound0[i]=sercaB2[i]
      sercaUnBound0[i]=sercaU[i]
    }
    
    pmca_k1=280
    pmca_k2=0.5 :Kd, ?0.2 ?M
    pmca_k3=5:RESTORE 0.200 :num/ms
 bound_pump=Tpmca*(pmca_k1*cai0)/(pmca_k1*cai0+pmca_k2+pmca_k3) :effective concentration of pumps in mM Handy 2016
   
: bound_pump=parea *(pmca_k1*cai0)/(pmca_k1*cai0+pmca_k2+pmca_k3)*perimFactor*(PI*diam1)/(volshell) * (1e15)/(6.02e23)*1000:effective concentration of pumps in mM

unbound_pump=Tpmca-bound_pump

  :  unbound_pump=parea *perimFactor*(PI*diam1)/volshell*(1e15)/(6.02e23)*1000-bound_pump

    bound_pump0=bound_pump

    : reconsider and revise initialization comments
    ica=0
    ica_pmp = 0
    ica_pmp_last = 0
    : If there is a voltage-gated calcium current, 
    : this is almost certainly the wrong initialization. 
    : In such a case, first do an initialization run, then use SaveState
    : On subsequent runs, restore the initial condition from the saved states.

    FROM i=0 TO Nannuli-1 {
:        ho[i] = Kinh/((ca[i])+Kinh)
:        hho[i] =ho[i] 

        ho[i]=Kinh /(Kinh+ca[i])  
        hho[i] =ho[i] 

        hc[i] = 1-ho[i]

        : jx = jp + jc
        : choose L so that jl = -jx
        : jl = L*(1 - (ca[i]/caer))
        : jp = (-vmax*ca[i]^2 / (ca[i]^2 + Kp^2))
        : jc = jmax*(1-(ca[i]/caer)) * ( (ip3i/(ip3i+Kip3)) * (ca[i]/(ca[i]+Kact)) * ho[i] )^3
        jx = (-vmax*ca[i]^2 / (ca[i]^2 + Kp^2))
        :jx = jx + jmax*(1-(ca[i]/caer[i])) * ( (ip3i/(ip3i+Kip3)) * (ca[i]/(ca[i]+Kact)) * ho[i] )^3

jx = jx + jmax*(1-(ca[i]/caer0)) * ( (ip3i/(ip3i+Kip3)) * (ca[i]/(ca[i]+Kact)) * ho[i] )^3

:        LL[i] = -jx/(1 - (ca[i]/caer[i]))
        LL[i] = -jx/(1 - (ca[i]/caer0))
    }
:ca[0]=0.001


  
  conversionFactorCa = gamma*PI*diam*(1e8)/(2*FARADAY*volshell) :get in mA/cm2, want out mM/liter, there are 1000 cm3 per liter, 1e4 um per cm, depth in um

 : conversionFactorCa = gamma*(1e4)/(2*FARADAY*depth)*1000 :get in mA/cm2, want out mM/liter, there are 1000 cm3 per liter, 1e4 um per cm, depth in um

reporter2=0

}

LOCAL frat[Nannuli]  : scales the rate constants for model geometry

PROCEDURE factors() {
    LOCAL r, dr2
    r = 1/2                 : starts at edge (half diam)
    dr2 = r/(Nannuli-1)/2   : full thickness of outermost annulus,
                            : half thickness of all other annuli

    vrat[0] = 0
    frat[0] = 2*r

    FROM i=0 TO Nannuli-2 {
        vrat[i] = vrat[i] + PI*(r-dr2/2)*2*dr2  : interior half
        r = r - dr2
        frat[i+1] = 2*PI*r/(2*dr2)  : outer radius of annulus
                                    : div by distance between centers
        r = r - dr2
        vrat[i+1] = PI*(r+dr2/2)*2*dr2  : outer half of annulus
    }
}

LOCAL dsq, dsqvol, K_gamma, v_3K, v_delta, v_glu   : can't define local variable in KINETIC block
                    :   or use in COMPARTMENT statement

KINETIC state {
    COMPARTMENT i, (1-ratioERsqrt)*diam*diam*vrat[i] {ca bufs cabufs bufm cabufm sercaB1 sercaB2 sercaU}
COMPARTMENT  (1-ratioERsqrt)*diam*diam/4*PI {ki}

    :note that the volume for caer should be ratioERsqrt but it crashes when this is done so (1-ratioERsqrt) is used instead and caer[i] is scaled proportionaly when used
    COMPARTMENT i, ratioERsqrt*diam*diam*vrat[i]  {caer}
   
    COMPARTMENT (1-ratioERsqrt)*diam*diam*vrat[0]  {unbound_pump bound_pump tempState }
    :COMPARTMENT volo {cao}
    LONGITUDINAL_DIFFUSION i, DCa*(1-ratioERsqrt)*diam*diam*vrat[i] {ca bufm cabufm}    
    LONGITUDINAL_DIFFUSION DK*(1-ratioERsqrt)*diam*diam/4*PI {ki}  
:  LONGITUDINAL_DIFFUSION 0.01*DK*(1-ratioERsqrt)*diam*diam/4*PI {ki}  
    dsq = diam*diam

    dsqvol = dsq*vrat[0]

    : dynamics of IP3 ions  
    
    K_gamma = K_R * (1 + (K_p / K_R) * ca[0] / (ca[0] + K_pi))
    v_3K = v_bar_3K * ca[0]^4/(ca[0]^4 + K_D^4) * ip3i / (ip3i + K_3) - v_bar_3K * cai0^4/(cai0^4 + K_D^4) * ip3i0 / (ip3i0 + K_3)
    v_delta = v_bar_delta/(1 + ip3i/kappa_delta)* ca[0]^2 / (ca[0]^2 + K_PLCdelta^2) -v_bar_delta/(1 + ip3i0/kappa_delta)* cai0^2 / (cai0^2 + K_PLCdelta^2)           
    : v_glu = v_bar_beta * modelStim^0.7 / (modelStim^0.7 + K_gamma^0.7)
    v_glu = v_bar_beta * modelStim / (modelStim + K_gamma)
    : ~ ip3i << ((PI*diam*diam*(v_glu + v_delta - v_3K - r_bar_5P * ip3i+r_bar_5P * ip3i0))/2)
    ~ ip3i << (0*v_glu)


:LOCAL rate
    :rate = (pumprate(v,nai,nao,cai,cao)-rate0)
    rate=-q10*imax*(KB*nao*nao*nao)/((kna*kna*kna + nao*nao*nao)*(kca + cao)*(1 + 0.1*KB))*ca[0]

    ica1 =- 2*rate :note sign
    iNaCa=-ica1 
    ina=3 * (rate-rate0)
:ica=-ica1 / conversionFactorCa 

:    ~ ca[0] << (1.0e0*glu2 * conversionFactorCa )
:ica=ica+1.0e0*glu2 

:RESTORE
    ~ca[0]+tempState <-> tempState ( (q10*imax*(KB*nao*nao*nao)/((kna*kna*kna + nao*nao*nao)*(kca + cao)*(1 + 0.1*KB))),cai0* ( ( 0* q10*imax*(Kqa*nai*nai*nai*cao-Kqa*nai0*nai0*nai0*cao0) + q10*imax*(KB*nao*nao*nao)/((kna*kna*kna + nao*nao*nao)*(kca + cao)*(1 + 0.1*KB)))) )

:RESTORE ica=ica-2*tempState * (q10*imax*(KB*nao*nao*nao)/((kna*kna*kna + nao*nao*nao)*(kca + cao)*(1 + 0.1*KB)))/conversionFactorCa 

:RESTORE ica=ica+tempState *cai0* (( ( 0* q10*imax*(Kqa*nai*nai*nai*cao-Kqa*nai0*nai0*nai0*cao0) + q10*imax*(KB*nao*nao*nao)/((kna*kna*kna + nao*nao*nao)*(kca + cao)*(1 + 0.1*KB))))) /conversionFactorCa 

    icaPLASMA= -pmca_k3*bound_pump 

:RESTORE ica=ica-pmca_k3*bound_pump /conversionFactorCa +pmca_k3*bound_pump0 /conversionFactorCa 

    : cell membrane ca pump
    ~ ca[0] + unbound_pump <-> bound_pump(pmca_k1, pmca_k2) :0.001 um/mm
    ~ bound_pump <-> unbound_pump( pmca_k3,0)
    ~ ca[0] << (bound_pump0*pmca_k3)
    : all currents except cell membrane ca pump   

    : radial diffusion
    FROM i=0 TO Nannuli-2 {
        ~ ca[i] <-> ca[i+1]  (DCa*frat[i+1], DCa*frat[i+1])     
        ~ caer[i] <-> caer[i+1]  (DCa*frat[i+1], DCa*frat[i+1])
        ~ bufm[i] <-> bufm[i+1]  (DBufm*frat[i+1], DBufm*frat[i+1])


    }
    : buffering
    dsq = diam*diam
    FROM i=0 TO Nannuli-1 {
        dsqvol = dsq*vrat[i]
        
        ~ ca[i] + bufs[i] <-> cabufs[i]  ((1-ratioERsqrt)* kfs*dsqvol, (1-ratioERsqrt)* (0.001*KDs)*kfs*dsqvol)
bufm[i] =0
cabufm[i]  =0
        :~ ca[i] + bufm[i] <-> cabufm[i]  ( (1-ratioERsqrt)* kfm*dsqvol, (1-ratioERsqrt)* (0.001*KDm)*kfm*dsqvol)

    }
    
    : SERCA pump, channel, and leak

    icaER=0
    FROM i=0 TO Nannuli-1 {
        dsqvol = dsq*vrat[i]
        : SERCA pump

        :use 1 um2 per um3 as the Sa to volume ratio of the ER
        ERarea_per_um=(1)*ratioERsqrt*vrat[i]*dsqvol


        :vmax = 1e-14:1e-12:0:0.9 *0.001/1000 / (1e-6):0:5e-18:5e-15 : from Handy 2016
        :vmax=velocity_max_in_mM_per_msec/sercaT_in_mM
        vmax=(0.0009/1000)/(1e-6)

        local_icaER=alpha*vmax *(1e15)*ERarea_per_um*sercaB2[i] 
        icaER= icaER +(- local_icaER)

:RESTORE
        ~ ca[i] + sercaU[i]<-> sercaB1[i]  (serca_k1,serca_k2)
        ~ ca[i] + sercaB1[i]<-> sercaB2[i]  (serca_k1,serca_k2)
        ~ sercaB2[i] <-> sercaU[i] (vmax ,sercaBound0[i] / sercaUnBound0[i] *vmax )

        :~ caer[i] << (max0((alpha*vmax *(1e15)*ERarea_per_um ) * ( ca[i]^2 / (ca[i]^2 + (0.001*Kp)^2) - cai0^2 / (cai0^2 + (0.001*Kp)^2) ) )):the 1e15 is needed because the volume of the compartment is in um3


        : ip3
        ~ hc[i] <-> ho[i]  ( 1000*kon*Kinh,1000* kon*ca[i])        
:ca[i]=1e-4
       : ho[i]=Kinh /(Kinh+ca[i])  
       : hc[i]=1-ho[i]    

:~ hc[i] <<(0)
:~ ho[i] <<(0)


   :     ~ hc[i] <-> ho[i]  (kon*Kinh, kon*ca[i])

        : ~ ca[i] << ( dsqvol*alpha*jmax*(1-(ca[i]/caer)) * ( (ip3i/(ip3i+Kip3)) * (ca[i]/(ca[i]+Kact)) * ho[i] )^3 )

        base1=((1-cai00/ ( 1 * caer0) ) *( ip3i0/(ip3i0+0.001*Kip3) * cai00/(cai00+Kact) * hho[i]  )^3 )     

        :val1= ((1-(ca[i])/ ( 1 * caer[i]) ) *( ip3i/(ip3i+0.001*Kip3) * (ca[i])/((ca[i])+Kact) * ho[i]  )^3 ) 

        val1= ((1- ca[i]/ caer0 ) *( ip3i0/(ip3i0+0.001*Kip3) * (ca[i])/((ca[i])+Kact) * ho[i]  )^3 ) 


        :jmax = 1.5e8
        :jmax = 1.5e-15
        :~ ca[i] << ( alpha*jmax* 1e15 *ERarea_per_um*  ( val1-base1)  ) 
        :~ ca[i] << ( 0.001* 0.000222 * caer0 *( val1-base1)  ) 

        :RESTORE
        ~ ca[i] << ( (1-ratioERsqrt)*dsqvol* 0.000222 * caer0 *max0( val1-base1)  ) 

:VERBATIM
:        printf("%f\n",ko);
:ENDVERBATIM

:reporter=( (1-ratioERsqrt)*dsqvol* 0.000222 * caer0 *max0( val1-base1)*1e9  ) 

:~ caer[i] << ( -(ratioERsqrt)*dsqvol* 0.000222 * caer[i] *max0( val1-base1)  ) 

        :~ caer[i] << (-( max0(alpha*jmax* (1e15) *ERarea_per_um*  ( (1-ca[i]/ ( 1 * caer[i]) ) *( ip3i/(ip3i+0.001*Kip3) * ca[i]/(ca[i]+0.001*Kact) * ho[i]  )^3  -  (1-cai0/caer0) *( ip3i0/(ip3i0+0.001*Kip3) * cai0/(cai0+0.001*Kact) *  hho[i]    )^3 ) )  ))      

    }

:    ~ ikState << (-0.01)
    ~ ikState << (-1)
	
    : fluo = cabufm[0]
    : fluoNew = (BufferAlpha * cabufm[0] + ca[0] - BufferAlpha*(TBufm - bufm_0) - cai0)/(BufferAlpha*(TBufm - bufm_0) + cai0)


:    if(icaER* 0.001/(ca[0])>1){
:    VERBATIM
:        printf("%f\n",icaER* 0.001/(ca[0]));
:    ENDVERBATIM
:    }

    ~ cai<<(0)   
   ~ ki << (-ik  * conversionFactor*volin  )
:~ ki << (0)


}

FUNCTION u(x, th) {
    if (x>th) {
        u = 1
    } else {
        u = 0
    }
}

FUNCTION max0(x) {
    if (x>0) {
        max0 = x
    } else {
        max0 = 0
    }
}

FUNCTION round1(x) {
    if (((x-cai00)*(x-cai00))< 1e-30){
        round1 = cai00
    } else {
        round1=x
    }
}

FUNCTION pumprate(v,nai,nao,cai,cao) {
	LOCAL q10, Kqa, KB, k,KNCXmN ,KNCXmC 
	k = R*(celsius + 273.14)/(F*1e-3)
	q10 = 3^((celsius - 37)/10 (degC))
	Kqa = exp(gamma1*v/k)
     KNCXmN =87.5 :mM
     KNCXmC =1.380 :mM

	KB = exp( (gamma1 - 1)*v/k)

	pumprate = q10*imax* nao0*nao0*nao0 * cao0* ( nai*nai*nai/(nao0*nao0*nao0) * Kqa- cai/cao0 * KB)/(KNCXmN *KNCXmN *KNCXmN +nao0*nao0*nao0) /(KNCXmC+cao0) /(1 + 0.1*KB)

:	pumprate = q10*imax* nao*nao*nao * cao *( nai*nai*nai/(nao*nao*nao) * Kqa- cai/cao * KB)/(KNCXmN *KNCXmN *KNCXmN +nao*nao*nao) /(KNCXmC+cao) /(1 + 0.1*KB)


:	pumprate = q10*imax*(Kqa*nai*nai*nai*cao-KB*nao*nao*nao*cai)/((kna*kna*kna + nao*nao*nao)*(kca + cao)*(1 + 0.1*KB))



}

VERBATIM
/** not executed in coreneuron and hence need empty stubs only */
static void bbcore_write(double* x, int* d, int* xx, int* offset, _threadargspr\
oto_) {
}
static void bbcore_read(double* x, int* d, int* xx, int* offset, _threadargspro\
to_) {
}
ENDVERBATIM


