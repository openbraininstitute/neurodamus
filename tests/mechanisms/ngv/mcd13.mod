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

vmax = 3.75 uM/s
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
        SUFFIX mcd
        BBCOREPOINTER glu2
        USEION ca READ ica WRITE cai VALENCE 2
        : USEION IP3 READ iIP3 WRITE IP3i VALENCE 2

        RANGE ica_pmp, cai0, fluo, fluoNew
        RANGE alpha : relative abundance of SERCA
        RANGE perimeter, cross_sectional_area,er_area,er_volume

        RANGE reportcai,ica_pmp, cai0, fluo,ikleak ,inka,nai,ko, incx,inaleak,fluoNew,reporter,reporter2,reporter3,reporter4,reporter5,reporter6,reporter7,reporter8,trigger ,j_ATP,perimFactor,eet,L,eATP,icaER,icaPLASMA,parea,pumpca00,IP3i ,pumpca,diam1
        GLOBAL vrat, TBufs, TBufm, BufferAlpha
        : vrat must be GLOBAL--see INITIAL block
        : however TBufs and TBufm may be RANGE

}


DEFINE Nannuli 4

UNITS {
    (mol)   = (1)
    (molar) = (1/liter)
    (uM)    = (micromolar)
    (mM)    = (millimolar)
    (um)    = (micron)
    (mA)    = (milliamp)
    FARADAY = (faraday)  (10000 coulomb)
    PI      = (pi)       (1)
}

PARAMETER {
    :baseline conditions
    cai0 = 50e-6 (mM)

    :external potassium
    ko0=3 (mM)

    eATP0   = 0.0000031 :Melani 2005
    :external sodium
    nao=145 (mM)
    nao0=145 (mM)

    :external calcium
    cao= 1.8 (mM)
    cao0= 1.8 (mM)

    :internal sodium
    nai0=10 (mM)

    ki0=135

    ip3i0 = 0.000001 (mM)

    reporter=0
    reporter2=0
    reporter3=0
    reporter4=0
    reporter5=0
    reporter6=0
    reporter7=0
    reporter8=0

    perimFactor=1

    fluo = 0     (mM)
    fluoNew = 0
    DCa   = 0.22 (um2/ms) : Fink et al. 2000 0.22
    BufferAlpha = 100

    : Bufs--endogenous, stationary buffer
    TBufs = 0.450 (mM) : total Bufs
    : just make kfs fast, and calculate krs as kfs*KDs
    kfs = 1000 (/mM-ms) : try these for now
    KDs = 10 (uM)

    : Bufm--fura2, for bradykinin experiments
    TBufm = 0.075 (mM) : total Bufm
    : just make kfm fast, and calculate krm as kfm*KDm
    kfm = 1000 (/mM-ms) : try these for now
    KDm = 0.24 (uM)
    DBufm = 0.050 (um2/ms)

    : Bufm--calcium green, for uncaging experiments
    :  TBufm = 0.075 (mM) : total Bufm
    : just make kfm fast, and calculate krm as kfm*KDm
    :  kfm = 1000 (/mM-ms) : try these for now
    :  KDm = 0.26 (/ms)
    :  DBufm = 0.0184 (um2/ms)

    : to eliminate ca pump, set gamma to 0 in hoc
    cath = 0.2e-3 (mM) : threshold for ca pump activity
    gamma = 8 (um/s) : ca pump flux density

    : SERCA params
    alpha = 1 (1) : relative abundance of SERCA mechanism as per Fig. 3

    : SERCA pump
    : jpump = alpha * vmax*ca^2 / (ca^2 + Kp^2)
    : jpump is uptake from cytoplasm into SER
    vmax = 3.75e-6 (mM/ms)
    Kp = 0.27e-3 (mM)

    : SERCA channel
    : jchnl is release from SER to cytoplasm
    jmax = 3.5e-3 (mM/ms)
    caer = 0.400 (mM)
    Kip3 = 0.8e-3 (mM)
    Kact = 0.3e-3 (mM)
    kon = 2.7 (/mM-ms)
    Kinh = 0.2e-3 (mM)

   : SERCA leak -- no fixed parameter other than caer
   : does have an adjustable parameter L

   trigger=0 : trigger for synaptic release

}

ASSIGNED {
    reportcai
    diam      (um)
    diam1 (um)
    L (um)
    perimeter (um)
    cross_sectional_area (um2)
    er_area (um2)
    er_volume (um3)
    ica       (mA/cm2)
    ica_pmp   (mA/cm2)
    ica_pmp_last   (mA/cm2)
    parea     (um)     : pump area per unit length

    sump      (mM)

    cai       (mM)
    :cao       (mM)

    nai
    ki
    ko

    vrat[Nannuli]  (1) : dimensionless
                        : numeric value of vrat[i] equals the volume
                        : of annulus i of a 1um diameter cylinder
                        : multiply by diam^2 to get volume per um length

    bufs_0 (mM)
    bufm_0 (mM)

    ip3i   (mM)

    LL[Nannuli] (mM/ms) : 0.1e-6 mM/ms nominally, but adjusted so that
    : jchnl + jpump + jleak = 0  when  ca = 0.05 uM and h = Kinh/(ca + Kinh)
   glu2 :variable for synaptic stimulation

   icaER
   icaPLASMA

}

CONSTANT { volo = 1e10 (um2) }

STATE {
    : ca[0] is equivalent to cai
    : ca[] are very small, so specify absolute tolerance
    : let it be ~1.5 - 2 orders of magnitude smaller than baseline level
    ca[Nannuli]       (mM) <1e-7>
    bufs[Nannuli]    (mM) <1e-3>
    cabufs[Nannuli]  (mM) <1e-7>
    bufm[Nannuli]    (mM) <1e-4>
    cabufm[Nannuli]  (mM) <1e-8>
    hc[Nannuli]
    ho[Nannuli]

    gluExt
    eet (mM)

    : IP3i
    eATP
    glutReceived

}

BREAKPOINT {


    :if (trigger){
    if (glu2>glutReceived){
        ca[0]=60e-3


        :ca[0]=ca[0]+1e-4

        : ca[0]=2e-3 :standard trigger
    :    ca[0]=2.0 :needed to match farr paper, very wrong too high
  :IP3i=5e-3
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


:        glu2=0
        glutReceived=glu2
        trigger=1
    }

    reporter=ca[0]

    SOLVE state METHOD sparse
    ica_pmp_last = ica_pmp
    ica = ica_pmp

    reportcai=cai/4992*1000000
}

LOCAL factors_done, jx

INITIAL {
    glutReceived=0
    if (factors_done == 0) {  : flag becomes 1 in the first segment
        factors_done = 1       :   all subsequent segments will have
        factors()              :   vrat = 0 unless vrat is GLOBAL
    }
    ip3i = ip3i0
    :cai = cai0


    bufs_0 = KDs*TBufs/(KDs + (1000)*cai0)
    bufm_0 = KDm*TBufm/(KDm + (1000)*cai0)

    FROM i=0 TO Nannuli-1 {
        ca[i] = cai
        bufs[i] = bufs_0
        cabufs[i] = TBufs - bufs_0
        bufm[i] = bufm_0
        cabufm[i] = TBufm - bufm_0
    }

    sump = cath
    :parea = PI*diam
    diam1=diam
    parea = PI*diam1*perimFactor

    : reconsider and revise initialization comments
    ica=0
    ica_pmp = 0
    ica_pmp_last = 0
    : If there is a voltage-gated calcium current,
    : this is almost certainly the wrong initialization.
    : In such a case, first do an initialization run, then use SaveState
    : On subsequent runs, restore the initial condition from the saved states.

    FROM i=0 TO Nannuli-1 {
        ho[i] = Kinh/(ca[i]+Kinh)
        hc[i] = 1-ho[i]

        : jx = jp + jc
        : choose L so that jl = -jx
        : jl = L*(1 - (ca[i]/caer))
        : jp = (-vmax*ca[i]^2 / (ca[i]^2 + Kp^2))
        : jc = jmax*(1-(ca[i]/caer)) * ( (ip3i/(ip3i+Kip3)) * (ca[i]/(ca[i]+Kact)) * ho[i] )^3
        jx = (-vmax*ca[i]^2 / (ca[i]^2 + Kp^2))
        jx = jx + jmax*(1-(ca[i]/caer)) * ( (ip3i/(ip3i+Kip3)) * (ca[i]/(ca[i]+Kact)) * ho[i] )^3
        LL[i] = -jx/(1 - (ca[i]/caer))
    }
eet=6

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

LOCAL dsq, dsqvol   : can't define local variable in KINETIC block
                    :   or use in COMPARTMENT statement

KINETIC state {
    COMPARTMENT i, diam*diam*vrat[i] {ca bufs cabufs bufm cabufm sump}
    COMPARTMENT volo {cao}
    LONGITUDINAL_DIFFUSION i, DCa*diam*diam*vrat[i] {ca}
    LONGITUDINAL_DIFFUSION i, DBufm*diam*diam*vrat[i] {bufm cabufm}

    : cell membrane ca pump
    ~ ca[0] <-> sump  ((0.001)*parea*gamma*u(ca[0]/(1 (mM)), cath/(1 (mM))), (0.001)*parea*gamma*u(ca[0]/(1 (mM)), cath/(1 (mM))))
    ica_pmp = 2*FARADAY*(f_flux - b_flux)/parea

    : all currents except cell membrane ca pump
    ~ ca[0] << (-(ica - ica_pmp_last)*PI*diam/(2*FARADAY))  : ica is Ca efflux
    : radial diffusion
    FROM i=0 TO Nannuli-2 {
  : if (ca[i] < cai0/2) {
  :    ca[i] = cai0/2
  : }
        ~ ca[i] <-> ca[i+1]  (DCa*frat[i+1], DCa*frat[i+1])
        ~ bufm[i] <-> bufm[i+1]  (DBufm*frat[i+1], DBufm*frat[i+1])
    }
    : buffering
    dsq = diam*diam
    FROM i=0 TO Nannuli-1 {
        dsqvol = dsq*vrat[i]
        ~ ca[i] + bufs[i] <-> cabufs[i]  (kfs*dsqvol, (0.001)*KDs*kfs*dsqvol)
        ~ ca[i] + bufm[i] <-> cabufm[i]  (kfm*dsqvol, (0.001)*KDm*kfm*dsqvol)
    }
    : SERCA pump, channel, and leak
    FROM i=0 TO Nannuli-1 {
        dsqvol = dsq*vrat[i]
        : pump
        ~ ca[i] << (-dsqvol*alpha*vmax*ca[i]^2 / (ca[i]^2 + Kp^2))
        : channel
        ~ hc[i] <-> ho[i]  (kon*Kinh, kon*ca[i])
        ~ ca[i] << ( dsqvol*alpha*jmax*(1-(ca[i]/caer)) * ( (ip3i/(ip3i+Kip3)) * (ca[i]/(ca[i]+Kact)) * ho[i] )^3 )
        : leak
        ~ ca[i] << (dsqvol*alpha*LL[i]*(1 - (ca[i]/caer)))
    }

    cai = ca[0]
    fluo = cabufm[0]
    fluoNew = (BufferAlpha * cabufm[0] + ca[0] - BufferAlpha*(TBufm - bufm_0) - cai0)/(BufferAlpha*(TBufm - bufm_0) + cai0)
}

FUNCTION u(x, th) {
    if (x>th) {
        u = 1
    } else {
        u = 0
    }
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
