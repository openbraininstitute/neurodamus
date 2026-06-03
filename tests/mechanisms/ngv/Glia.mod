TITLE SBML model: glia_2013 generated from file: Glia.xml
? this file has been modified so that units are in terms of ms not seconds
? should correspond to Renaud's working version
? it also uses extracellular ion dynamics from Wei et al. 2014 J Neurosci.
? and a model of astrocyte calcium from Bennett 2008

UNITS {
  (mA) = (milliamp)
  (mV) = (millivolt)
}

NEURON {
  THREADSAFE
  SUFFIX glia_2013
  BBCOREPOINTER glu2
  RANGE unit_compartment
  RANGE V
  RANGE Fin
  RANGE Jstimg
  RANGE Jstimn
  RANGE ADPn
  RANGE ADPg
  RANGE O2meanc
  RANGE Vv0
  RANGE JleakNan
  RANGE JleakNag
  RANGE Jpumpn
  RANGE Jpumpg
  RANGE JGLCce
  RANGE JGLCcg
  RANGE JGLCeg
  RANGE JGLCen
  RANGE JLACec
  RANGE JLACcg
  RANGE JLACeg
  RANGE JLACen
  RANGE JGLCec
  RANGE JGLCgc
  RANGE JGLCge
  RANGE JGLCne
  RANGE JLACce
  RANGE JLACgc
  RANGE JLACge
  RANGE JLACne
  RANGE JHKPFKn
  RANGE JHKPFKg
  RANGE JPGKn
  RANGE JPGKg
  RANGE JPKn
  RANGE JPKg
  RANGE JLDHn
  RANGE JLDHg
  RANGE Jmitoing
  RANGE Jmitoinn
  RANGE Jmitooutg
  RANGE Jmitooutn
  RANGE Jshuttleg
  RANGE Jshuttlen
  RANGE JCKg
  RANGE JCKn
  RANGE JO2mcg
  RANGE JO2mcn
  RANGE JO2c
  RANGE JGLCc
  RANGE JLACc
  RANGE IL
  RANGE INa
  RANGE IK
  RANGE ICa
  RANGE ImAHP
  RANGE Ipump
  RANGE Fout
  RANGE dAMPdATPn
  RANGE dAMPdATPg
  RANGE Ve
  RANGE Vcap
  RANGE Vg
  RANGE Vn
  RANGE zeta
  RANGE qAK
  RANGE A
  RANGE un
  RANGE ug

  RANGE rce
  RANGE rcn
  RANGE rcg
  RANGE Isyn
  RANGE kCKplusn
  RANGE kCKplusg
  RANGE Nan0
  RANGE F0
  RANGE SmVn
  RANGE SmVg
  RANGE R
  RANGE F
  RANGE RToverF
  RANGE psig
  RANGE KtGLCen
  RANGE KtGLCeg
  RANGE KtGLCcg
  RANGE KtGLCce
  RANGE KtLACen
  RANGE KtLACeg
  RANGE KtLACge
  RANGE KtLACgc
  RANGE KtLACcg
  RANGE KtLACec
  RANGE KIATP
  RANGE nH1
  RANGE KGLCg
  RANGE KO2
  RANGE HbOP
  RANGE nh
  RANGE KO2mito
  RANGE Cm
  RANGE gL
  RANGE gNa
  RANGE gK
  RANGE gCa
  RANGE gmAHP
  RANGE KD
  RANGE tauCa
  RANGE Ca0
  RANGE EK
  RANGE ECa
  RANGE phih
  RANGE phin
  RANGE tauv
  RANGE alphav
  RANGE O2a
  RANGE GLCa
  RANGE gNan
  RANGE gNag
  RANGE gKpas
  RANGE kpumpn
  RANGE kpumpg
  RANGE Kmpump
  RANGE C
  RANGE N
  RANGE Kmmito
  RANGE kLDHnplus
  RANGE kLDHnminus
  RANGE kLDHgplus
  RANGE kLDHgminus
  RANGE Mcyton
  RANGE Mcytog
  RANGE Mmiton
  RANGE Mmitog
  RANGE KmADPn
  RANGE KmADPg
  RANGE KmNADn
  RANGE KmNADg
  RANGE KmNADHn
  RANGE KmNADHg
  RANGE kCKn
  RANGE kCKg
  RANGE mNADg
  RANGE TmaxGLCen
  RANGE TmaxGLCce
  RANGE TmaxGLCeg
  RANGE TmaxGLCcg
  RANGE TmaxLACgc
  RANGE TmaxLACcg
  RANGE TmaxLACen
  RANGE TmaxLACne
  RANGE TmaxLACge
  RANGE TmaxLACeg
  RANGE TmaxLACec
  RANGE kHKPFKn
  RANGE kHKPFKg
  RANGE PScapoverVn
  RANGE PScapoverVg
  RANGE Vmaxoutn
  RANGE Vmaxoutg
  RANGE Vmaxinn
  RANGE Vmaxing
  RANGE kPGKn
  RANGE kPGKg
  RANGE kPKn
  RANGE kPKg
  RANGE JATPasesn
  RANGE JATPasesg
  RANGE kCKminusn
  RANGE kCKminusg
  RANGE TNADHn
  RANGE TNADHg
  RANGE LACa
  RANGE Rplusg
  RANGE Rminusg
  RANGE Rplusn
  RANGE Rminusn
  RANGE hh_alphan
  RANGE hh_alpham
  RANGE hh_betam
  RANGE hh_alphah
  RANGE hh_betah
  RANGE hh_betan
  RANGE hh_minfinity
  RANGE hh_ninfinity
  RANGE hh_hinfinity
  RANGE hh_taun
  RANGE hh_mCa
  RANGE hh_tauh
  RANGE hh_EL
  RANGE Finprime
  RANGE JGLUeg
  RANGE vPumpg0

  RANGE Ipumpnak
  RANGE Igliapump
  RANGE Iglia
  RANGE p
  RANGE Gglia
  RANGE volninfinity
  RANGE pio
  RANGE pii
  RANGE volo
  RANGE Ikcc2
  RANGE Inkcc1
  RANGE Ik
  RANGE Ina

  RANGE rhomax
  RANGE Ggliamax
  ? RANGE vtotal
  RANGE voln0
  RANGE volo0
  RANGE volg0

  RANGE Icil
  RANGE ecl
  RANGE Ukcc2
  RANGE Unkcc1
  RANGE Aminuse
  RANGE Aminusn

  RANGE Nag
  RANGE GLCn
  RANGE GLCg
  RANGE GAPn
  RANGE GAPg
  RANGE PEPn
  RANGE PEPg
  RANGE PYRn
  RANGE PYRg
  RANGE LACn
  RANGE LACg
  RANGE NADHcyton
  RANGE NADHcytog
  RANGE NADHmiton
  RANGE NADHmitog
  RANGE ATPn
  RANGE ATPg
  RANGE PCrn
  RANGE PCrg
  RANGE GLCe
  RANGE LACe
  RANGE O2n
  RANGE O2g

  RANGE Ke
  RANGE Cle
  RANGE Nae
  RANGE Kn
  RANGE Cln
  RANGE Nan
  RANGE Cae

  RANGE fo
  RANGE fv
  RANGE dslp
  RANGE beta
  RANGE Obath
  RANGE Kbath

  ?new range variables for Bennett 2008
  RANGE rhoG
  RANGE JpumpG1
  RANGE JleakG
  RANGE gating1
  RANGE JIP3

  RANGE pMusc
  RANGE iCaMusc
  RANGE kMyosin1
}

ASSIGNED {
  glu2
}

PARAMETER {
  unit_compartment = 1.0
  V = 1e-07

  Fin = 0.012
  Jstimg = 0
  Jstimn = 0
  ADPn = 1e-10
  ADPg = 1e-10
  O2meanc = 7.0
  Vv0 = 0.021 ?xxxxxxxxxxxx
  JleakNan = 0.0
  JleakNag = 0.0
  Jpumpn = 0.0
  Jpumpg = 0.0
  JGLCce = 1e-10
  JGLCcg = 1e-10
  JGLCeg = 1e-10
  JGLCen = 1e-10
  JLACec = 1e-10
  JLACcg = 1e-10
  JLACeg = 1e-10
  JLACen = 1e-10
  JGLCec = 1e-10
  JGLCgc = 1e-10
  JGLCge = 1e-10
  JGLCne = 1e-10
  JLACce = 1e-10
  JLACgc = 1e-10
  JLACge = 1e-10
  JLACne = 1e-10
  JHKPFKn = 0.0
  JHKPFKg = 0.0
  JPGKn = 0.0
  JPGKg = 0.0
  JPKn = 0.0
  JPKg = 0.0
  JLDHn = 0.0
  JLDHg = 0.0
  Jmitoing =  0
  Jmitoinn =  0
  Jmitooutg =  0
  Jmitooutn =  0
  Jshuttleg = 0.0
  Jshuttlen = 0.0
  JCKg = 0.0
  JCKn = 0.0
  JO2mcg = 1e-10
  JO2mcn = 1e-10
  JO2c = 0.0
  JGLCc = 0.0
  JLACc = 0.0
  IL = 0.0
  INa = 0.0
  IK = 0.0
  ICa = 0.0
  ImAHP = 1e-10
  Ipump = 1e-10
  Fout = 0.012
  dAMPdATPn = 0.0
  dAMPdATPg = 0.0

  Ve = 0.2
  Vcap = 0.0055
  Vg = 0.08
  Vn = 0.6

  zeta = 0.07
  qAK = 0.92
  A = 2.212
  un = 0.0
  ug = 0.0

  rce = 0.0275
  rcn = 0.0122
  rcg = 0.0220
  Isyn =0? 10.8
  kCKplusn =   0.0433
  kCKplusg = 0.00135

  Nan0 = 8.0
  F0 = 0.012

  SmVn = 25000.0
  SmVg = 25000.0
  R = 8.314151
  F = 96485.3
  RToverF = 26.73
  psig = -70.0

  KtGLCen = 8.0
  KtGLCeg = 8.0
  KtGLCcg = 8.0
  KtGLCce = 8.0
  KtLACen = 0.74
  KtLACeg = 3.5
  KtLACge = 3.5
  KtLACgc = 1.0
  KtLACcg = 1.0
  KtLACec = 1.0
  KIATP = 1.0
  nH1 = 4.0
  KGLCg = 0.05
  KO2 = 0.0361
  HbOP = 8.6
  nh = 2.73
  KO2mito = 0.001
  Cm = 0.001
  gL = 0.02
  gNa = 40.0
  gK = 18.0
  gCa = 0.02
  gmAHP = 6.5
  KD = 0.03
  tauCa = 0.15
  Ca0 = 5e-05
  EK = -71.0
  ECa = 120.0
  ?phih = 4.0
  phih = 4.0
  ?phin = 4.0
  phin = 4.0
  tauv = 35.0
  alphav = 0.5
  O2a = 8.35
  GLCa = 4.75
  gNan =    0.01 : 0.01 0.008 0.001 0.03 0.025 0.02 0.032 to 0.0335  0.0136
  gNag =    0.0061
  gKpas =    0.2035
  kpumpn =   2.2000e-06
  kpumpg =   4.5000e-07
  Kmpump = 0.5
  C = 10.0
  N = 0.212
  Kmmito = 0.04
  kLDHnplus =   72.3000
  kLDHnminus =    0.7200
  kLDHgplus =    1.5900
  kLDHgminus =    0.0710
  Mcyton =   4.9000e-08
  Mcytog =   2.5000e-04
  Mmiton = 393000
  Mmitog =       10600
  KmADPn =    0.00341
  KmADPg =   4.8300e-04
  KmNADHn =    0.0444
  KmNADHg =    0.0269
  kCKn =    0.0433
  kCKg =    0.00135
  KmNADn =    0.4090
  KmNADg =   40.3000
  TmaxGLCen =    0.0410
  TmaxGLCne =    0.0410
  TmaxGLCce =    0.2390
  TmaxGLCec =    0.2390
  TmaxGLCeg =    0.1470
  TmaxGLCge =    0.1470
  TmaxGLCcg =    0.0016

  TmaxLACcg = 0.00243
  TmaxLACne =   24.3000
  TmaxLACen =   24.3000
  TmaxLACge =  106.1000
  TmaxLACeg =  106.1000

  TmaxLACec =    0.2500
  TmaxLACce =    0.2500
  kHKPFKn = 0.050435
  kHKPFKg =    0.1850
  PScapoverVn =    1.6600
  PScapoverVg =    0.8700
  Vmaxoutn = 0.1640
  Vmaxoutg = 0.0640
  Vmaxinn =0.1303
  Vmaxing =5.7000
  kPGKn =    3.9700
  kPGKg =  401.7000
  kPKn =   36.7000
  kPKg =  135.2000
  JATPasesn =    0.1695
  JATPasesg =    0.1404
  kCKminusn =   2.8000e-04
  kCKminusg =   1.0000e-05
  TNADHn =       10330
  TNADHg =   150
  LACa =    0.5060
  Rplusg = 0.0
  Rminusg = 0.0
  Rplusn = 0.0
  Rminusn = 0.0
  hh_alphan = 0.0
  hh_alpham = 0.0
  hh_betam = 0.0
  hh_alphah = 0.0
  hh_betah = 0.0
  hh_betan = 0.0
  hh_minfinity = 0.0
  hh_ninfinity = 0.0
  hh_hinfinity = 0.0
  hh_taun = 1.0
  hh_mCa = 0.0
  hh_tauh = 1.0
  hh_EL = 0.0
  Finprime=0
  JGLUeg=0
  vPumpg0 =    0.0687

  Ipumpnak=0
  Igliapump=0
  Iglia=0
  p=1
  Gglia=0
  pio=0
  pii=0
  ? calculate volumes
  ? in 100 um cube, assume neurons are 60% and extracellular is 20%
  ?then cube volume is 1e-12

  volninfinity=0.6e-12 ? unit:m^3
  voln0=0.6e-12
  volo=0.2e-12
  volo0=0.2e-12
  volg0=0.08e-12

  Ukcc2 =0.019?     0.3 ?0.0189 kn keeps increasing 0.019 kn decreases
  Unkcc1 =0.0063333?  0.1 ?mM/s
  Aminuse =18?mM  was 132
  Aminusn = 142.1009 ?mM was 18

  Ik=0
  Ikcc2=0
  Ina=0
  Icil=0
  ecl=0
  Inkcc1=0
  Ggliamax=5
  rhomax=0.8 ?maximal pump rate

  Ke=4
  Kn=140
  Cle=130
  Cln=6
  Nae=144
  Cae=1

  Nan= 12.4561
  Nag= 16.5746
  GLCn= 1.19524
  GLCg= 1.39161
  GAPn= 9.74448e-05
  GAPg= 0.000690546
  PEPn= 0.00124715
  PEPg= 0.0017559
  PYRn= 0.254503
  PYRg= 0.180087
  LACn= 0.270329
  LACg= 0.371995
  NADHcyton= 0.0017769
  NADHcytog= 0.114224
  ATPn= 1.84582
  ATPg= 2.02548
  PCrn= 2.61273
  PCrg= 4.88877
  O2n= 0.0140514
  O2g= 0.0192578

  GLCe= 2.90865
  LACe= 0.270835
  NADHmiton= 0.114473
  NADHmitog= 0.125153

  fo=1
  fv=1
  dslp=0.25
  beta=1
  Obath = 32
  Kbath=4 ?was 4 and 4.63 5.5

  ?new parameters for Bennett 2008
  VmaxG = 0.2 ?mM/s max pumping rate into ER
  KGlu=0.1
  Kpg = 0.000244 ?mM pump dissoc constant
  PlG = 0.01? flux constant, needs tuning
  CagER =0.4 ?mM, Ca2+ concentration in ER
  BcytoG  = 0.0244 ?Endogenous buffer parameter
  kGlutamateDecay = 0.001 ?s-1 glutamate decay rate
  kmGluR_forward = 1.000
  kmGluR_backward = 0.666 ? =kmGluR_forward  * Kglu but Kglu unknown
  Gdelta = 0?ratio of the activities of the unbound and bound receptors
  kplusG=1.000 ?on rate for g protein
  kminusG = 8.820  ?=Kg* ka since KG =kd/ka
  Grh = 0.0100 ? mM um-2 s-1 IP3 production rate
  kdeg_IP3 =1.25 ?s-1 IP3 degradation rate
  konG = 0.002 ?mM /s IP3 channelkinetic parameter
  Kinh = 0.0001 ?mMIP3 channel kinetic parameter
  VEET =3e-6 ? mM um-2 s-1 Production rate
  Cagmin = 0.0001 ?mM Minimum Ca2+for production
  KI = 0.00003 ?mM IP3 channel kinetic parameter
  Kact = 0.00017 ? mM IP3 channel kinetic parameter
  Jmax =50000000000 ? mM/s Maximum channel current

  rhoG=0
  JpumpG1=0
  JleakG=0
  gating1=0
  JIP3=0

  pMusc=0
  iCaMusc=0
  kMyosin1=0

  kMusc= 9.5  ? mV Open probability parameter
  kappaMusc= 4.1  ? pAV-1 Single-channel current parameter
  gammaEET= 400 ? V uM-1 s-1 EET conversion factor
  VhMusc= 6.3 ?mV open probability parameter
  VAveMusc= 1.44 ?mV Single-channel current parameter
  alphaMusc= 0.003 ?these constants are not given
  betaMusc= 400 ?these constants are not given
  CaMusc0= 0.00005 ?mM Background Ca0
  kMyosin=80 ?uM-4s-1 Myosin dephosphorylation rate
  nMusc =4 ?power
  kMyosin2=20 ?s-1 Myosin dephosphorylation rate
  kMyosin3=  4 ?s-1 Actin binding rate
  kMyosin4=  1 ?s-1 Actin binding rate
}

STATE {
  NNag
  NGLCn
  NGLCg
  NGAPn
  NGAPg
  NPEPn
  NPEPg
  NPYRn
  NPYRg
  NLACn
  NLACg
  NNADHcyton
  NNADHcytog
  NNADHmiton
  NNADHmitog
  NATPn
  NATPg
  NPCrn
  NPCrg
  NGLCe
  NLACe
  NO2n
  NO2g
  NKe
  NKn
  NCle
  NCln
  NNae
  NNan
  NCae

  O2c
  GLCc
  LACc
  Vv
  Hb
  psin
  h
  n
  Ca
  Vvp

  ?Kg
  ?Clg

  O2e
  voln

  ?new states for Bennett
  Glu
  mGluR_inactive
  mGluR_active
  Ginactive
  Gactive
  IP3
  Cag
  hIP3
  EET

  CaMusc
  Myosin
  MyosinP
  ActinP
  VMusc
}

INITIAL {
  Fout= 0.012
  Vvp= -5.22695e-008



  O2e=32 ?mg/L this needs to be converted to a molar concentration later

  voln=0.6e-12


  NGLCn= 1.19524*voln0
  NGLCg= 1.46601*volg0
  NGAPn= 0.00102725*voln0
  NGAPg= 0.00402778*volg0
  NPEPn= 0.00154052*voln0
  NPEPg= 0.006917*volg0
  NPYRn= 0.301334*voln0
  NPYRg= 0.163207*volg0
  NLACn= 2.0888*voln0
  NLACg= 2.08478*volg0
  NNADHcyton= 0.0142633*voln0
  NNADHcytog= 0.134347*volg0
  NATPn= 1.247*voln0
  NATPg= 2.17581*volg0
  NPCrn= 0.127424*voln0
  NPCrg= 3.55308*volg0
  NO2n=  0.0136719*voln0
  NO2g= 0.0195101*volg0
  NGLCe= 2.89486*volo0
  NLACe=  2.08286*volo0
  NNADHmiton= 0.115507*voln0
  NNADHmitog= 0.124814*volg0
  O2c=6.53158
  GLCc=4.55546
  LACc=0.918012
  Vv=0.0210009
  Hb=0.0575034
  psin=-66.7883
  h=0.976053
  n= 0.0396341
  Ca= 5.43815e-05

  NNag=1.22231e-12
  NKe=7.88428e-13
  NKn=8.59001e-11
  NNae=3.18377e-11
  NNan=7.97988e-12
  NCle=2.53824e-11
  NCln=5.4524e-12
  NCae=2e-13



  ?new states for Bennett
  Glu = 0
  mGluR_inactive = 1
  mGluR_active = 0
  Ginactive =1
  Gactive =0
  IP3 =0
  Cag =0
  hIP3 =0
  EET =0

  CaMusc=0.0003
  Myosin=1
  MyosinP=0
  ActinP=0
  VMusc=-30
}

BREAKPOINT {
  SOLVE states METHOD euler
  ? Need to check order in which assignments/event assignments should be updated!!!

  ? Assignment rule here: Rminusn = NADHcyton / (N - NADHcyton)
  Rminusn = NADHcyton / (N - NADHcyton)

  ? Assignment rule here: Rplusn = (N - NADHmiton) / NADHmiton
  Rplusn = (N - NADHmiton) / NADHmiton

  ? Assignment rule here: Rminusg = NADHcytog / (N - NADHcytog)
  Rminusg = NADHcytog / (N - NADHcytog)

  ? Assignment rule here: Rplusg = (N - NADHmitog) / NADHmitog
  Rplusg = (N - NADHmitog) / NADHmitog

  ? Assignment rule here: JleakNag = SmVg / F * gNag * (RToverF * log(Nae / Nag) - psig)
  JleakNag = SmVg / F * gNag * (RToverF * log(Nae / Nag) - psig)

  ? Assignment rule here: JleakNan = SmVn / F * gNan * (RToverF * log(Nae / Nan) - psin)
  JleakNan = SmVn / F * gNan * (RToverF * log(Nae / Nan) - psin)

  ? Assignment rule here: Jpumpg = SmVg * kpumpg * ATPg * Nag * pow(1 + ATPg / Kmpump, -1)
  Jpumpg = SmVg * kpumpg * ATPg * Nag * pow(1 + ATPg / Kmpump, -1)

  ? Assignment rule here: Jpumpn = SmVn * kpumpn * ATPn * Nan * pow(1 + ATPn / Kmpump, -1)
  Jpumpn = SmVn * kpumpn * ATPn * Nan * pow(1 + ATPn / Kmpump, -1)

  ? Assignment rule here: JGLCce = TmaxGLCce * (GLCc / (GLCc + KtGLCce) - GLCe / (GLCe + KtGLCce))
  JGLCce = TmaxGLCce * (GLCc / (GLCc + KtGLCce) - GLCe / (GLCe + KtGLCce))

  ? Assignment rule here: JGLCcg = TmaxGLCcg * (GLCc / (GLCc + KtGLCcg) - GLCg / (GLCg + KtGLCcg))
  JGLCcg = TmaxGLCcg * (GLCc / (GLCc + KtGLCcg) - GLCg / (GLCg + KtGLCcg))

  ? Assignment rule here: JGLCeg = TmaxGLCeg * (GLCe / (GLCe + KtGLCeg) - GLCg / (GLCg + KtGLCeg))
  JGLCeg = TmaxGLCeg * (GLCe / (GLCe + KtGLCeg) - GLCg / (GLCg + KtGLCeg))

  ? Assignment rule here: JGLCen = TmaxGLCen * (GLCe / (GLCe + KtGLCen) - GLCn / (GLCn + KtGLCen))
  JGLCen = TmaxGLCen * (GLCe / (GLCe + KtGLCen) - GLCn / (GLCn + KtGLCen))

  ? Assignment rule here: JHKPFKn = kHKPFKn * ATPn * (GLCn / (GLCn + KGLCg)) * pow(1 + pow(ATPn / KIATP, nH1), -1)
  JHKPFKn = kHKPFKn * ATPn * (GLCn / (GLCn + KGLCg)) * pow(1 + pow(ATPn / KIATP, nH1), -1)

  ? Assignment rule here: JHKPFKg = kHKPFKg * ATPg * (GLCg / (GLCg + KGLCg)) * pow(1 + pow(ATPg / KIATP, nH1), -1)
  JHKPFKg = kHKPFKg * ATPg * (GLCg / (GLCg + KGLCg)) * pow(1 + pow(ATPg / KIATP, nH1), -1)

  ? Assignment rule here: ADPn = ATPn / 2 * (pow(qAK * qAK + 4 * qAK * (A / ATPn - 1), 0.5) - qAK)
  ADPn = ATPn / 2 * (pow(qAK * qAK + 4 * qAK * (A / ATPn - 1), 0.5) - qAK)

  ? Assignment rule here: ADPg = ATPg / 2 * (pow(qAK * qAK + 4 * qAK * (A / ATPg - 1), 0.5) - qAK)
  ADPg = ATPg / 2 * (pow(qAK * qAK + 4 * qAK * (A / ATPg - 1), 0.5) - qAK)

  ? Assignment rule here: JPGKn = kPGKn * GAPn * ADPn * ((N - NADHcyton) / NADHcyton)
  JPGKn = kPGKn * GAPn * ADPn * ((N - NADHcyton) / NADHcyton)

  ? Assignment rule here: JPGKg = kPGKg * GAPg * ADPg * ((N - NADHcytog) / NADHcytog)
  JPGKg = kPGKg * GAPg * ADPg * ((N - NADHcytog) / NADHcytog)

  ? Assignment rule here: JPKn = kPKn * PEPn * ADPn
  JPKn = kPKn * PEPn * ADPn

  ? Assignment rule here: JPKg = kPKg * PEPg * ADPg
  JPKg = kPKg * PEPg * ADPg

  ? Assignment rule here: JLDHn = kLDHnplus * PYRn * NADHcyton - kLDHnminus * LACn * (N - NADHcyton)
  JLDHn = kLDHnplus * PYRn * NADHcyton - kLDHnminus * LACn * (N - NADHcyton)

  ? Assignment rule here: JLDHg = kLDHgplus * PYRg * NADHcytog - kLDHgminus * LACg * (N - NADHcytog)
  JLDHg = kLDHgplus * PYRg * NADHcytog - kLDHgminus * LACg * (N - NADHcytog)

  ? Assignment rule here: JLACec = TmaxLACec * (LACe / (LACe + KtLACec) - LACc / (LACc + KtLACec))
  JLACec = TmaxLACec * (LACe / (LACe + KtLACec) - LACc / (LACc + KtLACec))

  ? Assignment rule here: JLACcg = TmaxLACcg * (LACc / (LACc + KtLACcg) - LACg / (LACg + KtLACcg))
  JLACcg = TmaxLACcg * (LACc / (LACc + KtLACcg) - LACg / (LACg + KtLACcg))

  ? Assignment rule here: JLACeg = TmaxLACeg * (LACe / (LACe + KtLACeg) - LACg / (LACg + KtLACeg))
  JLACeg = TmaxLACeg * (LACe / (LACe + KtLACeg) - LACg / (LACg + KtLACeg))

  ? Assignment rule here: JLACen = TmaxLACen * (LACe / (LACe + KtLACen) - LACn / (LACn + KtLACen))
  JLACen = TmaxLACen * (LACe / (LACe + KtLACen) - LACn / (LACn + KtLACen))

  ?note this is KmNADg not KmNADHg
  ? Assignment rule here: Jmitoing = Vmaxing * PYRg / (PYRg + Kmmito) * ((N - NADHmitog) / (N - NADHmitog + KmNADg))
  Jmitoing = Vmaxing * PYRg / (PYRg + Kmmito) * ((N - NADHmitog) / (N - NADHmitog + KmNADg))

  ?note this is KmNADn not KmNADHn
  ? Assignment rule here: Jmitoinn = Vmaxinn * PYRn / (PYRn + Kmmito) * ((N - NADHmiton) / (N - NADHmiton + KmNADn))
  Jmitoinn = Vmaxinn * PYRn / (PYRn + Kmmito) * ((N - NADHmiton) / (N - NADHmiton + KmNADn))

  ? Assignment rule here: Jmitooutn = Vmaxoutn * O2n / (O2n + KO2mito) * ADPn / (ADPn + KmADPn) * (NADHmiton / (NADHmiton + KmNADHn))
  Jmitooutn = Vmaxoutn * O2n / (O2n + KO2mito) * ADPn / (ADPn + KmADPn) * (NADHmiton / (NADHmiton + KmNADHn))

  ? Assignment rule here: Jmitooutg = Vmaxoutg * O2g / (O2g + KO2mito) * ADPg / (ADPg + KmADPg) * (NADHmitog / (NADHmitog + KmNADHg))
  Jmitooutg = Vmaxoutg * O2g / (O2g + KO2mito) * ADPg / (ADPg + KmADPg) * (NADHmitog / (NADHmitog + KmNADHg))

  ? Assignment rule here: Jshuttlen = TNADHn * Rminusn / (Mcyton + Rminusn) * (Rplusn / (Mmiton + Rplusn))
  Jshuttlen = TNADHn * Rminusn / (Mcyton + Rminusn) * (Rplusn / (Mmiton + Rplusn))

  ? Assignment rule here: Jshuttleg = TNADHg * Rminusg / (Mcytog + Rminusg) * (Rplusg / (Mmitog + Rplusg))
  Jshuttleg = TNADHg * Rminusg / (Mcytog + Rminusg) * (Rplusg / (Mmitog + Rplusg))

  ? Assignment rule here: JCKn = kCKplusn * PCrn * ADPn - kCKminusn * (C - PCrn) * ATPn
  JCKn = kCKplusn * PCrn * ADPn - kCKminusn * (C - PCrn) * ATPn

  ? Assignment rule here: JCKg = kCKplusg * PCrg * ADPg - kCKminusg * (C - PCrg) * ATPg
  JCKg = kCKplusg * PCrg * ADPg - kCKminusg * (C - PCrg) * ATPg

  ? Assignment rule here: JO2mcn = PScapoverVn * (KO2 * pow(HbOP / O2c - 1, -1 / nh) - O2n)
  JO2mcn = PScapoverVn * (KO2 * pow(HbOP / O2c - 1, -1 / nh) - O2n)

  ? Assignment rule here: JO2mcg = PScapoverVg * (KO2 * pow(HbOP / O2c - 1, -1 / nh) - O2g)
  JO2mcg = PScapoverVg * (KO2 * pow(HbOP / O2c - 1, -1 / nh) - O2g)

  ? Assignment rule here: JO2c = 2 * Fin / Vcap * (O2a - O2c)
  JO2c = 2 * Fin / Vcap * (O2a - O2c)

  ? Assignment rule here: JGLCc = 2 * Fin / Vcap * (GLCa - GLCc)
  JGLCc = 2 * Fin / Vcap * (GLCa - GLCc)

  ? Assignment rule here: JLACc = 2 * Fin / Vcap * (LACa - LACc)
  JLACc = 2 * Fin / Vcap * (LACa - LACc)

  ? Assignment rule here: O2meanc = 2 * O2c - O2a
  O2meanc = 2 * O2c - O2a

  ? Assignment rule here: hh_EL = (gKpas * EK + gNan * RToverF * log(Nae / Nan)) / (gKpas + gNan)
  hh_EL = (gKpas * EK + gNan * RToverF * log(Nae / Nan)) / (gKpas + gNan)

  ? Assignment rule here: IL = gL * (psin - hh_EL)
  IL = gL * (psin - hh_EL)

  ? Assignment rule here: hh_alpham = -0.1 * ((psin + 33) / (exp(-0.1 * (psin + 33)) - 1))
  hh_alpham = -0.1 * ((psin + 33) / (exp(-0.1 * (psin + 33)) - 1))

  ? Assignment rule here: hh_betam = 4 * exp((psin + 58) / -12)
  hh_betam = 4 * exp((psin + 58) / -12)

  ? Assignment rule here: hh_alphah = 0.07 * exp((0 - (psin + 50)) / 10)
  hh_alphah = 0.07 * exp((0 - (psin + 50)) / 10)

  ? Assignment rule here: hh_betah = 1 / (exp(-0.1 * (psin + 20)) + 1)
  hh_betah = 1 / (exp(-0.1 * (psin + 20)) + 1)

  ? Assignment rule here: hh_alphan = -0.01 * ((psin + 34) / (exp(-0.1 * (psin + 34)) - 1))
  hh_alphan = -0.01 * ((psin + 34) / (exp(-0.1 * (psin + 34)) - 1))

  ? Assignment rule here: hh_betan = 0.125 * exp((0 - (psin + 44)) / 25)
  hh_betan = 0.125 * exp((0 - (psin + 44)) / 25)

  ? Assignment rule here: hh_mCa = 1 / (exp((psin + 20) / -9) + 1)
  hh_mCa = 1 / (exp((psin + 20) / -9) + 1)

  ? Assignment rule here: hh_tauh = 0.001 / (hh_alphah + hh_betah)
  hh_tauh = 0.001 / (hh_alphah + hh_betah)

  ? Assignment rule here: hh_minfinity = hh_alpham / (hh_alpham + hh_betam)
  hh_minfinity = hh_alpham / (hh_alpham + hh_betam)

  ? Assignment rule here: hh_ninfinity = hh_alphan / (hh_alphan + hh_betan)
  hh_ninfinity = hh_alphan / (hh_alphan + hh_betan)

  ? Assignment rule here: hh_hinfinity = hh_alphah / (hh_alphah + hh_betah)
  hh_hinfinity = hh_alphah / (hh_alphah + hh_betah)

  ? Assignment rule here: hh_taun = 0.001 / (hh_alphan + hh_betan)
  hh_taun = 0.001 / (hh_alphan + hh_betan)

  ? Assignment rule here: INa = gNa * pow(hh_minfinity, 3) * h * (psin - RToverF * log(Nae / Nan))
  INa = gNa * pow(hh_minfinity, 3) * h * (psin - RToverF * log(Nae / Nan))

  ? Assignment rule here: IK = gK * pow(n, 4) * (psin - EK)
  IK = gK * pow(n, 4) * (psin - EK)

  ? Assignment rule here: ICa = gCa * pow(hh_mCa, 2) * (psin - ECa)
  ICa = gCa * pow(hh_mCa, 2) * (psin - ECa)

  ? Assignment rule here: ImAHP = gmAHP * Ca / (Ca + KD) * (psin - EK)
  ImAHP = gmAHP * Ca / (Ca + KD) * (psin - EK)

  ? Assignment rule here: Ipump = F * kpumpn * ATPn * (Nan - Nan0) * pow(1 + ATPn / Kmpump, -1)
  Ipump = F * kpumpn * ATPn * (Nan - Nan0) * pow(1 + ATPn / Kmpump, -1)

  ? Assignment rule here: Fout = F0 * pow(Vv / Vv0, 1 / alphav) + F0 * (tauv / Vv0) * pow(Vv / Vv0, -(1 / 2))
  Fout =  F0 * pow(Vv / Vv0, 1 / alphav) + F0 * (tauv / Vv0) * Vvp * pow( Vv / Vv0 , -(1 / 2))

  ? Assignment rule here: rce = Vcap / Ve
  rce = Vcap / Ve

  ? Assignment rule here: rcn = Vcap / Vn
  rcn = Vcap / Vn

  ? Assignment rule here: rcg = Vcap / Vg
  rcg = Vcap / Vg

  ? Assignment rule here: JGLCec = 0 - JGLCce
  JGLCec = 0 - JGLCce

  ? Assignment rule here: JGLCgc = 0 - JGLCcg
  JGLCgc = 0 - JGLCcg

  ? Assignment rule here: JGLCge = 0 - JGLCeg
  JGLCge = 0 - JGLCeg

  ? Assignment rule here: JGLCne = 0 - JGLCen
  JGLCne = 0 - JGLCen

  ? Assignment rule here: JLACce = 0 - JLACec
  JLACce = 0 - JLACec

  ? Assignment rule here: JLACgc = 0 - JLACcg
  JLACgc = 0 - JLACcg

  ? Assignment rule here: JLACge = 0 - JLACeg
  JLACge = 0 - JLACeg

  ? Assignment rule here: JLACne = 0 - JLACen
  JLACne = 0 - JLACen

  ? Assignment rule here: un = qAK * qAK + 4 * qAK * (A / ATPn - 1)
  un = qAK * qAK + 4 * qAK * (A / ATPn - 1)

  ? Assignment rule here: ug = qAK * qAK + 4 * qAK * (A / ATPg - 1)
  ug = qAK * qAK + 4 * qAK * (A / ATPg - 1)

  ? Assignment rule here: dAMPdATPn = -1 + qAK / 2 + -0.5 * pow(un, 0.5) + qAK * A / (ATPn * pow(un, 0.5))
  dAMPdATPn = -1 + qAK / 2 + -0.5 * pow(un, 0.5) + qAK * A / (ATPn * pow(un, 0.5))

  ? Assignment rule here: dAMPdATPg = -1 + qAK / 2 + -0.5 * pow(ug, 0.5) + qAK * A / (ATPg * pow(ug, 0.5))
  dAMPdATPg = -1 + qAK / 2 + -0.5 * pow(ug, 0.5) + qAK * A / (ATPg * pow(ug, 0.5))

  Igliapump=  1/3*p/(1+exp((25-Nag)/3))*1.0/(1+exp(3.5-Ke))
  Iglia= Gglia/(1+exp((18-Ke)/2.5))



  pio=Nae+Ke+Cle+Aminuse
  pii=Nan+Kn+Cln+Aminusn

  ?be sure to change this back
  ?volninfinity=voln0*(1.1029-0.1029*exp((pio-pii)/20))
  volninfinity=voln0

  ? glia surface to volume ratio is 10 to 20 Chao TI Rickmann M Wolff JR 2002 book by Volterra and Magistretti, tripartite synaptic transmission with glia
  ? my calculations of neuron SA to volume ratio are about 5, which agrees with Kubota 2011 nature scientific reports
  ? SCIENTIFIC REPORTS | 1 : 89 | DOI: 10.1038/srep00089

  p=rhomax/(1+exp((20-O2e)/3))?/gamma
  Ipumpnak=p/(1+exp((25-Nan)/3.0))*1.0/(1+exp(3.5-Ke))

  Gglia=Ggliamax/(1+exp(-(O2e-2.5)/0.2))

  volo=volo0+voln0-voln

  ecl=RToverF * log(Cln/Cle)
  Icil=SmVn / F *0.07*(-70-ecl) * 2.1
  Ikcc2=Ukcc2 * log(Kn * Cln/Ke/Cle )
  Inkcc1=Unkcc1 * (1/(1+exp(16-Ke))) * ( log(Kn * Cln/Ke/Cle )+log(Nan *Cln/Nae/Cle ))
  Ke = NKe/volo
  Kn = NKn/voln
  Nae = NNae/volo
  Nan = NNan/voln
  Cle = NCle/volo
  Cln = NCln/voln
  Cae = NCae/volo

  Nag=NNag/volg0
  GLCg=NGLCg/volg0
  GAPg=NGAPg/volg0
  GAPn=NGAPn/voln
  PEPg=NPEPg/volg0
  PEPn=NPEPn/voln
  PYRg=NPYRg/volg0
  PYRn=NPYRn/voln
  LACn=NLACn/voln
  LACg=NLACg/volg0
  NADHcyton=NNADHcyton/voln
  NADHcytog=NNADHcytog/volg0
  NADHmiton=NNADHmiton/voln
  NADHmitog=NNADHmitog/volg0
  ATPn=NATPn/voln
  ATPg=NATPg/volg0
  PCrg=NPCrg/volg0
  PCrn=NPCrn/voln
  O2n=NO2n/voln
  O2g=NO2g/volg0
  GLCe=NGLCe/volo
  LACe=NLACe/volo

  beta = voln/volo
  fo = 1.0/(1+exp(-(Obath-2.5)/0.2))
  fv = 1.0/(1+exp((-20+beta)/2))
  dslp = 0.25 *fo*fv ?was 0.25

  ?Bennett 2008 equations
  JpumpG1=VmaxG *(Cag *Cag /(Cag*Cag +Kpg * Kpg))
  JleakG=PlG *(1-Cag /(CagER))
  gating1= (IP3/(IP3+KI)* Cag/(Cag+Kact) *hIP3)
  JIP3=Jmax * gating1 *gating1 *gating1 *hIP3
  rhoG=Glu/(KGlu+Glu)

   pMusc=1/(1+exp((VhMusc-VMusc)/kMusc))
   iCaMusc=kappaMusc * (VMusc-VAveMusc)/(1-exp(0.075 * (VMusc-VAveMusc)))
   kMyosin1= kMyosin * pow(CaMusc, nMusc)

}

DERIVATIVE states {
  NNag' =1/1000*(  JleakNag - 3 * Jpumpg + Jstimg + 3 * JGLUeg)*volg0
  NGLCn' =1/1000*( JGLCen - JHKPFKn)*voln
  NGLCg' =1/1000*( JGLCcg + JGLCeg - JHKPFKg)*volg0
  NGAPg' =1/1000*( 2 * JHKPFKg - JPGKg)*volg0
  NGAPn' =1/1000*( 2 * JHKPFKn - JPGKn)*voln
  NPEPg' =1/1000*( JPGKg - JPKg)*volg0
  NPEPn' =1/1000*( JPGKn - JPKn)*voln
  NPYRg' = 1/1000*(JPKg - JLDHg - Jmitoing)*volg0
  NPYRn' =1/1000*( JPKn - JLDHn - Jmitoinn)*voln
  NLACn' =1/1000*( JLDHn - JLACne)*voln
  NLACg' =1/1000*( JLDHg - JLACge - JLACgc)*volg0
  NNADHcyton' =1/1000*( pow(1 - zeta, -1) * (JPGKn - JLDHn - Jshuttlen))*voln
  NNADHcytog' =1/1000*( pow(1 - zeta, -1) * (JPGKg - JLDHg - Jshuttleg))*volg0
  NNADHmiton' =1/1000*( pow(zeta, -1) * (4 * Jmitoinn - Jmitooutn + Jshuttlen))*voln
  NNADHmitog' = 1/1000*(pow(zeta, -1) * (4 * Jmitoing - Jmitooutg + Jshuttleg))*volg0
  NATPn' = 1/1000*((-2 * JHKPFKn + JPGKn + JPKn - JATPasesn - 1 * Jpumpn + 3.6 * Jmitooutn + JCKn) * pow(1 - dAMPdATPn, -1))*voln
  NATPg' = 1/1000*((-2 * JHKPFKg + JPGKg + JPKg - JATPasesg - 1.75 * Jpumpg + 0.75 * vPumpg0+3.6 * Jmitooutg + JCKg) * pow(1 - dAMPdATPg, -1))*volg0
  NPCrg' = 1/1000*(-JCKg)*volg0
  NPCrn' = 1/1000*(-JCKn)*voln
  NO2n' = 1/1000*(JO2mcn - 0.625 * Jmitooutn)*voln
  NO2g' = 1/1000*(JO2mcg - 0.625 * Jmitooutg)*volg0
  NGLCe' =1/1000*(  JGLCce - JGLCeg * volg0/volo - JGLCen* voln0/volo )*volo
  NLACe' =  1/1000*(JLACne * voln0/volo +  JLACge * volg0/volo -  JLACec)*volo

  O2c' = 1/1000*(JO2c - 1 / rcn * JO2mcn - 1 / rcg * JO2mcg)
  GLCc' =1/1000*( JGLCc - 1 / rce * JGLCce - 1 / rcg * JGLCcg)
  LACc' =1/1000*(JLACc + 1 / rce * JLACec + 1 / rcg * JLACgc)

  Vv' =0? 1/1000*((Fin -  Fout))
  Hb' =  1/1000*(Fin * (O2a - O2meanc) * Vv' /(Fout+Vv'))
  psin' =0?1/1000*( 1 / Cm * (-IL - INa - IK - ICa - ImAHP - Ipumpnak + Isyn))
  h' = 0?1/1000*(phih / hh_tauh * (hh_hinfinity - h))
  n' = 0?1/1000*( phin / hh_taun * (hh_ninfinity - n))
  Ca' =1/1000*( -SmVn * ICa / F - 1 / tauCa * (Ca - Ca0))

  ?Vvp should be the same as Vv', for constant Fin. Vv'=Fin-Fout, so Vv'=Fout'. We worked thought this and solved for Vv'', which is set to Vvp in the following equation
  ?Vvp'=(F0 *Vv' *(-2 *Vv0 *pow(Vv/Vv0,(1/2 + 1/alphav)) + alphav* tauv* Vv'))/(2 *alphav *Vv *(F0 *tauv + Vv0* pow(Vv/Vv0,1/2))) )
  Vvp'=0?1/1000*((2 *alphav *Finprime *Vv0 *Vv *pow((Vv/Vv0),1/2) -  2 *F0 *Vv0 *pow((Vv/Vv0),(1/2 + 1/alphav)) *Vv' + alphav *F0 *tauv *Vv' *Vv')/(2 *alphav *Vv *(F0 * tauv + Vv0 * pow(Vv/Vv0,1/2) ) ))

  voln'=(volninfinity-voln)/250

  NKe'=1*(1/1000*(  Ik -2*Ipumpnak+ Ikcc2+Inkcc1)*voln0 +1/1000*(-Iglia-2*Igliapump)*volg0 -1/1000* dslp*(NKe  -Kbath*volo))
  NKn'=-1/1000*(  Ik -2*Ipumpnak+ Ikcc2+Inkcc1)*voln0
  ?Jleaknan 2.36 to 2.361
?  NNan' = 1/1000*( 2.355 *JleakNan - 3 * Jpumpn + Jstimn -Inkcc1-3*Ipumpnak+INa)*voln0
?  NNae'= 1*(-1/1000*( 2.355 * JleakNan - 3 * Jpumpn + Jstimn -Inkcc1-3*Ipumpnak+INa)*voln0-1/1000*(  JleakNag - 3 * Jpumpg + Jstimg + 3 * JGLUeg)*volg0)

  NNan' = 1/1000*(  2.355 *JleakNan -  3 * Jpumpn +  Jstimn -Inkcc1-3*Ipumpnak+0.310880 *INa)*voln0
  NNae'= 1*(-1/1000*(  2.355 * JleakNan - 3 * Jpumpn +  Jstimn -Inkcc1-3*Ipumpnak+0.310880 *INa)*voln0-1/1000*(  JleakNag - 3 * Jpumpg + Jstimg + 3 * JGLUeg)*volg0)

  NCae'=0
  NCle'=1/1000*( -Icil+Ikcc2+2*Inkcc1)*volo
  NCln'= -1/1000*( -Icil+Ikcc2+2*Inkcc1)*voln0

  ?new eqns for Bennett 2008
  Glu'=-kGlutamateDecay * Glu
  mGluR_inactive'=- kmGluR_forward * Glu* mGluR_inactive + kmGluR_backward * mGluR_active
  mGluR_active'= kmGluR_forward * Glu* mGluR_inactive - kmGluR_backward * mGluR_active
  Ginactive' = -kplusG*( Gdelta + mGluR_active ) * Ginactive + kminusG * Gactive
  Gactive'= kplusG*( Gdelta + mGluR_active ) * Ginactive - kminusG * Gactive
  IP3'= Grh * Gactive - kdeg_IP3 * IP3
  Cag'=BcytoG *(JIP3-  JpumpG1+JleakG)
  hIP3'=konG *(Kinh-(Cag + Kinh) * hIP3)
  EET'=VEET * (Cag-Cagmin) -0.001 * EET

   Myosin'=-kMyosin1 * Myosin + kMyosin2 * MyosinP
   MyosinP'= kMyosin1 * Myosin - kMyosin2 * MyosinP - kMyosin3 * MyosinP + kMyosin4 * ActinP
   ActinP'=  kMyosin3 * MyosinP - kMyosin4 * ActinP
   CaMusc' = alphaMusc * pMusc * iCaMusc- betaMusc*(CaMusc - CaMusc0)
   VMusc'=  -gammaEET * EET + (-30-VMusc)/15
}

VERBATIM
/** not executed in coreneuron and hence need empty stubs only */
static void bbcore_write(double* x, int* d, int* xx, int* offset, _threadargsproto_) {
}
static void bbcore_read(double* x, int* d, int* xx, int* offset, _threadargsproto_) {
}
ENDVERBATIM



