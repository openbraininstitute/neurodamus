NEURON {
    POINT_PROCESS EFieldIntegrator
    POINTER e_ext
    :POINTER delay, duration, ramp_up, ramp_down : TODO: add nfields vector and vectors will have size = nElectrodeSource
    POINTER phase, frequency, X, Y, Z           : TODO: will have size = sum(nFields for each ElectrodeSource)
    RANGE delay, duration, ramp_up, ramp_down
    RANGE displacementX, displacementY, displacementZ
    RANGE enabled
}

PARAMETER {
}

UNITS {
    PI = (pi) (1)
}

ASSIGNED {
    e_ext (mV)
    delay
    duration
    ramp_up
    ramp_down
    phase
    frequency
    X
    Y
    Z
    displacementX
    displacementY
    displacementZ
    enabled : set when pointer is assigned, this prevents mcomplex calculation to access unassigned e_ext reference
}

INITIAL {
    if( enabled ) {
        e_ext = 0
    }
}

PROCEDURE add_electrode_source() {
    : receive data elements for an ElectrodeSource
    : delay, duration, ramp_up_time, ramp_down_time
    : vectors for n fields: x, y, z, freq, phase,

VERBATIM
#ifndef CORENEURON_BUILD
    delay = *getarg(1);
    duration = *getarg(2);
    ramp_up = *getarg(3);
    ramp_down = *getarg(4);

    auto* argX = vector_arg(5);
    auto* argY = vector_arg(6);
    auto* argZ = vector_arg(7);
    auto* argphase = vector_arg(8);
    auto* argfreq = vector_arg(9);

    int size = vector_capacity(argX);
    if( _p_phase != nullptr ) {
        fprintf( stderr, "support for multiple ElectrodeSource info not implemented yet\n" );
        //TODO: vector_resize( vphase, vector_capacity(vec) );
    } else {
        _p_X = vector_new1(size);
        _p_Y = vector_new1(size);
        _p_Z = vector_new1(size);
        _p_phase = vector_new1(size);
        _p_frequency = vector_new1(size);

        auto *vX = *reinterpret_cast<IvocVect**>(&_p_X);
        auto *vY = *reinterpret_cast<IvocVect**>(&_p_Y);
        auto *vZ = *reinterpret_cast<IvocVect**>(&_p_Z);
        auto *vphase = *reinterpret_cast<IvocVect**>(&_p_phase);
        auto *vfreq = *reinterpret_cast<IvocVect**>(&_p_frequency);

        for( int i=0; i<size; i++ ) {
            vector_vec(vX)[i] = vector_vec(argX)[i];
            vector_vec(vY)[i] = vector_vec(argY)[i];
            vector_vec(vZ)[i] = vector_vec(argZ)[i];
            vector_vec(vphase)[i] = vector_vec(argphase)[i];
            vector_vec(vfreq)[i] = vector_vec(argfreq)[i];
        }
    }
#endif
ENDVERBATIM
}

BEFORE BREAKPOINT {
    LOCAL efield_accum

    : for each electrode source (pending)
    :    for each field
    :       in time window? compute contribution factor and apply to segment's e_ref via pointer
VERBATIM
#ifndef CORENEURON_BUILD
    int i, size;
    double rufactor=1, rdfactor=1;
    _lefield_accum = 0;

/* TODO: Adapting to multiple ElectrodeSources */
    if( delay < t && t < delay+duration+ramp_up+ramp_down ) {
       auto *vX = *reinterpret_cast<IvocVect**>(&_p_X);
       auto *vY = *reinterpret_cast<IvocVect**>(&_p_Y);
       auto *vZ = *reinterpret_cast<IvocVect**>(&_p_Z);
       auto *vphase = *reinterpret_cast<IvocVect**>(&_p_phase);
       auto *vfreq = *reinterpret_cast<IvocVect**>(&_p_frequency);

       // TODO: if in a ramp up/down window, then scale further
       if( delay < t && t < delay+ramp_up ) {
           rufactor = (t-delay) / ramp_up;
       }
       if( delay+ramp_up+duration < t && t < delay+duration+ramp_up+ramp_down ) {
           rdfactor = 1 - (t-(delay+ramp_up+duration)) / ramp_down;
       }

       size = vector_capacity(vX);
       for( i=0; i<size; i++ ) {
           double wavefactor = cos(2 * PI * vector_vec(vfreq)[i] / 1000 * (t-delay) + vector_vec(vphase)[i] );
	   _lefield_accum += 1e3 * rufactor * rdfactor * (displacementX * vector_vec(vX)[i] * wavefactor + displacementY * vector_vec(vY)[i] * wavefactor + displacementZ * vector_vec(vZ)[i] * wavefactor);
       }
    }
#endif
ENDVERBATIM
    if( enabled ) {
        e_ext = efield_accum
    }
}

: currently, extracellular stimulus is not supported by coreneuron, so will
: skip using BBCOREPOINTER and bbcore_write/bbcore_read

