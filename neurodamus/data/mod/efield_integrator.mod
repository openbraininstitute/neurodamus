NEURON {
    POINT_PROCESS EFieldIntegrator
    POINTER e_ext
    :POINTER delay, duration, ramp_up, ramp_down, nfields : TODO: vectors will have size = nElectrodeSource
    POINTER phase, frequency, X, Y, Z           : TODO: will have size = sum(nFields for each ElectrodeSource)
    RANGE delay, duration, ramp_up, ramp_down
    RANGE displacementX, displacementY, displacementZ
    RANGE enabled
}

PARAMETER {
}

ASSIGNED {
    e_ext (mV)
    delay
    duration
    ramp_up
    ramp_down
    nfields
    phase
    frequency
    X
    Y
    Z
    displacementX
    displacementY
    displacementZ
    enabled
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

    //fprintf( stderr, "add electrode source [%lf %lf]\n", delay, delay + duration );

    // copy data into locally managed vectors rather than storing pointer
    //  the idea is to put all data into a single vector and track offsets

    int i, size;
    auto* argX = vector_arg(5);
    auto* argY = vector_arg(6);
    auto* argZ = vector_arg(7);
    auto* argphase = vector_arg(8);
    auto* argfreq = vector_arg(9);

    size = vector_capacity(argX);
    if( _p_phase != nullptr ) {
        fprintf( stderr, "support for multiple ElectrodeSource info not implemented yet\n" );
        //vector_resize( vphase, vector_capacity(vec) );
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
    
        for( i=0; i<size; i++ ) {
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
    LOCAL factor, efield_accum

    : for each electrode source (pending)
    :    for each field
    :       in time window? compute contribution factor and apply to segment's e_ref via pointer
VERBATIM
#ifndef CORENEURON_BUILD
    int i, size;
    double rufactor=1, rdfactor=1;
    _lefield_accum = 0;

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
       if( delay + ramp_up + duration < t && t < delay+duration+ramp_up+ramp_down ) {
           rdfactor = 1 - (t-(delay+ramp_up+duration)) / ramp_down;
       }

       size = vector_capacity(vX);
       for( i=0; i<size; i++ ) {
           _lfactor = cos(2 * 3.141592654 * vector_vec(vfreq)[i] / 1000 * (t-delay) + vector_vec(vphase)[i] );
	   _lefield_accum += 1e3 * rufactor * rdfactor * (displacementX * vector_vec(vX)[i]*_lfactor + displacementY * vector_vec(vY)[i]*_lfactor + displacementZ * vector_vec(vZ)[i]*_lfactor);
       }
    } else {
      //fprintf( stderr, "outside time window [%lf, %lf]\n", delay, delay+duration );
    }
    
/* Adapting to multiple ElectrodeSources
    int i, size;
    double delay_item, dur_item;
    auto *vdelay = *reinterpret_cast<IvocVect**>(&delay);
    auto *vdur = *reinterpret_cast<IvocVect**>(&duration);

    size = vector_capacity(vdelay);
    for( i=0; i<size; i++ ) {
        delay_item = vector_vec(vdelay)[i];
        dur_item = vector_vec(vdur)[i];
        if( delay_item < t && t < delay_item + dur_item ) {
            // for each field
                //as above
            // end for each field
        }
    }
*/
    //fprintf( stderr, "t %lf %lf\n", t, _lefield_accum );
#endif
ENDVERBATIM
    if( enabled ) {
        e_ext = efield_accum
    }
}

: currently, extracellular stimulus is not supported by coreneuron, so will
: skip using BBCOREPOINTER and bbcore_write/bbcore_read

