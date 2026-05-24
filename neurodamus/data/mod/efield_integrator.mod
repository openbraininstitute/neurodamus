NEURON {
    POINT_PROCESS EFieldIntegrator
    POINTER e_ext
    POINTER phase, frequency, X, Y, Z, delay, duration, ramp_up, ramp_down
    RANGE enabled  : set when e_ext pointer is assigned, this prevents mcomplex calculation to access unassigned e_ext reference
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
    enabled
}

INITIAL {
    if( enabled ) {
        e_ext = 0
    }

VERBATIM
#ifndef CORENEURON_BUILD
    if( !_p_X ) {
        _p_X = vector_new1(0);
        _p_Y = vector_new1(0);
        _p_Z = vector_new1(0);
        _p_phase = vector_new1(0);
        _p_frequency = vector_new1(0);
        _p_delay = vector_new1(0);
        _p_duration = vector_new1(0);
        _p_ramp_up = vector_new1(0);
        _p_ramp_down = vector_new1(0);
    }
#endif
ENDVERBATIM
}

PROCEDURE add_electrode_source() {
    : receive data elements for an ElectrodeSource
    : vectors for n fields: delay, duration, ramp_up_time, ramp_down_time, x, y, z, freq, phase,

VERBATIM
#ifndef CORENEURON_BUILD
    auto* argdelay = vector_arg(1);
    auto* argduration = vector_arg(2);
    auto* argramp_up = vector_arg(3);
    auto* argramp_down = vector_arg(4);
    auto* argX = vector_arg(5);
    auto* argY = vector_arg(6);
    auto* argZ = vector_arg(7);
    auto* argphase = vector_arg(8);
    auto* argfreq = vector_arg(9);

    int size = vector_capacity(argX);
    _p_X = vector_new1(size);
    _p_Y = vector_new1(size);
    _p_Z = vector_new1(size);
    _p_phase = vector_new1(size);
    _p_frequency = vector_new1(size);
    _p_delay = vector_new1(size);
    _p_duration = vector_new1(size);
    _p_ramp_up = vector_new1(size);
    _p_ramp_down = vector_new1(size);

    auto *vX = *reinterpret_cast<IvocVect**>(&_p_X);
    auto *vY = *reinterpret_cast<IvocVect**>(&_p_Y);
    auto *vZ = *reinterpret_cast<IvocVect**>(&_p_Z);
    auto *vphase = *reinterpret_cast<IvocVect**>(&_p_phase);
    auto *vfreq = *reinterpret_cast<IvocVect**>(&_p_frequency);
    auto *vdelay = *reinterpret_cast<IvocVect**>(&_p_delay);
    auto *vduration = *reinterpret_cast<IvocVect**>(&_p_duration);
    auto *vramp_up = *reinterpret_cast<IvocVect**>(&_p_ramp_up);
    auto *vramp_down = *reinterpret_cast<IvocVect**>(&_p_ramp_down);

    for( int i=0; i<size; i++ ) {
        vector_vec(vX)[i] = vector_vec(argX)[i];
        vector_vec(vY)[i] = vector_vec(argY)[i];
        vector_vec(vZ)[i] = vector_vec(argZ)[i];
        vector_vec(vphase)[i] = vector_vec(argphase)[i];
        vector_vec(vfreq)[i] = vector_vec(argfreq)[i];
        vector_vec(vdelay)[i] = vector_vec(argdelay)[i];
        vector_vec(vduration)[i] = vector_vec(argduration)[i];
        vector_vec(vramp_up)[i] = vector_vec(argramp_up)[i];
        vector_vec(vramp_down)[i] = vector_vec(argramp_down)[i];
    }
#endif
ENDVERBATIM
}

FUNCTION get_potential_amplitude(i) {
    : get potential amplitude for ith electrod field, in mV
VERBATIM
    int idx = (int)_li;
    auto *vX = *reinterpret_cast<IvocVect**>(&_p_X);
    auto *vY = *reinterpret_cast<IvocVect**>(&_p_Y);
    auto *vZ = *reinterpret_cast<IvocVect**>(&_p_Z);
    _lget_potential_amplitude = 1e3 * (vector_vec(vX)[idx] + vector_vec(vY)[idx] + vector_vec(vZ)[idx]);
ENDVERBATIM
}

BEFORE BREAKPOINT {
    LOCAL efield_accum

VERBATIM
#ifndef CORENEURON_BUILD
    int i, size;
    double ramp_factor=1;
    double cur_delay, cur_duration, cur_ramp_up, cur_ramp_down;
    _lefield_accum = 0;

    auto *vX = *reinterpret_cast<IvocVect**>(&_p_X);
    auto *vY = *reinterpret_cast<IvocVect**>(&_p_Y);
    auto *vZ = *reinterpret_cast<IvocVect**>(&_p_Z);
    auto *vphase = *reinterpret_cast<IvocVect**>(&_p_phase);
    auto *vfreq = *reinterpret_cast<IvocVect**>(&_p_frequency);
    auto *vdelay = *reinterpret_cast<IvocVect**>(&_p_delay);
    auto *vduration = *reinterpret_cast<IvocVect**>(&_p_duration);
    auto *vramp_up = *reinterpret_cast<IvocVect**>(&_p_ramp_up);
    auto *vramp_down = *reinterpret_cast<IvocVect**>(&_p_ramp_down);
    size = vector_capacity(vX);

    for( i=0; i<size; i++ ) {
        cur_delay = vector_vec(vdelay)[i];
        cur_duration = vector_vec(vduration)[i];
        cur_ramp_up = vector_vec(vramp_up)[i];
        cur_ramp_down = vector_vec(vramp_down)[i];
        if( cur_delay < t && t < cur_delay + cur_duration + cur_ramp_up + cur_ramp_down ) {
            if( cur_delay < t && t < cur_delay + cur_ramp_up ) {
                ramp_factor = (t-cur_delay) / cur_ramp_up;
            }
            if( cur_delay + cur_ramp_up + cur_duration < t && t < cur_delay + cur_duration + cur_ramp_up + cur_ramp_down ) {
                ramp_factor = 1 - (t - (cur_delay + cur_ramp_up + cur_duration)) / cur_ramp_down;
            }
            double wavefactor = cos(2 * PI * vector_vec(vfreq)[i] / 1000 * (t-cur_delay) + vector_vec(vphase)[i] );
            _lefield_accum += ramp_factor * get_potential_amplitude(i) * wavefactor;
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

