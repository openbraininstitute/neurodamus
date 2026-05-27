NEURON {
    POINT_PROCESS EFieldIntegrator
    POINTER e_ext
    POINTER phase, frequency, peak_potential, delay, duration, ramp_up, ramp_down
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
    peak_potential
    enabled
}

INITIAL {
    if( enabled ) {
        e_ext = 0
    }

VERBATIM
#ifndef CORENEURON_BUILD
    if( !_p_peak_potential ) {
        _p_peak_potential = vector_new1(0);
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
    : vectors for n fields: delay, duration, ramp_up_time, ramp_down_time, peak_potential, freq, phase

VERBATIM
#ifndef CORENEURON_BUILD
    auto* argdelay = vector_arg(1);
    auto* argduration = vector_arg(2);
    auto* argramp_up = vector_arg(3);
    auto* argramp_down = vector_arg(4);
    auto* argpeak_potential = vector_arg(5);
    auto* argphase = vector_arg(6);
    auto* argfreq = vector_arg(7);

    int size = vector_capacity(argpeak_potential);
    _p_peak_potential = vector_new1(size);
    _p_phase = vector_new1(size);
    _p_frequency = vector_new1(size);
    _p_delay = vector_new1(size);
    _p_duration = vector_new1(size);
    _p_ramp_up = vector_new1(size);
    _p_ramp_down = vector_new1(size);

    auto *vpeak = *reinterpret_cast<IvocVect**>(&_p_peak_potential);
    auto *vphase = *reinterpret_cast<IvocVect**>(&_p_phase);
    auto *vfreq = *reinterpret_cast<IvocVect**>(&_p_frequency);
    auto *vdelay = *reinterpret_cast<IvocVect**>(&_p_delay);
    auto *vduration = *reinterpret_cast<IvocVect**>(&_p_duration);
    auto *vramp_up = *reinterpret_cast<IvocVect**>(&_p_ramp_up);
    auto *vramp_down = *reinterpret_cast<IvocVect**>(&_p_ramp_down);

    for( int i=0; i<size; i++ ) {
        vector_vec(vpeak)[i] = vector_vec(argpeak_potential)[i];
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

FUNCTION get_peak_potential(i) {
    : get peak potential for ith electric field, in mV
    : this is the dot product E dot d (field * displacement) pre-computed on the Python side
VERBATIM
    int idx = (int)_li;
    auto *vpeak = *reinterpret_cast<IvocVect**>(&_p_peak_potential);
    _lget_peak_potential = vector_vec(vpeak)[idx];
ENDVERBATIM
}

BEFORE BREAKPOINT {
    LOCAL efield_accum

VERBATIM
#ifndef CORENEURON_BUILD
    _lefield_accum = 0;

    auto *vphase = *reinterpret_cast<IvocVect**>(&_p_phase);
    auto *vfreq = *reinterpret_cast<IvocVect**>(&_p_frequency);
    auto *vdelay = *reinterpret_cast<IvocVect**>(&_p_delay);
    auto *vduration = *reinterpret_cast<IvocVect**>(&_p_duration);
    auto *vramp_up = *reinterpret_cast<IvocVect**>(&_p_ramp_up);
    auto *vramp_down = *reinterpret_cast<IvocVect**>(&_p_ramp_down);

    for( int i=0; i < vector_capacity(vdelay); i++ ) {
        double ramp_factor = 1;
        double cur_delay = vector_vec(vdelay)[i];
        double cur_duration = vector_vec(vduration)[i];
        double cur_ramp_up = vector_vec(vramp_up)[i];
        double cur_ramp_down = vector_vec(vramp_down)[i];
        if( cur_delay < t && t < cur_delay + cur_duration + cur_ramp_up + cur_ramp_down ) {
            if( t < cur_delay + cur_ramp_up ) {
                ramp_factor = (t-cur_delay) / cur_ramp_up;
            }
            if( cur_delay + cur_ramp_up + cur_duration < t ) {
                ramp_factor = 1 - (t - (cur_delay + cur_ramp_up + cur_duration)) / cur_ramp_down;
            }
            double wavefactor = cos(2 * PI * vector_vec(vfreq)[i] / 1000 * (t-cur_delay) + vector_vec(vphase)[i]);
            _lefield_accum += ramp_factor * get_peak_potential(i) * wavefactor;
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
