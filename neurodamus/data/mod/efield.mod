NEURON {
    POINT_PROCESS efield
    POINTER e_ext
    RANGE phase, frequency, X, Y, Z
}

PARAMETER {
    phase = 0.
    frequency = 1 (Hz)
    X = 0.
    Y = 0.
    Z = 0.
}

ASSIGNED {
    e_ext (mV)
}

INITIAL {
    e_ext = 0
}

BEFORE BREAKPOINT {
    LOCAL c
    c = cos(2 * 3.141592654 * frequency / 1000 * t + phase)
    e_ext = (X*c + Y*c + Z*c) * 1000
}
