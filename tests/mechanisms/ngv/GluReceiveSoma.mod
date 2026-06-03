COMMENT
/**
 * @file GlutReceiveSoma.mod
 * @brief
 * @author keller
 * @date 2018-06-17
 * @remark Copyright © BBP/EPFL 2005-2011; All rights reserved. Do not distribute without further notice.
 */
ENDCOMMENT

TITLE Glutamate Receive Object for Metabolism Model in Soma


COMMENT
make a glutamate receive function
ENDCOMMENT


NEURON {
    THREADSAFE
    POINT_PROCESS GlutReceiveSoma
    RANGE glut
}


PARAMETER {
    glut
}


ASSIGNED {
    stimtime
}


STATE {
}


INITIAL{
    stimtime=-1000
    glut=0
}


BREAKPOINT {
    if (glut>0 && t>stimtime+1){
        glut=0
    }
}


PROCEDURE state() {
}


NET_RECEIVE (weight){
    INITIAL{
    }
    glut=glut+1
    stimtime=t
:VERBATIM
:        printf("SOMA RECEIVE\n");
:ENDVERBATIM

}

